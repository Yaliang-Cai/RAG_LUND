#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RUN_ROOT = (
    "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421"
)
DEFAULT_SHARED_WORKSPACE_ID = "docbench_shared_graphbm25_20260421_v0_v1_v2"
DEFAULT_SURGE_WORKSPACE_ID = "surge_fast_graphbm25_20260421_v0_v1_v2"
DEFAULT_OUTPUT_ROOT = "evaluate_local/retrieval_ablation_runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


def _with_unified_retrieval(item: dict[str, Any]) -> dict[str, Any]:
    retrieval_mode = str(item.get("retrieval_mode", "dense"))
    return {
        **item,
        "retrieval_mode": retrieval_mode,
        "entity_retrieval_mode": retrieval_mode,
        "chunk_retrieval_mode": retrieval_mode,
    }


def _base_hybrid_experiment(*, task: str, name: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task": task,
        "name": name,
        "query_mode": "hybrid",
        "keyword_fanout_mode": "joined",
        "retrieval_mode": "dense",
        "exclude_synonym_edges": True,
        "kg_chunk_selection_source": "truncated",
        "answer_context_mode": "kg_prompt",
    }
    if task == "surge":
        base["kg_chunk_selection_source"] = "untruncated"
        base.pop("answer_context_mode", None)
    return _with_unified_retrieval(base)


def _ppr_experiment(
    *,
    task: str,
    name: str,
    keyword_fanout_mode: str,
    retrieval_mode: str,
) -> dict[str, Any]:
    return _with_unified_retrieval(
        {
            "task": task,
            "name": name,
            "query_mode": "ppr",
            "keyword_fanout_mode": keyword_fanout_mode,
            "retrieval_mode": retrieval_mode,
            "exclude_synonym_edges": False,
            "answer_context_mode": "chunk_only_prompt",
        }
    )


def build_reduced_experiment_matrix(task: str) -> list[dict[str, Any]]:
    normalized_task = str(task).strip().lower()
    if normalized_task in {"docbench", "shared"}:
        task_name = "shared"
        baseline = _base_hybrid_experiment(task=task_name, name="baseline_kg")
        experiments = [
            baseline,
            {
                **baseline,
                "name": "per_keyword_kg",
                "keyword_fanout_mode": "per_keyword_rrf",
            },
            _with_unified_retrieval(
                {
                    **baseline,
                    "name": "retrieval_hybrid_kg",
                    "retrieval_mode": "hybrid",
                }
            ),
            {
                **baseline,
                "name": "untruncated_kg",
                "kg_chunk_selection_source": "untruncated",
            },
            {
                **baseline,
                "name": "baseline_chunk_only",
                "answer_context_mode": "chunk_only_prompt",
            },
            {
                **baseline,
                "name": "untruncated_chunk_only",
                "kg_chunk_selection_source": "untruncated",
                "answer_context_mode": "chunk_only_prompt",
            },
            _ppr_experiment(
                task=task_name,
                name="ppr_dense",
                keyword_fanout_mode="joined",
                retrieval_mode="dense",
            ),
            _ppr_experiment(
                task=task_name,
                name="ppr_hybrid_per_keyword",
                keyword_fanout_mode="per_keyword_rrf",
                retrieval_mode="hybrid",
            ),
        ]
        return experiments

    if normalized_task == "surge":
        baseline = _base_hybrid_experiment(task="surge", name="baseline")
        return [
            baseline,
            {
                **baseline,
                "name": "per_keyword",
                "keyword_fanout_mode": "per_keyword_rrf",
            },
            _with_unified_retrieval(
                {
                    **baseline,
                    "name": "retrieval_hybrid",
                    "retrieval_mode": "hybrid",
                }
            ),
            _ppr_experiment(
                task="surge",
                name="ppr_dense",
                keyword_fanout_mode="joined",
                retrieval_mode="dense",
            ),
            _ppr_experiment(
                task="surge",
                name="ppr_hybrid_per_keyword",
                keyword_fanout_mode="per_keyword_rrf",
                retrieval_mode="hybrid",
            ),
        ]

    raise ValueError(f"Unknown reduced retrieval task: {task!r}")


def build_full_experiment_matrix(
    *,
    query_modes: list[str],
    keyword_fanout_modes: list[str],
    retrieval_modes: list[str],
    exclude_synonym_edges_values: list[bool],
    kg_chunk_selection_sources: list[str],
    answer_context_modes: list[str],
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for query_mode in query_modes:
        if query_mode == "ppr_local":
            continue
        for keyword_fanout_mode in keyword_fanout_modes:
            for retrieval_mode in retrieval_modes:
                for exclude_synonym_edges in exclude_synonym_edges_values:
                    if query_mode == "ppr":
                        experiments.append(
                            _with_unified_retrieval(
                                {
                                    "query_mode": query_mode,
                                    "keyword_fanout_mode": keyword_fanout_mode,
                                    "retrieval_mode": retrieval_mode,
                                    "exclude_synonym_edges": bool(exclude_synonym_edges),
                                    "answer_context_mode": "chunk_only_prompt",
                                }
                            )
                        )
                    else:
                        for kg_chunk_selection_source in kg_chunk_selection_sources:
                            for answer_context_mode in answer_context_modes:
                                experiments.append(
                                    _with_unified_retrieval(
                                        {
                                            "query_mode": query_mode,
                                            "keyword_fanout_mode": keyword_fanout_mode,
                                            "retrieval_mode": retrieval_mode,
                                            "exclude_synonym_edges": bool(exclude_synonym_edges),
                                            "kg_chunk_selection_source": kg_chunk_selection_source,
                                            "answer_context_mode": answer_context_mode,
                                        }
                                    )
                                )
    return experiments


def build_experiment_matrix(**kwargs: Any) -> list[dict[str, Any]]:
    """Backward-compatible alias for the old cartesian-product helper."""
    if "retrieval_modes" not in kwargs:
        entity_modes = kwargs.pop("entity_retrieval_modes", None)
        chunk_modes = kwargs.pop("chunk_retrieval_modes", None)
        kwargs["retrieval_modes"] = entity_modes or chunk_modes or ["dense", "hybrid"]
    kwargs.setdefault("kg_chunk_selection_sources", ["truncated", "untruncated"])
    return build_full_experiment_matrix(**kwargs)


def _experiment_name(item: dict[str, Any]) -> str:
    if item.get("name"):
        return str(item["name"])
    parts = [
        str(item["query_mode"]),
        str(item["keyword_fanout_mode"]),
        f"ret-{item['retrieval_mode']}",
        f"syn-{int(bool(item['exclude_synonym_edges']))}",
    ]
    if "kg_chunk_selection_source" in item:
        parts.append(str(item["kg_chunk_selection_source"]))
    if item["query_mode"] != "ppr":
        parts.append(str(item["answer_context_mode"]))
    return "__".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval-focused ablations on existing graphbm25_20260421 workspaces."
    )
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--shared-workspace-id", default=DEFAULT_SHARED_WORKSPACE_ID)
    parser.add_argument("--surge-workspace-id", default=DEFAULT_SURGE_WORKSPACE_ID)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--tasks", choices=["both", "shared", "surge"], default="both")
    parser.add_argument("--matrix-mode", choices=["reduced", "full"], default="reduced")
    parser.add_argument("--query-modes", default="hybrid,ppr")
    parser.add_argument("--keyword-fanout-modes", default="joined,per_keyword_rrf")
    parser.add_argument("--retrieval-modes", default="dense,hybrid")
    parser.add_argument("--entity-retrieval-modes", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-retrieval-modes", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--answer-context-modes", default="kg_prompt,chunk_only_prompt")
    parser.add_argument("--kg-chunk-selection-sources", default="truncated,untruncated")
    parser.add_argument("--exclude-synonym-edges-values", default="true,false")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--shared-chunk-top-k", type=int, default=20)
    parser.add_argument("--surge-chunk-top-k", type=int, default=0)
    parser.add_argument("--chunk-top-k", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-total-tokens", type=int, default=45000)
    parser.add_argument("--recognition-top-k", type=int, default=20)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=49)
    parser.add_argument("--max-async-ingest", type=int, default=4)
    parser.add_argument("--max-async-generate", type=int, default=6)
    parser.add_argument("--max-async-judge", type=int, default=32)
    parser.add_argument("--k-list", default="5,10,20,30,50")
    parser.add_argument("--survey-k-list", default="50,100,200,500")
    parser.add_argument(
        "--bypass-query-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bypass-keywords-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _bool_tokens(raw: str) -> list[bool]:
    values: list[bool] = []
    for token in _parse_csv(raw):
        lowered = token.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            values.append(True)
        elif lowered in {"0", "false", "no", "n", "off"}:
            values.append(False)
    return values or [True, False]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_command(*, command: list[str], cwd: Path, env: dict[str, str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# command\n")
        f.write(" ".join(command) + "\n\n")
        f.flush()
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode)


def _build_workspace_env(base_env: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT),
            str(PROJECT_ROOT.parent / "lightrag"),
            str(env.get("PYTHONPATH", "")).strip(),
        ]
    ).strip(os.pathsep)
    env["DOCBENCH_SHARED_WORKING_DIR_ROOT"] = str(args.run_root)
    env["SURGE_FAST_RAG_STORAGE_DIR"] = str(args.run_root)
    return env


def _shared_command(args: argparse.Namespace, experiment: dict[str, Any], output_dir: Path) -> list[str]:
    cmd = [
        args.python_exe,
        "-m",
        "evaluate_local.DocBench.evaluate_shared",
        "--mode",
        "generate",
        "--start_id",
        str(args.start_id),
        "--end_id",
        str(args.end_id),
        "--shared_workspace_id",
        args.shared_workspace_id,
        "--raganything_eval_setup",
        "--max_async_ingest",
        str(args.max_async_ingest),
        "--max_async_generate",
        str(args.max_async_generate),
        "--max_async_judge",
        str(args.max_async_judge),
        "--query_mode",
        str(experiment["query_mode"]),
        "--recognition_top_k",
        str(args.recognition_top_k),
        "--max_total_tokens",
        str(args.max_total_tokens),
        "--keyword_fanout_mode",
        str(experiment["keyword_fanout_mode"]),
        "--entity_retrieval_mode",
        str(experiment["entity_retrieval_mode"]),
        "--chunk_retrieval_mode",
        str(experiment["chunk_retrieval_mode"]),
        "--exclude_synonym_edges",
        "true" if experiment["exclude_synonym_edges"] else "false",
    ]
    if args.bypass_query_cache:
        cmd.append("--bypass_query_cache")
    if args.bypass_keywords_cache:
        cmd.append("--bypass_keywords_cache")
    if "kg_chunk_selection_source" in experiment:
        cmd.extend(
            [
                "--kg_chunk_selection_source",
                str(experiment["kg_chunk_selection_source"]),
            ]
        )
    if experiment["query_mode"] != "ppr":
        cmd.extend(["--answer_context_mode", str(experiment["answer_context_mode"])])
    return cmd


def _surge_command(args: argparse.Namespace, experiment: dict[str, Any], output_dir: Path) -> list[str]:
    cmd = [
        args.python_exe,
        "-m",
        "evaluate_local.SurGE.evaluate_surge_fast",
        "--mode",
        "retrieval",
        "--workspace-id",
        args.surge_workspace_id,
        "--query-mode",
        str(experiment["query_mode"]),
        "--top-k",
        str(args.top_k),
        "--chunk-top-k",
        str(args.surge_chunk_top_k),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--k-list",
        str(args.k_list),
        "--survey-k-list",
        str(args.survey_k_list),
        "--keyword_fanout_mode",
        str(experiment["keyword_fanout_mode"]),
        "--entity_retrieval_mode",
        str(experiment["entity_retrieval_mode"]),
        "--chunk_retrieval_mode",
        str(experiment["chunk_retrieval_mode"]),
        "--exclude_synonym_edges",
        "true" if experiment["exclude_synonym_edges"] else "false",
    ]
    if args.bypass_query_cache:
        cmd.append("--bypass_query_cache")
    if args.bypass_keywords_cache:
        cmd.append("--bypass_keywords_cache")
    if "kg_chunk_selection_source" in experiment:
        cmd.extend(
            [
                "--kg-chunk-selection-source",
                str(experiment["kg_chunk_selection_source"]),
            ]
        )
    return cmd


def _build_full_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    retrieval_modes = _parse_csv(args.retrieval_modes)
    if not retrieval_modes:
        retrieval_modes = _parse_csv(args.entity_retrieval_modes or "") or _parse_csv(
            args.chunk_retrieval_modes or ""
        )
    return build_full_experiment_matrix(
        query_modes=_parse_csv(args.query_modes),
        keyword_fanout_modes=_parse_csv(args.keyword_fanout_modes),
        retrieval_modes=retrieval_modes or ["dense", "hybrid"],
        exclude_synonym_edges_values=_bool_tokens(args.exclude_synonym_edges_values),
        kg_chunk_selection_sources=_parse_csv(args.kg_chunk_selection_sources),
        answer_context_modes=_parse_csv(args.answer_context_modes),
    )


def _selected_experiments(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.chunk_top_k is not None:
        args.shared_chunk_top_k = int(args.chunk_top_k)
        args.surge_chunk_top_k = int(args.chunk_top_k)

    if args.matrix_mode == "full":
        full = _build_full_experiments(args)
        shared = [{**item, "task": "shared"} for item in full]
        surge = [
            {
                **item,
                "task": "surge",
            }
            for item in full
            if not (
                item["query_mode"] != "ppr"
                and item.get("answer_context_mode") == "chunk_only_prompt"
            )
        ]
        return shared, surge

    shared = build_reduced_experiment_matrix("shared")
    surge = build_reduced_experiment_matrix("surge")
    return shared, surge


def _run_one(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    run_root: Path,
    progress_file: Path,
    task: str,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    name = _experiment_name(experiment)
    prefix = "docbench" if task == "shared" else "surge"
    output_dir = run_root / f"{prefix}__{name}"
    status_row = {
        "timestamp": _now_iso(),
        "experiment": name,
        "task": task,
        "config": experiment,
        "status": "skipped",
    }
    _append_jsonl(progress_file, {**status_row, "status": "running"})
    if args.dry_run:
        status_row["status"] = "dry_run"
    elif task == "shared":
        env["DOCBENCH_SHARED_OUTPUT_DIR"] = str(output_dir / "evaluate_shared")
        code = _run_command(
            command=_shared_command(args, experiment, output_dir),
            cwd=PROJECT_ROOT,
            env=env,
            log_file=output_dir / "logs" / "shared_generate.log",
        )
        status_row["status"] = "ok" if code == 0 else f"failed:{code}"
    else:
        env["SURGE_FAST_OUTPUT_DIR"] = str(output_dir / "evaluate_surge_fast")
        code = _run_command(
            command=_surge_command(args, experiment, output_dir),
            cwd=PROJECT_ROOT,
            env=env,
            log_file=output_dir / "logs" / "surge_retrieval.log",
        )
        status_row["status"] = "ok" if code == 0 else f"failed:{code}"
    _append_jsonl(progress_file, {**status_row, "status": "completed"})
    return status_row


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shared_experiments, surge_experiments = _selected_experiments(args)

    run_root = Path(args.output_root) / args.run_id
    progress_file = run_root / "progress.jsonl"
    summary_file = run_root / "summary.json"
    _write_json(
        run_root / "config.json",
        {
            "generated_at": _now_iso(),
            "run_root": args.run_root,
            "shared_workspace_id": args.shared_workspace_id,
            "surge_workspace_id": args.surge_workspace_id,
            "matrix_mode": args.matrix_mode,
            "shared_experiments": shared_experiments,
            "surge_experiments": surge_experiments,
        },
    )

    env = _build_workspace_env(dict(os.environ), args)

    results: list[dict[str, Any]] = []
    if args.tasks in {"both", "shared"}:
        for experiment in shared_experiments:
            results.append(
                _run_one(
                    args=args,
                    env=env,
                    run_root=run_root,
                    progress_file=progress_file,
                    task="shared",
                    experiment=experiment,
                )
            )
    if args.tasks in {"both", "surge"}:
        for experiment in surge_experiments:
            results.append(
                _run_one(
                    args=args,
                    env=env,
                    run_root=run_root,
                    progress_file=progress_file,
                    task="surge",
                    experiment=experiment,
                )
            )

    _write_json(summary_file, {"generated_at": _now_iso(), "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
