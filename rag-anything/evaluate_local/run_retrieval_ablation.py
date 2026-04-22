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


def build_experiment_matrix(
    *,
    query_modes: list[str],
    keyword_fanout_modes: list[str],
    entity_retrieval_modes: list[str],
    chunk_retrieval_modes: list[str],
    exclude_synonym_edges_values: list[bool],
    answer_context_modes: list[str],
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for query_mode in query_modes:
        if query_mode == "ppr_local":
            continue
        for keyword_fanout_mode in keyword_fanout_modes:
            for entity_retrieval_mode in entity_retrieval_modes:
                for exclude_synonym_edges in exclude_synonym_edges_values:
                    if query_mode == "ppr":
                        for chunk_retrieval_mode in chunk_retrieval_modes:
                            experiments.append(
                                {
                                    "query_mode": query_mode,
                                    "keyword_fanout_mode": keyword_fanout_mode,
                                    "entity_retrieval_mode": entity_retrieval_mode,
                                    "chunk_retrieval_mode": chunk_retrieval_mode,
                                    "exclude_synonym_edges": bool(exclude_synonym_edges),
                                    "answer_context_mode": "chunk_only_prompt",
                                }
                            )
                    else:
                        for chunk_retrieval_mode in chunk_retrieval_modes:
                            for answer_context_mode in answer_context_modes:
                                experiments.append(
                                    {
                                        "query_mode": query_mode,
                                        "keyword_fanout_mode": keyword_fanout_mode,
                                        "entity_retrieval_mode": entity_retrieval_mode,
                                        "chunk_retrieval_mode": chunk_retrieval_mode,
                                        "exclude_synonym_edges": bool(exclude_synonym_edges),
                                        "answer_context_mode": answer_context_mode,
                                    }
                                )
    return experiments


def _experiment_name(item: dict[str, Any]) -> str:
    parts = [
        str(item["query_mode"]),
        str(item["keyword_fanout_mode"]),
        f"er-{item['entity_retrieval_mode']}",
        f"syn-{int(bool(item['exclude_synonym_edges']))}",
    ]
    if "chunk_retrieval_mode" in item:
        parts.append(f"cr-{item['chunk_retrieval_mode']}")
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
    parser.add_argument("--query-modes", default="hybrid,ppr")
    parser.add_argument("--keyword-fanout-modes", default="joined,per_keyword_rrf")
    parser.add_argument("--entity-retrieval-modes", default="dense,hybrid")
    parser.add_argument("--chunk-retrieval-modes", default="dense,hybrid")
    parser.add_argument("--answer-context-modes", default="kg_prompt,chunk_only_prompt")
    parser.add_argument("--exclude-synonym-edges-values", default="true,false")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--chunk-top-k", type=int, default=20)
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
        str(args.chunk_top_k),
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
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiments = build_experiment_matrix(
        query_modes=_parse_csv(args.query_modes),
        keyword_fanout_modes=_parse_csv(args.keyword_fanout_modes),
        entity_retrieval_modes=_parse_csv(args.entity_retrieval_modes),
        chunk_retrieval_modes=_parse_csv(args.chunk_retrieval_modes),
        exclude_synonym_edges_values=_bool_tokens(args.exclude_synonym_edges_values),
        answer_context_modes=_parse_csv(args.answer_context_modes),
    )

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
            "experiments": experiments,
        },
    )

    env = _build_workspace_env(dict(os.environ), args)

    results: list[dict[str, Any]] = []
    for experiment in experiments:
        name = _experiment_name(experiment)
        output_dir = run_root / name
        status_row = {
            "timestamp": _now_iso(),
            "experiment": name,
            "config": experiment,
            "shared_status": "skipped",
            "surge_status": "skipped",
        }
        _append_jsonl(progress_file, {**status_row, "status": "running"})
        if not args.dry_run and args.tasks in {"both", "shared"}:
            env["DOCBENCH_SHARED_OUTPUT_DIR"] = str(output_dir / "evaluate_shared")
            shared_code = _run_command(
                command=_shared_command(args, experiment, output_dir),
                cwd=PROJECT_ROOT,
                env=env,
                log_file=output_dir / "logs" / "shared_generate.log",
            )
            status_row["shared_status"] = "ok" if shared_code == 0 else f"failed:{shared_code}"
        if not args.dry_run and args.tasks in {"both", "surge"}:
            env["SURGE_FAST_OUTPUT_DIR"] = str(output_dir / "evaluate_surge_fast")
            surge_code = _run_command(
                command=_surge_command(args, experiment, output_dir),
                cwd=PROJECT_ROOT,
                env=env,
                log_file=output_dir / "logs" / "surge_retrieval.log",
            )
            status_row["surge_status"] = "ok" if surge_code == 0 else f"failed:{surge_code}"
        results.append(status_row)
        _append_jsonl(progress_file, {**status_row, "status": "completed"})

    _write_json(summary_file, {"generated_at": _now_iso(), "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
