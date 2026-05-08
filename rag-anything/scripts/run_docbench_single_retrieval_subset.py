#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "evaluate_local"
    / "retrieval_ablation_runs"
    / "docbench_single_v0_syn_20260508"
)


class RunPaths(NamedTuple):
    run_root: Path
    working_dir_root: Path
    index_state_dir: Path

    def output_dir_for(self, group_name: str) -> Path:
        return self.run_root / f"docbench_single__{group_name}" / "evaluate"


def _with_retrieval(group: dict[str, Any], retrieval_mode: str) -> dict[str, Any]:
    return {
        **group,
        "retrieval_mode": retrieval_mode,
        "entity_retrieval_mode": retrieval_mode,
        "chunk_retrieval_mode": retrieval_mode,
    }


def _with_windows(group: dict[str, Any]) -> dict[str, Any]:
    return {
        **group,
        "top_k": 20,
        "chunk_top_k": 10,
        "naive_top_k": 20,
    }


def build_docbench_single_experiments() -> list[dict[str, Any]]:
    v4_baseline = _with_retrieval(
        {
            "name": "v4_baseline_non_ppr",
            "query_mode": "hybrid",
            "keyword_fanout_mode": "joined",
            "exclude_synonym_edges": True,
            "kg_chunk_selection_source": "truncated",
            "answer_context_mode": "kg_prompt",
            "enable_rerank": True,
            "enable_kg_rerank": False,
        },
        "dense",
    )
    v7_ppr = _with_retrieval(
        {
            "name": "v7_baseline_ppr_all_on",
            "query_mode": "ppr",
            "keyword_fanout_mode": "per_keyword_rrf",
            "exclude_synonym_edges": False,
            "answer_context_mode": "chunk_only_prompt",
            "enable_rerank": True,
            "enable_kg_rerank": False,
            "ppr_top_k": 50,
            "ppr_qa_top_k": 10,
            "ppr_post_rerank_fusion": "none",
        },
        "hybrid",
    )
    return [
        _with_windows(
            _with_retrieval(
                {
                    "name": "v4_naive_dense",
                    "query_mode": "naive",
                    "keyword_fanout_mode": "joined",
                    "exclude_synonym_edges": True,
                    "enable_rerank": True,
                    "enable_kg_rerank": False,
                },
                "dense",
            )
        ),
        _with_windows(v4_baseline),
        _with_windows(
            {
                **v4_baseline,
                "name": "v4_non_ppr_chunk_only",
                "answer_context_mode": "chunk_only_prompt",
            }
        ),
        _with_windows(
            _with_retrieval(
                {
                    "name": "v6_baseline_non_ppr_all_on",
                    "query_mode": "hybrid",
                    "keyword_fanout_mode": "per_keyword_rrf",
                    "exclude_synonym_edges": True,
                    "kg_chunk_selection_source": "truncated",
                    "answer_context_mode": "chunk_only_prompt",
                    "enable_rerank": True,
                    "enable_kg_rerank": True,
                },
                "hybrid",
            )
        ),
        _with_windows(
            _with_retrieval(
                {
                    "name": "v4_ppr_default",
                    "query_mode": "ppr",
                    "keyword_fanout_mode": "joined",
                    "exclude_synonym_edges": True,
                    "answer_context_mode": "chunk_only_prompt",
                    "enable_rerank": False,
                    "enable_kg_rerank": False,
                    "ppr_top_k": 50,
                    "ppr_qa_top_k": 10,
                    "ppr_post_rerank_fusion": "none",
                },
                "dense",
            )
        ),
        _with_windows(v7_ppr),
        _with_windows(
            {
                **v7_ppr,
                "name": "v7_ppr_no_synonym_edges",
                "exclude_synonym_edges": True,
            }
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DocBench single-document V0+synonym retrieval subset."
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--stage",
        "--phase",
        choices=["generate", "evaluate", "stats", "all"],
        default="generate",
        help=(
            "Execution stage. Default runs retrieval/generation only. "
            "Run evaluate after starting the judge model, then run stats."
        ),
    )
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=49)
    parser.add_argument("--max-async-ingest-docs", type=int, default=2)
    parser.add_argument("--max-async-query-docs", type=int, default=1)
    parser.add_argument("--max-async-generate", type=int, default=6)
    parser.add_argument("--max-async-judge", type=int, default=50)
    parser.add_argument("--doc-flush-every", type=int, default=4)
    parser.add_argument("--synonymy-threshold", type=float, default=0.8)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--bypass-query-cache", action="store_true")
    parser.add_argument("--bypass-keywords-cache", action="store_true")
    return parser


def resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    run_root = Path(args.run_root).expanduser().resolve()
    return RunPaths(
        run_root=run_root,
        working_dir_root=run_root / "index" / "rag_workspaces",
        index_state_dir=run_root / "index" / "state",
    )


def _bool_arg(value: bool) -> str:
    return "true" if bool(value) else "false"


def _evaluate_py_command(
    *,
    args: argparse.Namespace,
    paths: RunPaths,
    experiment: dict[str, Any],
    mode: str,
) -> list[str]:
    output_dir = paths.output_dir_for(str(experiment["name"]))
    command = [
        str(args.python_exe),
        str(PROJECT_ROOT / "evaluate_local" / "DocBench" / "evaluate.py"),
        "--mode",
        mode,
        "--start_id",
        str(args.start_id),
        "--end_id",
        str(args.end_id),
        "--run_output_dir",
        str(output_dir),
        "--working_dir_root",
        str(paths.working_dir_root),
        "--index_state_dir",
        str(paths.index_state_dir),
    ]
    if args.no_resume:
        command.append("--no_resume")
    if mode == "generate":
        command.extend(
            [
                "--max_async_ingest_docs",
                str(args.max_async_ingest_docs),
                "--max_async_query_docs",
                str(args.max_async_query_docs),
                "--max_async_generate",
                str(args.max_async_generate),
                "--doc_flush_every",
                str(args.doc_flush_every),
                "--query_mode",
                str(experiment["query_mode"]),
                "--top_k",
                str(experiment["top_k"]),
                "--chunk_top_k",
                str(experiment["chunk_top_k"]),
                "--naive_top_k",
                str(experiment["naive_top_k"]),
                "--keyword_fanout_mode",
                str(experiment["keyword_fanout_mode"]),
                "--entity_retrieval_mode",
                str(experiment["entity_retrieval_mode"]),
                "--chunk_retrieval_mode",
                str(experiment["chunk_retrieval_mode"]),
                "--exclude_synonym_edges",
                _bool_arg(experiment["exclude_synonym_edges"]),
                "--enable_rerank",
                _bool_arg(experiment["enable_rerank"]),
                "--enable_kg_rerank",
                _bool_arg(experiment["enable_kg_rerank"]),
                "--apply_synonym_edges",
                "true",
                "--synonymy_threshold",
                str(args.synonymy_threshold),
            ]
        )
        if "kg_chunk_selection_source" in experiment:
            command.extend(
                ["--kg_chunk_selection_source", str(experiment["kg_chunk_selection_source"])]
            )
        if experiment["query_mode"] != "ppr":
            if "answer_context_mode" in experiment:
                command.extend(
                    ["--answer_context_mode", str(experiment["answer_context_mode"])]
                )
        else:
            command.extend(
                [
                    "--ppr_top_k",
                    str(experiment["ppr_top_k"]),
                    "--ppr_qa_top_k",
                    str(experiment["ppr_qa_top_k"]),
                    "--ppr_post_rerank_fusion",
                    str(experiment["ppr_post_rerank_fusion"]),
                ]
            )
        if args.bypass_query_cache:
            command.append("--bypass_query_cache")
        if args.bypass_keywords_cache:
            command.append("--bypass_keywords_cache")
    elif mode == "evaluate":
        command.extend(["--max_async_judge", str(args.max_async_judge)])
    return command


def _run_command(command: list[str], *, dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_run_paths(args)
    paths.run_root.mkdir(parents=True, exist_ok=True)
    paths.working_dir_root.mkdir(parents=True, exist_ok=True)
    paths.index_state_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_docbench_single_experiments()
    (paths.run_root / "experiments.json").write_text(
        json.dumps(experiments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    modes = (
        ("generate", "evaluate", "stats")
        if args.stage == "all"
        else (str(args.stage),)
    )
    for experiment in experiments:
        paths.output_dir_for(str(experiment["name"])).mkdir(parents=True, exist_ok=True)
    for mode in modes:
        for experiment in experiments:
            code = _run_command(
                _evaluate_py_command(
                    args=args,
                    paths=paths,
                    experiment=experiment,
                    mode=mode,
                ),
                dry_run=bool(args.dry_run),
            )
            if code != 0:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
