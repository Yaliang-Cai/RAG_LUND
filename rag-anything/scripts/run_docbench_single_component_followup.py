#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RUN_ROOT = (
    PROJECT_ROOT
    / "evaluate_local"
    / "retrieval_ablation_runs"
    / "docbench_single_v0_syn_20260508"
)
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "evaluate_local"
    / "retrieval_ablation_runs"
    / "docbench_single_v0_syn_component_followup_20260511"
)
INDEX_PROFILE_FILE = ".docbench_index_profile.json"
EXPECTED_SYNONYMY_THRESHOLD = 0.8
EXPECTED_PROFILE_FIELDS: dict[str, Any] = {
    "base_profile": "v0",
    "enable_entity_disambiguation": False,
    "enable_synonym_linking": False,
    "enable_multi_hop": False,
    "synonym_edges_postprocess_enabled": True,
    "synonym_edges_threshold": EXPECTED_SYNONYMY_THRESHOLD,
    "enable_entity_surface_normalization": True,
    "enable_keyword_case_normalization": True,
    "strict_relation_endpoint_entity_match": True,
    "qdrant_enable_sparse_bm25": True,
    "qdrant_retrieval_mode": "dense",
}


class RunPaths(NamedTuple):
    run_root: Path
    source_run_root: Path
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


def _with_windows(
    group: dict[str, Any],
    *,
    top_k: int = 20,
    chunk_top_k: int = 10,
    naive_top_k: int = 20,
) -> dict[str, Any]:
    return {
        **group,
        "top_k": int(top_k),
        "chunk_top_k": int(chunk_top_k),
        "naive_top_k": int(naive_top_k),
        "multimodal_top_k": 3,
    }


def _baseline() -> dict[str, Any]:
    return _with_retrieval(
        {
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


def _no_chunk_only_all_on() -> dict[str, Any]:
    return _with_retrieval(
        {
            "query_mode": "hybrid",
            "keyword_fanout_mode": "per_keyword_rrf",
            "exclude_synonym_edges": True,
            "kg_chunk_selection_source": "truncated",
            "answer_context_mode": "kg_prompt",
            "enable_rerank": True,
            "enable_kg_rerank": True,
        },
        "hybrid",
    )


def build_docbench_single_component_experiments() -> list[dict[str, Any]]:
    baseline = _baseline()
    no_chunk_only_all_on = _no_chunk_only_all_on()
    return [
        _with_windows(
            {
                **_with_retrieval(baseline, "hybrid"),
                "name": "baseline_plus_hybrid_retrieval",
            }
        ),
        _with_windows(
            {
                **baseline,
                "name": "baseline_plus_per_keyword",
                "keyword_fanout_mode": "per_keyword_rrf",
            }
        ),
        _with_windows(
            {
                **baseline,
                "name": "baseline_plus_kg_rerank",
                "enable_kg_rerank": True,
            }
        ),
        _with_windows(
            {
                **no_chunk_only_all_on,
                "name": "no_chunk_only_all_on",
            }
        ),
        _with_windows(
            {
                **baseline,
                "name": "baseline_wide_40_20",
            },
            top_k=40,
            chunk_top_k=20,
        ),
        _with_windows(
            {
                **no_chunk_only_all_on,
                "name": "no_chunk_only_all_on_wide_40_20",
            },
            top_k=40,
            chunk_top_k=20,
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DocBench single-document component follow-up experiments."
    )
    parser.add_argument("--source-run-root", default=str(DEFAULT_SOURCE_RUN_ROOT))
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
    parser.add_argument("--max-async-ingest-docs", type=int, default=1)
    parser.add_argument("--max-async-query-docs", type=int, default=1)
    parser.add_argument("--max-async-generate", type=int, default=6)
    parser.add_argument("--max-async-judge", type=int, default=50)
    parser.add_argument("--doc-flush-every", type=int, default=4)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bypass-query-cache", action="store_true")
    parser.add_argument("--bypass-keywords-cache", action="store_true")
    return parser


def resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    source_run_root = Path(args.source_run_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    return RunPaths(
        run_root=run_root,
        source_run_root=source_run_root,
        working_dir_root=source_run_root / "index" / "rag_workspaces",
        index_state_dir=source_run_root / "index" / "state",
    )


def _load_manifest(index_state_dir: Path) -> dict[str, Any]:
    manifest_path = index_state_dir / "single_ingest_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source ingest manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("docs"), dict):
        raise ValueError(f"Invalid source ingest manifest: {manifest_path}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _close_float(left: Any, right: float) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return False


def _profile_mismatches(profile: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key, expected in EXPECTED_PROFILE_FIELDS.items():
        actual = profile.get(key)
        if isinstance(expected, float):
            if not _close_float(actual, expected):
                mismatches.append(f"{key}={actual!r} expected {expected!r}")
        elif actual != expected:
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    return mismatches


def preflight_source_index(paths: RunPaths, *, start_id: int, end_id: int) -> None:
    if not paths.working_dir_root.exists():
        raise FileNotFoundError(f"Missing source graph root: {paths.working_dir_root}")
    if not paths.index_state_dir.exists():
        raise FileNotFoundError(f"Missing source index state dir: {paths.index_state_dir}")

    manifest = _load_manifest(paths.index_state_dir)
    docs = manifest["docs"]
    missing: list[str] = []
    bad_status: list[str] = []
    missing_profiles: list[str] = []
    bad_profiles: list[str] = []
    bad_synonym_records: list[str] = []
    for doc_id in range(start_id, end_id):
        doc_name = str(doc_id)
        workspace_id = f"docbench_{doc_name}"
        record = docs.get(doc_name)
        if not isinstance(record, dict):
            missing.append(doc_name)
            continue
        if record.get("status") != "ok":
            bad_status.append(f"{doc_name}:{record.get('status')!r}")
        profile_path = paths.working_dir_root / workspace_id / INDEX_PROFILE_FILE
        if not profile_path.exists():
            missing_profiles.append(str(profile_path))
            continue
        workspace_profile = _load_json_object(profile_path)
        manifest_profile = record.get("index_profile")
        if manifest_profile != workspace_profile:
            bad_profiles.append(f"{doc_name}: manifest/workspace profile mismatch")
        profile_mismatches = _profile_mismatches(workspace_profile)
        if profile_mismatches:
            bad_profiles.append(f"{doc_name}: {', '.join(profile_mismatches)}")

        synonym_edges = record.get("synonym_edges")
        if not isinstance(synonym_edges, dict):
            bad_synonym_records.append(f"{doc_name}: missing synonym_edges record")
        elif not bool(synonym_edges.get("applied")):
            bad_synonym_records.append(f"{doc_name}: synonym_edges.applied is not true")
        elif not _close_float(
            synonym_edges.get("threshold"), EXPECTED_SYNONYMY_THRESHOLD
        ):
            bad_synonym_records.append(
                f"{doc_name}: synonym_edges.threshold={synonym_edges.get('threshold')!r}"
            )

    problems: list[str] = []
    if missing:
        problems.append(f"missing manifest docs={missing}")
    if bad_status:
        problems.append(f"non-ok manifest docs={bad_status}")
    if missing_profiles:
        problems.append(f"missing workspace profiles={missing_profiles}")
    if bad_profiles:
        problems.append(f"profile mismatches={bad_profiles}")
    if bad_synonym_records:
        problems.append(f"bad synonym edge records={bad_synonym_records}")
    if problems:
        raise RuntimeError(
            "Source DocBench single graph preflight failed: " + "; ".join(problems)
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
                "--multimodal_top_k",
                str(experiment["multimodal_top_k"]),
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
                str(EXPECTED_SYNONYMY_THRESHOLD),
            ]
        )
        command.extend(
            ["--kg_chunk_selection_source", str(experiment["kg_chunk_selection_source"])]
        )
        command.extend(["--answer_context_mode", str(experiment["answer_context_mode"])])
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
    preflight_source_index(paths, start_id=int(args.start_id), end_id=int(args.end_id))
    paths.run_root.mkdir(parents=True, exist_ok=True)

    experiments = build_docbench_single_component_experiments()
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
