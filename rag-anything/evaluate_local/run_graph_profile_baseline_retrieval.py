#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_ROOT = PROJECT_ROOT.parent
LOCAL_LIGHTRAG_ROOT = PROJECTS_ROOT / "lightrag"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if LOCAL_LIGHTRAG_ROOT.exists() and str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

from raganything.constants import (
    DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES,
    DEFAULT_QDRANT_ENABLE_SPARSE_BM25,
    DEFAULT_QDRANT_SPARSE_BM25_MODEL,
)


DEFAULT_RUN_ID = "graphbm25_20260429"
DEFAULT_RUNS_ROOT = "evaluate_local/ablation_runs"
DEFAULT_OUTPUT_ROOT = "evaluate_local/retrieval_ablation_runs"
DEFAULT_DOCBENCH_DATA_ROOT = "/data/y50056788/Yaliang/datasets_for_eval/data_for_DocBench"
DEFAULT_SURGE_DATA_ROOT = "/data/y50056788/Yaliang/datasets_for_eval/data_for_SurGE"
DEFAULT_SHARED_MINERU_OUTPUT_DIR = (
    "evaluate_local/DocBench/docbench_shared_results/mineru_outputs"
)
ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "event",
    "artifact",
    "work",
    "naturalentity",
    "concept",
    "process",
]

DOCBENCH_MIN_RERANK_SCORE = 0.3
SURGE_MIN_RERANK_SCORE = 0.0


@dataclass(frozen=True)
class ProfileSpec:
    key: str
    description: str
    enable_entity_disambiguation: bool
    enable_synonym_linking: bool
    enable_multi_hop: bool = False


PROFILE_SPECS: dict[str, ProfileSpec] = {
    "v0": ProfileSpec(
        key="v0",
        description="V0: no entity disambiguation, no synonym linking",
        enable_entity_disambiguation=False,
        enable_synonym_linking=False,
    ),
    "v0_v1": ProfileSpec(
        key="v0_v1",
        description="V1: entity disambiguation on, synonym linking off",
        enable_entity_disambiguation=True,
        enable_synonym_linking=False,
    ),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_arg(value: bool) -> str:
    return "true" if bool(value) else "false"


def _sanitize_workspace_fragment(raw: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_-]+", "_", str(raw or "").strip())
    token = token.strip("_")
    return token or DEFAULT_RUN_ID


def _resolve_root(raw: str, *, default: str) -> Path:
    root = Path(str(raw or default))
    if root.is_absolute():
        return root.resolve()
    return (PROJECT_ROOT / root).resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _make_base_env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_entries = [str(PROJECT_ROOT)]
    if LOCAL_LIGHTRAG_ROOT.exists():
        pythonpath_entries.append(str(LOCAL_LIGHTRAG_ROOT))
    existing = str(env.get("PYTHONPATH", "")).strip()
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["QDRANT_ENABLE_SPARSE_BM25"] = (
        "true" if DEFAULT_QDRANT_ENABLE_SPARSE_BM25 else "false"
    )
    env["QDRANT_SPARSE_BM25_MODEL"] = DEFAULT_QDRANT_SPARSE_BM25_MODEL
    return env


def _profile_from_token(raw: str) -> ProfileSpec:
    token = str(raw or "").strip().lower().replace("-", "_")
    if token in PROFILE_SPECS:
        return PROFILE_SPECS[token]
    raise ValueError(
        f"unknown profile {raw!r}; valid profiles: {', '.join(PROFILE_SPECS)}"
    )


def _resolve_profiles(raw_profiles: list[str]) -> list[ProfileSpec]:
    resolved: list[ProfileSpec] = []
    seen: set[str] = set()
    for raw in raw_profiles:
        profile = _profile_from_token(raw)
        if profile.key in seen:
            continue
        seen.add(profile.key)
        resolved.append(profile)
    return resolved


def _docbench_retrieval_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "query_mode": "hybrid",
        "keyword_fanout_mode": "joined",
        "entity_retrieval_mode": "dense",
        "chunk_retrieval_mode": "dense",
        "enable_rerank": True,
        "enable_kg_rerank": False,
        "exclude_synonym_edges": True,
        "kg_chunk_selection_source": "truncated",
        "answer_context_mode": "kg_prompt",
        "min_rerank_score": DOCBENCH_MIN_RERANK_SCORE,
        "top_k": int(args.top_k),
        "chunk_top_k": int(args.shared_chunk_top_k),
        "max_total_tokens": int(args.max_total_tokens),
        "multimodal_top_k": int(args.docbench_multimodal_top_k),
        "keyword_entity_rrf_k": int(args.keyword_entity_rrf_k),
        "keyword_relation_rrf_k": int(args.keyword_relation_rrf_k),
        "bypass_query_cache": True,
        "bypass_keywords_cache": False,
    }


def _surge_retrieval_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "query_mode": "hybrid",
        "keyword_fanout_mode": "joined",
        "entity_retrieval_mode": "dense",
        "chunk_retrieval_mode": "dense",
        "enable_rerank": True,
        "enable_kg_rerank": False,
        "exclude_synonym_edges": True,
        "kg_chunk_selection_source": "untruncated",
        "min_rerank_score": SURGE_MIN_RERANK_SCORE,
        "top_k": int(args.top_k),
        "chunk_top_k": int(args.surge_chunk_top_k),
        "max_total_tokens": int(args.max_total_tokens),
        "k_list": str(args.k_list),
        "survey_k_list": str(args.survey_k_list),
        "keyword_entity_rrf_k": int(args.keyword_entity_rrf_k),
        "keyword_relation_rrf_k": int(args.keyword_relation_rrf_k),
        "bypass_query_cache": True,
        "bypass_keywords_cache": False,
    }


def _shared_workspace_id(run_id: str, profile: ProfileSpec) -> str:
    return f"docbench_shared_{run_id}_{profile.key}"


def _surge_workspace_id(run_id: str, profile: ProfileSpec) -> str:
    return f"surge_fast_{run_id}_{profile.key}"


def _profile_construction_settings(profile: ProfileSpec) -> dict[str, Any]:
    entity_id_shape = (
        "entity_name|entity_type"
        if profile.enable_entity_disambiguation
        else "entity_name"
    )
    return {
        "key": profile.key,
        "profile": profile.key,
        "description": profile.description,
        "enable_entity_disambiguation": profile.enable_entity_disambiguation,
        "enable_synonym_linking": profile.enable_synonym_linking,
        "enable_multi_hop": profile.enable_multi_hop,
        "entity_id_shape": entity_id_shape,
    }


def _base_ablation_args(
    *,
    profile: ProfileSpec,
    args: argparse.Namespace,
) -> list[str]:
    return [
        "--enable-entity-disambiguation",
        _bool_arg(profile.enable_entity_disambiguation),
        "--enable-synonym-linking",
        _bool_arg(profile.enable_synonym_linking),
        "--enable-multi-hop",
        _bool_arg(profile.enable_multi_hop),
        "--multi-hop-depth",
        str(args.multi_hop_depth),
        "--ppr-damping",
        str(args.ppr_damping),
        "--ppr-top-k",
        str(args.ppr_top_k),
        "--ppr-qa-top-k",
        str(args.ppr_qa_top_k),
        "--passage-node-weight",
        str(args.passage_node_weight),
    ]


def _build_docbench_command(
    *,
    profile: ProfileSpec,
    workspace_id: str,
    args: argparse.Namespace,
) -> list[str]:
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
        workspace_id,
        "--raganything_eval_setup",
        "--max_async_ingest",
        str(args.max_async_ingest),
        "--max_async_generate",
        str(args.max_async_generate),
        "--max_async_judge",
        str(args.max_async_judge),
        *_base_ablation_args(profile=profile, args=args),
        "--query_mode",
        "hybrid",
        "--keyword_fanout_mode",
        "joined",
        "--keyword_entity_rrf_k",
        str(args.keyword_entity_rrf_k),
        "--keyword_relation_rrf_k",
        str(args.keyword_relation_rrf_k),
        "--entity_retrieval_mode",
        "dense",
        "--chunk_retrieval_mode",
        "dense",
        "--exclude_synonym_edges",
        "true",
        "--enable_rerank",
        "true",
        "--enable_kg_rerank",
        "false",
        "--kg_chunk_selection_source",
        "truncated",
        "--answer_context_mode",
        "kg_prompt",
        "--max_total_tokens",
        str(args.max_total_tokens),
        "--multimodal_top_k",
        str(args.docbench_multimodal_top_k),
        "--bypass_query_cache",
    ]
    if not bool(args.resume):
        cmd.append("--no_resume")
    return cmd


def _build_surge_command(
    *,
    mode: str,
    profile: ProfileSpec,
    workspace_id: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        args.python_exe,
        "-m",
        "evaluate_local.SurGE.evaluate_surge_fast",
        "--mode",
        mode,
    ]
    if mode == "survey":
        cmd.extend(["--survey-stage", "retrieval"])
    cmd.extend(
        [
            "--data-root",
            str(args.surge_data_root),
            "--subset-dir",
            str(args.surge_subset_dir),
            "--queries-file",
            str(args.surge_queries_file),
            "--surveys-file",
            str(args.surge_surveys_file),
            "--chunks-file",
            str(args.surge_chunks_file),
            "--corpus-file",
            str(args.surge_corpus_file),
            "--workspace-id",
            workspace_id,
            "--query-mode",
            "hybrid",
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
            *_base_ablation_args(profile=profile, args=args),
            "--keyword_fanout_mode",
            "joined",
            "--keyword_entity_rrf_k",
            str(args.keyword_entity_rrf_k),
            "--keyword_relation_rrf_k",
            str(args.keyword_relation_rrf_k),
            "--entity_retrieval_mode",
            "dense",
            "--chunk_retrieval_mode",
            "dense",
            "--exclude_synonym_edges",
            "true",
            "--enable-rerank",
            "true",
            "--enable-kg-rerank",
            "false",
            "--kg-chunk-selection-source",
            "untruncated",
            "--batch-doc-concurrency",
            str(args.batch_doc_concurrency),
            "--ingest-batch-size",
            str(args.ingest_batch_size),
            "--llm-model-max-async",
            str(args.llm_model_max_async),
            "--max-concurrency",
            str(args.max_concurrency),
            "--max-retries",
            str(args.max_retries),
            "--limit",
            str(args.limit),
            "--bypass_query_cache",
        ]
    )
    return cmd


def _stage_payload(
    *,
    profile: ProfileSpec,
    task: str,
    stage: str,
    command: list[str],
    env_overrides: dict[str, str],
    output_dir: Path,
    log_file: Path,
) -> dict[str, Any]:
    return {
        "profile": profile.key,
        "task": task,
        "stage": stage,
        "command": command,
        "env_overrides": env_overrides,
        "output_dir": str(output_dir),
        "log_file": str(log_file),
    }


def _build_stages(
    *,
    profiles: list[ProfileSpec],
    run_id: str,
    ablation_run_root: Path,
    output_run_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    shared_mineru_output_dir = Path(str(args.shared_mineru_output_dir))
    if not shared_mineru_output_dir.is_absolute():
        shared_mineru_output_dir = (PROJECT_ROOT / shared_mineru_output_dir).resolve()

    for profile in profiles:
        profile_output_root = output_run_root / profile.key
        profile_log_dir = profile_output_root / "logs"

        shared_workspace_id = _shared_workspace_id(run_id, profile)
        shared_state_dir = ablation_run_root / "_workspace_cache" / "docbench_shared" / profile.key
        shared_output_dir = profile_output_root / "evaluate_shared"
        shared_env = {
            "DOCBENCH_SHARED_OUTPUT_DIR": str(shared_output_dir),
            "DOCBENCH_SHARED_WORKING_DIR_ROOT": str(
                shared_state_dir / "rag_workspaces"
            ),
            "DOCBENCH_SHARED_INGEST_MANIFEST_FILE": str(
                shared_state_dir / "shared_ingest_manifest.json"
            ),
            "DOCBENCH_SHARED_INGEST_FAILURES_FILE": str(
                shared_state_dir / "shared_ingest_failures.jsonl"
            ),
            "DOCBENCH_SHARED_MINERU_OUTPUT_DIR": str(shared_mineru_output_dir),
            "NEO4J_WORKSPACE": shared_workspace_id,
            "QDRANT_WORKSPACE": shared_workspace_id,
        }
        if str(args.docbench_data_root or "").strip():
            shared_env["DOCBENCH_SHARED_DATA_ROOT"] = str(args.docbench_data_root).strip()

        if args.tasks in {"both", "shared"}:
            stages.append(
                _stage_payload(
                    profile=profile,
                    task="shared",
                    stage="docbench_generate",
                    command=_build_docbench_command(
                        profile=profile,
                        workspace_id=shared_workspace_id,
                        args=args,
                    ),
                    env_overrides=shared_env,
                    output_dir=shared_output_dir,
                    log_file=profile_log_dir / "docbench_generate.log",
                )
            )

        surge_workspace_id = _surge_workspace_id(run_id, profile)
        surge_state_dir = ablation_run_root / "_workspace_cache" / "surge_fast" / profile.key
        surge_output_dir = profile_output_root / "evaluate_surge_fast"
        surge_env = {
            "SURGE_FAST_OUTPUT_DIR": str(surge_output_dir),
            "SURGE_FAST_RAG_STORAGE_DIR": str(surge_state_dir / "rag_storage"),
            "SURGE_FAST_RAG_OUTPUT_DIR": str(surge_state_dir / "rag_outputs"),
            "NEO4J_WORKSPACE": surge_workspace_id,
            "QDRANT_WORKSPACE": surge_workspace_id,
        }
        if args.tasks in {"both", "surge"}:
            stages.append(
                _stage_payload(
                    profile=profile,
                    task="surge",
                    stage="surge_retrieval",
                    command=_build_surge_command(
                        mode="retrieval",
                        profile=profile,
                        workspace_id=surge_workspace_id,
                        args=args,
                    ),
                    env_overrides=surge_env,
                    output_dir=surge_output_dir,
                    log_file=profile_log_dir / "surge_retrieval.log",
                )
            )
            stages.append(
                _stage_payload(
                    profile=profile,
                    task="surge",
                    stage="surge_survey_retrieval",
                    command=_build_surge_command(
                        mode="survey",
                        profile=profile,
                        workspace_id=surge_workspace_id,
                        args=args,
                    ),
                    env_overrides=surge_env,
                    output_dir=surge_output_dir,
                    log_file=profile_log_dir / "surge_survey_retrieval.log",
                )
            )

    return stages


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


def _execute_stage(
    *,
    stage: dict[str, Any],
    base_env: dict[str, str],
    progress_file: Path,
    dry_run: bool,
) -> dict[str, Any]:
    started = time.time()
    running_row = {
        "timestamp": _now_iso(),
        "profile": stage["profile"],
        "task": stage["task"],
        "stage": stage["stage"],
        "status": "running",
        "command": stage["command"],
        "log_file": stage["log_file"],
    }
    _append_jsonl(progress_file, running_row)

    if dry_run:
        result = {
            **running_row,
            "timestamp": _now_iso(),
            "status": "dry_run",
            "elapsed_sec": 0.0,
        }
        _append_jsonl(progress_file, result)
        return result

    env = dict(base_env)
    env.update(stage["env_overrides"])
    code = _run_command(
        command=stage["command"],
        cwd=PROJECT_ROOT,
        env=env,
        log_file=Path(stage["log_file"]),
    )
    ok = code == 0
    result = {
        **running_row,
        "timestamp": _now_iso(),
        "status": "ok" if ok else "failed",
        "returncode": code,
        "elapsed_sec": round(time.time() - started, 3),
    }
    _append_jsonl(progress_file, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build v0/v0_v1 graph profiles and run the matching baseline retrieval "
            "for DocBench and SurGE."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--profiles", nargs="+", default=["v0", "v0_v1"])
    parser.add_argument("--tasks", choices=["both", "shared", "surge"], default="both")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--allow-reuse-run-id",
        action="store_true",
        help="Allow writing into an existing non-empty output/run directory.",
    )
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=49)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--docbench-data-root", default=DEFAULT_DOCBENCH_DATA_ROOT)
    parser.add_argument(
        "--shared-mineru-output-dir",
        default=DEFAULT_SHARED_MINERU_OUTPUT_DIR,
    )
    parser.add_argument("--surge-data-root", default=DEFAULT_SURGE_DATA_ROOT)
    parser.add_argument("--surge-subset-dir", default="subset_output")
    parser.add_argument("--surge-queries-file", default="subset_queries.json")
    parser.add_argument("--surge-surveys-file", default="subset_surveys.json")
    parser.add_argument("--surge-chunks-file", default="subset_chunks.jsonl")
    parser.add_argument("--surge-corpus-file", default="subset_corpus.json")

    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--shared-chunk-top-k", type=int, default=20)
    parser.add_argument("--surge-chunk-top-k", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=45000)
    parser.add_argument("--docbench-multimodal-top-k", type=int, default=3)
    parser.add_argument("--k-list", default="5,10,20,30,50")
    parser.add_argument("--survey-k-list", default="50,100,200,500")
    parser.add_argument("--keyword-entity-rrf-k", type=int, default=10)
    parser.add_argument("--keyword-relation-rrf-k", type=int, default=20)

    parser.add_argument("--max-async-ingest", type=int, default=4)
    parser.add_argument("--max-async-generate", type=int, default=6)
    parser.add_argument("--max-async-judge", type=int, default=32)
    parser.add_argument("--batch-doc-concurrency", type=int, default=2)
    parser.add_argument("--ingest-batch-size", type=int, default=384)
    parser.add_argument("--llm-model-max-async", type=int, default=48)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=0)

    parser.add_argument("--multi-hop-depth", type=int, default=2)
    parser.add_argument("--ppr-damping", type=float, default=0.5)
    parser.add_argument("--ppr-top-k", type=int, default=50)
    parser.add_argument("--ppr-qa-top-k", type=int, default=5)
    parser.add_argument("--passage-node-weight", type=float, default=0.05)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.end_id <= args.start_id:
        raise ValueError(f"--end-id must be greater than --start-id, got {args.end_id}")
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be > 0, got {args.top_k}")
    if args.shared_chunk_top_k <= 0:
        raise ValueError(
            f"--shared-chunk-top-k must be > 0, got {args.shared_chunk_top_k}"
        )
    if args.surge_chunk_top_k < 0:
        raise ValueError(
            f"--surge-chunk-top-k must be >= 0, got {args.surge_chunk_top_k}"
        )
    if args.max_total_tokens <= 0:
        raise ValueError(f"--max-total-tokens must be > 0, got {args.max_total_tokens}")
    if args.limit < 0:
        raise ValueError(f"--limit must be >= 0, got {args.limit}")
    if not (0.0 < float(args.ppr_damping) < 1.0):
        raise ValueError(f"--ppr-damping must be in (0,1), got {args.ppr_damping}")
    if args.ppr_top_k <= 0 or args.ppr_qa_top_k <= 0:
        raise ValueError("--ppr-top-k and --ppr-qa-top-k must be > 0")
    if args.ppr_qa_top_k > args.ppr_top_k:
        raise ValueError(
            f"--ppr-qa-top-k must be <= --ppr-top-k, got "
            f"{args.ppr_qa_top_k} > {args.ppr_top_k}"
        )


def _config_payload(
    *,
    run_id: str,
    ablation_run_root: Path,
    output_run_root: Path,
    profiles: list[ProfileSpec],
    stages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "run_id": run_id,
        "ablation_run_root": str(ablation_run_root),
        "output_run_root": str(output_run_root),
        "profiles": [_profile_construction_settings(profile) for profile in profiles],
        "construction_settings": {
            "entity_types": ENTITY_TYPES,
            "prompt": {
                "source": "LightRAG PROMPTS entity extraction templates",
                "type_rule": "Use ONLY the entity types listed in <Entity_types>.",
            },
            "normalization": {
                "enable_entity_surface_normalization": True,
                "enable_keyword_case_normalization": True,
            },
            "endpoint_constraints": {
                "strict_relation_endpoint_entity_match": True,
            },
            "context_window": {
                "ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE": "true",
                "CONTEXT_ZERO_WINDOW_CONTENT_TYPES": DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES,
            },
            "qdrant": {
                "QDRANT_ENABLE_SPARSE_BM25": (
                    "true" if DEFAULT_QDRANT_ENABLE_SPARSE_BM25 else "false"
                ),
                "QDRANT_SPARSE_BM25_MODEL": DEFAULT_QDRANT_SPARSE_BM25_MODEL,
            },
        },
        "retrieval_settings": {
            "docbench": _docbench_retrieval_settings(args),
            "surge_query_level": _surge_retrieval_settings(args),
            "surge_survey_level": {
                **_surge_retrieval_settings(args),
                "mode": "survey",
                "survey_stage": "retrieval",
            },
        },
        "planned_stages": stages,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    run_id = _sanitize_workspace_fragment(str(args.run_id or DEFAULT_RUN_ID))
    profiles = _resolve_profiles(args.profiles)
    ablation_run_root = _resolve_root(args.runs_root, default=DEFAULT_RUNS_ROOT) / run_id
    output_run_root = _resolve_root(args.output_root, default=DEFAULT_OUTPUT_ROOT) / run_id

    if (
        not bool(args.allow_reuse_run_id)
        and (
            (ablation_run_root.exists() and any(ablation_run_root.iterdir()))
            or (output_run_root.exists() and any(output_run_root.iterdir()))
        )
    ):
        raise RuntimeError(
            "run-id output already exists and is not empty. "
            f"ablation_run_root={ablation_run_root}; output_run_root={output_run_root}. "
            "Use a new --run-id or pass --allow-reuse-run-id."
        )

    ablation_run_root.mkdir(parents=True, exist_ok=True)
    output_run_root.mkdir(parents=True, exist_ok=True)
    progress_file = output_run_root / "progress.jsonl"
    summary_file = output_run_root / "summary.json"
    config_file = output_run_root / "config.json"

    stages = _build_stages(
        profiles=profiles,
        run_id=run_id,
        ablation_run_root=ablation_run_root,
        output_run_root=output_run_root,
        args=args,
    )
    config = _config_payload(
        run_id=run_id,
        ablation_run_root=ablation_run_root,
        output_run_root=output_run_root,
        profiles=profiles,
        stages=stages,
        args=args,
    )
    _write_json(config_file, config)

    base_env = _make_base_env()
    results: list[dict[str, Any]] = []
    for stage in stages:
        result = _execute_stage(
            stage=stage,
            base_env=base_env,
            progress_file=progress_file,
            dry_run=bool(args.dry_run),
        )
        results.append(result)
        if result["status"] == "failed" and not bool(args.continue_on_error):
            break

    failed = [item for item in results if item["status"] == "failed"]
    summary = {
        "generated_at": _now_iso(),
        "run_id": run_id,
        "status": "ok" if not failed else "failed",
        "config_file": str(config_file),
        "progress_file": str(progress_file),
        "results": results,
    }
    _write_json(summary_file, summary)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
