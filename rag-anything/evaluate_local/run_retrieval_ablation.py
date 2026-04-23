#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
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

from evaluate_local.ablation_flags import add_ablation_arguments, validate_ablation_flags


DEFAULT_RUN_ROOT = (
    "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421"
)
DEFAULT_SHARED_WORKSPACE_ID = "docbench_shared_graphbm25_20260421_v0_v1_v2"
DEFAULT_SURGE_WORKSPACE_ID = "surge_fast_graphbm25_20260421_v0_v1_v2"
DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "evaluate_local" / "retrieval_ablation_runs")
DEFAULT_SHARED_PPR_TOP_K = 50
DEFAULT_SHARED_PPR_QA_TOP_K = 20
DEFAULT_SURGE_PPR_TOP_K = 50
DEFAULT_SURGE_PPR_QA_TOP_K = 50
INDEX_PROFILE_FILE = ".ablation_index_profile.json"
_PROFILE_HINTS: dict[str, dict[str, bool]] = {
    "v0_v1_v2_v3": {
        "enable_entity_disambiguation": True,
        "enable_synonym_linking": True,
        "enable_multi_hop": True,
    },
    "v0_v1_v2": {
        "enable_entity_disambiguation": True,
        "enable_synonym_linking": True,
        "enable_multi_hop": False,
    },
    "v0_v1": {
        "enable_entity_disambiguation": True,
        "enable_synonym_linking": False,
        "enable_multi_hop": False,
    },
    "db_only": {
        "enable_entity_disambiguation": False,
        "enable_synonym_linking": False,
        "enable_multi_hop": False,
    },
    "v0": {
        "enable_entity_disambiguation": False,
        "enable_synonym_linking": False,
        "enable_multi_hop": False,
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


def _argv_mentions_option(argv: list[str], *option_names: str) -> bool:
    for token in argv:
        for option_name in option_names:
            if token == option_name or token.startswith(option_name + "="):
                return True
    return False


def _infer_profile_flags_from_workspace_id(workspace_id: str) -> dict[str, bool] | None:
    normalized = str(workspace_id or "").strip().lower()
    if not normalized:
        return None
    for profile_key in sorted(_PROFILE_HINTS.keys(), key=len, reverse=True):
        marker = f"_{profile_key}"
        if normalized.endswith(marker) or marker in normalized:
            return dict(_PROFILE_HINTS[profile_key])
    return None


def _with_unified_retrieval(item: dict[str, Any]) -> dict[str, Any]:
    retrieval_mode = str(item.get("retrieval_mode", "dense"))
    return {
        **item,
        "retrieval_mode": retrieval_mode,
        "entity_retrieval_mode": retrieval_mode,
        "chunk_retrieval_mode": retrieval_mode,
        "enable_rerank": bool(item.get("enable_rerank", True)),
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
    enable_rerank: bool,
    ppr_top_k: int,
    ppr_qa_top_k: int,
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
            "enable_rerank": bool(enable_rerank),
            "ppr_top_k": int(ppr_top_k),
            "ppr_qa_top_k": int(ppr_qa_top_k),
        }
    )


def _validate_ppr_controls(
    *,
    query_mode: str,
    ppr_top_k: int | None,
    ppr_qa_top_k: int | None,
    context: str,
) -> None:
    if str(query_mode).strip() != "ppr":
        return
    if ppr_top_k is None or int(ppr_top_k) <= 0:
        raise ValueError(f"{context}: ppr_top_k must be > 0, got {ppr_top_k!r}")
    if ppr_qa_top_k is None or int(ppr_qa_top_k) <= 0:
        raise ValueError(f"{context}: ppr_qa_top_k must be > 0, got {ppr_qa_top_k!r}")
    if int(ppr_qa_top_k) > int(ppr_top_k):
        raise ValueError(
            f"{context}: ppr_qa_top_k must be <= ppr_top_k, got "
            f"{int(ppr_qa_top_k)} > {int(ppr_top_k)}"
        )


def build_reduced_experiment_matrix(
    task: str,
    *,
    shared_ppr_top_k: int = DEFAULT_SHARED_PPR_TOP_K,
    shared_ppr_qa_top_k: int = DEFAULT_SHARED_PPR_QA_TOP_K,
    surge_ppr_top_k: int = DEFAULT_SURGE_PPR_TOP_K,
    surge_ppr_qa_top_k: int = DEFAULT_SURGE_PPR_QA_TOP_K,
) -> list[dict[str, Any]]:
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
                name="ppr_dense_rerank",
                keyword_fanout_mode="joined",
                retrieval_mode="dense",
                enable_rerank=True,
                ppr_top_k=shared_ppr_top_k,
                ppr_qa_top_k=shared_ppr_qa_top_k,
            ),
            _ppr_experiment(
                task=task_name,
                name="ppr_dense_no_rerank",
                keyword_fanout_mode="joined",
                retrieval_mode="dense",
                enable_rerank=False,
                ppr_top_k=shared_ppr_top_k,
                ppr_qa_top_k=shared_ppr_qa_top_k,
            ),
            _ppr_experiment(
                task=task_name,
                name="ppr_hybrid_per_keyword",
                keyword_fanout_mode="per_keyword_rrf",
                retrieval_mode="hybrid",
                enable_rerank=True,
                ppr_top_k=shared_ppr_top_k,
                ppr_qa_top_k=shared_ppr_qa_top_k,
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
                name="ppr_dense_rerank",
                keyword_fanout_mode="joined",
                retrieval_mode="dense",
                enable_rerank=True,
                ppr_top_k=surge_ppr_top_k,
                ppr_qa_top_k=surge_ppr_qa_top_k,
            ),
            _ppr_experiment(
                task="surge",
                name="ppr_dense_no_rerank",
                keyword_fanout_mode="joined",
                retrieval_mode="dense",
                enable_rerank=False,
                ppr_top_k=surge_ppr_top_k,
                ppr_qa_top_k=surge_ppr_qa_top_k,
            ),
            _ppr_experiment(
                task="surge",
                name="ppr_hybrid_per_keyword",
                keyword_fanout_mode="per_keyword_rrf",
                retrieval_mode="hybrid",
                enable_rerank=True,
                ppr_top_k=surge_ppr_top_k,
                ppr_qa_top_k=surge_ppr_qa_top_k,
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
                                    "enable_rerank": True,
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
                                            "enable_rerank": True,
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
    parser.add_argument("--shared-ppr-top-k", type=int, default=DEFAULT_SHARED_PPR_TOP_K)
    parser.add_argument(
        "--shared-ppr-qa-top-k",
        type=int,
        default=DEFAULT_SHARED_PPR_QA_TOP_K,
    )
    parser.add_argument("--surge-ppr-top-k", type=int, default=DEFAULT_SURGE_PPR_TOP_K)
    parser.add_argument(
        "--surge-ppr-qa-top-k",
        type=int,
        default=DEFAULT_SURGE_PPR_QA_TOP_K,
    )
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
    add_ablation_arguments(parser)
    parser.add_argument(
        "--allow-legacy-index-profile-adoption",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow adopting existing workspaces that contain index artifacts but lack "
            ".ablation_index_profile.json metadata."
        ),
    )
    parser.add_argument(
        "--require-existing-workspaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resolve prebuilt DocBench/SurGE workspaces under run-root and fail fast if they "
            "are missing. Disable only when you intentionally want fresh workspaces."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _apply_inferred_ablation_flag_defaults(
    args: argparse.Namespace, argv: list[str] | None
) -> None:
    raw_argv = list(argv or [])
    explicit_entity = _argv_mentions_option(
        raw_argv,
        "--enable-entity-disambiguation",
        "--enable_entity_disambiguation",
    )
    explicit_synonym = _argv_mentions_option(
        raw_argv,
        "--enable-synonym-linking",
        "--enable_synonym_linking",
    )
    explicit_multi_hop = _argv_mentions_option(
        raw_argv,
        "--enable-multi-hop",
        "--enable_multi_hop",
    )
    if explicit_entity and explicit_synonym and explicit_multi_hop:
        return

    inferred_candidates = [
        _infer_profile_flags_from_workspace_id(args.shared_workspace_id),
        _infer_profile_flags_from_workspace_id(args.surge_workspace_id),
    ]
    inferred = [item for item in inferred_candidates if item is not None]
    if not inferred:
        return
    first = inferred[0]
    if any(item != first for item in inferred[1:]):
        raise ValueError(
            "Workspace profile inference mismatch between shared/surge workspace ids. "
            "Pass explicit ablation flags to disambiguate."
        )

    if not explicit_entity:
        args.enable_entity_disambiguation = bool(first["enable_entity_disambiguation"])
    if not explicit_synonym:
        args.enable_synonym_linking = bool(first["enable_synonym_linking"])
    if not explicit_multi_hop:
        args.enable_multi_hop = bool(first["enable_multi_hop"])


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


def _repair_legacy_index_profile_if_needed(
    *,
    workspace_dir: str | Path,
    workspace_id: str,
    ablation_flags: Any,
    allow_legacy_adoption: bool,
) -> dict[str, Any] | None:
    if not allow_legacy_adoption:
        return None

    inferred = _infer_profile_flags_from_workspace_id(workspace_id)
    if inferred is None:
        return None

    target_entity = bool(getattr(ablation_flags, "enable_entity_disambiguation", False))
    target_synonym = bool(getattr(ablation_flags, "enable_synonym_linking", False))
    if (
        bool(inferred["enable_entity_disambiguation"]) != target_entity
        or bool(inferred["enable_synonym_linking"]) != target_synonym
    ):
        return None

    profile_path = Path(workspace_dir) / INDEX_PROFILE_FILE
    if not profile_path.exists():
        return None

    try:
        raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw_profile, dict):
        return None

    existing_entity = bool(raw_profile.get("enable_entity_disambiguation", True))
    existing_synonym = bool(raw_profile.get("enable_synonym_linking", False))
    if existing_entity == target_entity and existing_synonym == target_synonym:
        return None

    repaired_profile = dict(raw_profile)
    repaired_profile["profile_version"] = int(repaired_profile.get("profile_version", 1))
    repaired_profile["enable_entity_disambiguation"] = target_entity
    repaired_profile["enable_synonym_linking"] = target_synonym

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = profile_path.with_name(f"{INDEX_PROFILE_FILE}.bak.{timestamp}")
    shutil.copy2(profile_path, backup_path)
    profile_path.write_text(
        json.dumps(repaired_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "workspace_id": workspace_id,
        "profile_path": str(profile_path),
        "backup_path": str(backup_path),
        "previous_profile": raw_profile,
        "repaired_profile": repaired_profile,
    }


def _glob_dirs(pattern: Path) -> list[Path]:
    return sorted(
        Path(item).resolve() for item in glob.glob(str(pattern)) if Path(item).is_dir()
    )


def _resolve_workspace_candidates(*patterns: Path) -> list[Path]:
    matches: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for item in _glob_dirs(pattern):
            key = str(item)
            if key not in seen:
                seen.add(key)
                matches.append(item)
    return matches


def _select_unique_workspace_match(kind: str, workspace_id: str, matches: list[Path]) -> Path | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    ordered = sorted(matches, key=lambda item: len(item.parts), reverse=True)
    deepest = ordered[0]
    deepest_str = str(deepest)
    if all(
        deepest_str == str(item) or deepest_str.startswith(str(item) + os.sep)
        for item in ordered[1:]
    ):
        return deepest
    raise ValueError(
        f"Ambiguous {kind} workspace resolution for {workspace_id!r}. "
        f"Matches: {[str(item) for item in matches]}"
    )


def resolve_shared_workspace_layout(
    *,
    run_root: str | Path,
    workspace_id: str,
    require_existing: bool = True,
) -> dict[str, str]:
    run_root_path = Path(run_root).resolve()
    matches = _resolve_workspace_candidates(
        run_root_path
        / "_workspace_cache"
        / "docbench_shared"
        / "*"
        / "rag_workspaces"
        / workspace_id
        / workspace_id,
        run_root_path
        / "_workspace_cache"
        / "docbench_shared"
        / "*"
        / "rag_workspaces"
        / workspace_id,
        run_root_path / "*" / "evaluate_shared" / "rag_workspaces" / workspace_id / workspace_id,
        run_root_path / "*" / "evaluate_shared" / "rag_workspaces" / workspace_id,
        run_root_path / workspace_id / workspace_id,
        run_root_path / workspace_id,
    )
    workspace_dir = _select_unique_workspace_match("DocBench shared", workspace_id, matches)
    if workspace_dir is None:
        if require_existing:
            raise FileNotFoundError(
                "DocBench shared workspace not found under run_root. "
                f"run_root={run_root_path} workspace_id={workspace_id}"
            )
        workspace_dir = (run_root_path / workspace_id).resolve()

    working_dir_root = run_root_path
    state_dir: Path | None = None
    if workspace_dir.parent.name == workspace_id and workspace_dir.parent.parent.name == "rag_workspaces":
        working_dir_root = workspace_dir.parent
        state_dir = workspace_dir.parent.parent.parent
    elif workspace_dir.parent.name == "rag_workspaces":
        working_dir_root = workspace_dir.parent
        state_dir = workspace_dir.parent.parent

    payload = {
        "workspace_dir": str(workspace_dir),
        "working_dir_root": str(working_dir_root),
    }
    if state_dir is not None:
        payload["state_dir"] = str(state_dir)
        payload["manifest_file"] = str(state_dir / "shared_ingest_manifest.json")
        payload["failures_file"] = str(state_dir / "shared_ingest_failures.jsonl")
    return payload


def resolve_surge_workspace_layout(
    *,
    run_root: str | Path,
    workspace_id: str,
    require_existing: bool = True,
) -> dict[str, str]:
    run_root_path = Path(run_root).resolve()
    matches = _resolve_workspace_candidates(
        run_root_path
        / "_workspace_cache"
        / "surge_fast"
        / "*"
        / "rag_storage"
        / workspace_id
        / workspace_id,
        run_root_path
        / "_workspace_cache"
        / "surge_fast"
        / "*"
        / "rag_storage"
        / workspace_id,
        run_root_path / "*" / "evaluate_surge_fast" / "rag_storage" / workspace_id / workspace_id,
        run_root_path / "*" / "evaluate_surge_fast" / "rag_storage" / workspace_id,
        run_root_path / workspace_id / workspace_id,
        run_root_path / workspace_id,
    )
    workspace_dir = _select_unique_workspace_match("SurGE", workspace_id, matches)
    if workspace_dir is None:
        if require_existing:
            raise FileNotFoundError(
                "SurGE workspace not found under run_root. "
                f"run_root={run_root_path} workspace_id={workspace_id}"
            )
        workspace_dir = (run_root_path / workspace_id).resolve()

    storage_root = run_root_path
    state_dir: Path | None = None
    if workspace_dir.parent.name == workspace_id and workspace_dir.parent.parent.name == "rag_storage":
        storage_root = workspace_dir.parent
        state_dir = workspace_dir.parent.parent.parent
    elif workspace_dir.parent.name == "rag_storage":
        storage_root = workspace_dir.parent
        state_dir = workspace_dir.parent.parent

    payload = {
        "workspace_dir": str(workspace_dir),
        "storage_root": str(storage_root),
    }
    if state_dir is not None:
        payload["state_dir"] = str(state_dir)
    return payload


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


def _build_workspace_env(
    base_env: dict[str, str],
    args: argparse.Namespace,
    *,
    shared_layout: dict[str, str],
    surge_layout: dict[str, str],
) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT),
            str(PROJECT_ROOT.parent / "lightrag"),
            str(env.get("PYTHONPATH", "")).strip(),
        ]
    ).strip(os.pathsep)
    env["DOCBENCH_SHARED_WORKING_DIR_ROOT"] = str(shared_layout["working_dir_root"])
    env["SURGE_FAST_RAG_STORAGE_DIR"] = str(surge_layout["storage_root"])
    manifest_file = shared_layout.get("manifest_file")
    if manifest_file:
        env["DOCBENCH_SHARED_INGEST_MANIFEST_FILE"] = manifest_file
    failures_file = shared_layout.get("failures_file")
    if failures_file:
        env["DOCBENCH_SHARED_INGEST_FAILURES_FILE"] = failures_file
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
        "--enable-entity-disambiguation",
        "true" if bool(args.enable_entity_disambiguation) else "false",
        "--enable-synonym-linking",
        "true" if bool(args.enable_synonym_linking) else "false",
        "--enable-multi-hop",
        "true" if bool(args.enable_multi_hop) else "false",
        "--multi-hop-depth",
        str(args.multi_hop_depth),
        "--ppr-damping",
        str(args.ppr_damping),
        "--passage-node-weight",
        str(args.passage_node_weight),
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
        "--enable_rerank",
        "true" if experiment.get("enable_rerank", True) else "false",
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
    if experiment["query_mode"] == "ppr":
        cmd.extend(
            [
                "--ppr_top_k",
                str(experiment["ppr_top_k"]),
                "--ppr_qa_top_k",
                str(experiment["ppr_qa_top_k"]),
            ]
        )
    if args.allow_legacy_index_profile_adoption:
        cmd.append("--allow_legacy_index_profile_adoption")
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
        "--enable-entity-disambiguation",
        "true" if bool(args.enable_entity_disambiguation) else "false",
        "--enable-synonym-linking",
        "true" if bool(args.enable_synonym_linking) else "false",
        "--enable-multi-hop",
        "true" if bool(args.enable_multi_hop) else "false",
        "--multi-hop-depth",
        str(args.multi_hop_depth),
        "--ppr-damping",
        str(args.ppr_damping),
        "--passage-node-weight",
        str(args.passage_node_weight),
        "--keyword_fanout_mode",
        str(experiment["keyword_fanout_mode"]),
        "--entity_retrieval_mode",
        str(experiment["entity_retrieval_mode"]),
        "--chunk_retrieval_mode",
        str(experiment["chunk_retrieval_mode"]),
        "--exclude_synonym_edges",
        "true" if experiment["exclude_synonym_edges"] else "false",
        "--enable-rerank",
        "true" if experiment.get("enable_rerank", True) else "false",
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
    if experiment["query_mode"] == "ppr":
        cmd.extend(
            [
                "--ppr-top-k",
                str(experiment["ppr_top_k"]),
                "--ppr-qa-top-k",
                str(experiment["ppr_qa_top_k"]),
            ]
        )
    if args.allow_legacy_index_profile_adoption:
        cmd.append("--allow-legacy-index-profile-adoption")
    return cmd


def _finalize_experiment_for_task(
    *,
    task: str,
    experiment: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    finalized = dict(experiment)
    finalized["task"] = task
    finalized["enable_rerank"] = bool(finalized.get("enable_rerank", True))
    if finalized.get("query_mode") == "ppr":
        if task == "shared":
            finalized["ppr_top_k"] = int(
                finalized.get("ppr_top_k", args.shared_ppr_top_k)
            )
            finalized["ppr_qa_top_k"] = int(
                finalized.get("ppr_qa_top_k", args.shared_ppr_qa_top_k)
            )
        else:
            finalized["ppr_top_k"] = int(
                finalized.get("ppr_top_k", args.surge_ppr_top_k)
            )
            finalized["ppr_qa_top_k"] = int(
                finalized.get("ppr_qa_top_k", args.surge_ppr_qa_top_k)
            )
        _validate_ppr_controls(
            query_mode=str(finalized["query_mode"]),
            ppr_top_k=finalized.get("ppr_top_k"),
            ppr_qa_top_k=finalized.get("ppr_qa_top_k"),
            context=f"{task}:{finalized.get('name', finalized.get('query_mode', 'experiment'))}",
        )
    return finalized


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
        shared = [
            _finalize_experiment_for_task(task="shared", experiment=item, args=args)
            for item in full
        ]
        surge = [
            _finalize_experiment_for_task(task="surge", experiment=item, args=args)
            for item in full
            if not (
                item["query_mode"] != "ppr"
                and item.get("answer_context_mode") == "chunk_only_prompt"
            )
        ]
        return shared, surge

    shared = [
        _finalize_experiment_for_task(
            task="shared",
            experiment=item,
            args=args,
        )
        for item in build_reduced_experiment_matrix(
            "shared",
            shared_ppr_top_k=args.shared_ppr_top_k,
            shared_ppr_qa_top_k=args.shared_ppr_qa_top_k,
            surge_ppr_top_k=args.surge_ppr_top_k,
            surge_ppr_qa_top_k=args.surge_ppr_qa_top_k,
        )
    ]
    surge = [
        _finalize_experiment_for_task(
            task="surge",
            experiment=item,
            args=args,
        )
        for item in build_reduced_experiment_matrix(
            "surge",
            shared_ppr_top_k=args.shared_ppr_top_k,
            shared_ppr_qa_top_k=args.shared_ppr_qa_top_k,
            surge_ppr_top_k=args.surge_ppr_top_k,
            surge_ppr_qa_top_k=args.surge_ppr_qa_top_k,
        )
    ]
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    _apply_inferred_ablation_flag_defaults(args, raw_argv)
    args.ablation_flags = validate_ablation_flags(args, naming_style="hyphen")
    shared_experiments, surge_experiments = _selected_experiments(args)
    shared_layout = resolve_shared_workspace_layout(
        run_root=args.run_root,
        workspace_id=args.shared_workspace_id,
        require_existing=bool(args.require_existing_workspaces),
    )
    surge_layout = resolve_surge_workspace_layout(
        run_root=args.run_root,
        workspace_id=args.surge_workspace_id,
        require_existing=bool(args.require_existing_workspaces),
    )
    repaired_profiles: list[dict[str, Any]] = []
    shared_profile_repair = _repair_legacy_index_profile_if_needed(
        workspace_dir=shared_layout["workspace_dir"],
        workspace_id=args.shared_workspace_id,
        ablation_flags=args.ablation_flags,
        allow_legacy_adoption=bool(args.allow_legacy_index_profile_adoption),
    )
    if shared_profile_repair is not None:
        repaired_profiles.append(shared_profile_repair)
    surge_profile_repair = _repair_legacy_index_profile_if_needed(
        workspace_dir=surge_layout["workspace_dir"],
        workspace_id=args.surge_workspace_id,
        ablation_flags=args.ablation_flags,
        allow_legacy_adoption=bool(args.allow_legacy_index_profile_adoption),
    )
    if surge_profile_repair is not None:
        repaired_profiles.append(surge_profile_repair)
    for repaired in repaired_profiles:
        print(
            "Repaired legacy workspace index profile metadata:",
            repaired["profile_path"],
            "backup:",
            repaired["backup_path"],
            file=sys.stderr,
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
            "shared_workspace_layout": shared_layout,
            "surge_workspace_layout": surge_layout,
            "matrix_mode": args.matrix_mode,
            "allow_legacy_index_profile_adoption": bool(
                args.allow_legacy_index_profile_adoption
            ),
            "ablation_flags": args.ablation_flags.to_dict(),
            "repaired_legacy_index_profiles": repaired_profiles,
            "shared_experiments": shared_experiments,
            "surge_experiments": surge_experiments,
        },
    )

    env = _build_workspace_env(
        dict(os.environ),
        args,
        shared_layout=shared_layout,
        surge_layout=surge_layout,
    )

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
