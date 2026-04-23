#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-query retrieval debugger for graphbm25_20260421 workspaces.

Edit the constants below and run on Linux:
python scripts/debug_retrieval_ablation.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from raganything.services.local_rag import LocalRagService, LocalRagSettings
from evaluate_local.run_retrieval_ablation import build_reduced_experiment_matrix


# ---------------------------------------------------------------------------
# Editable experiment config
# ---------------------------------------------------------------------------
RUN_ROOT = "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421"
WORKING_DIR = None  # None -> f"{RUN_ROOT}/{WORKSPACE_ID}"

WORKSPACE_ID = "docbench_shared_graphbm25_20260421_v0_v1_v2"
# WORKSPACE_ID = "surge_fast_graphbm25_20260421_v0_v1_v2"
DATASET = "docbench"  # docbench | surge
RUN_GROUP_MATRIX = False

QUERY = "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?"
QUERY_MODE = "hybrid"  # hybrid | ppr
TOP_K = 40
CHUNK_TOP_K = 20
SURGE_CHUNK_TOP_K = 0
MAX_TOTAL_TOKENS = 45000
RECOGNITION_TOP_K = 20

KEYWORD_FANOUT_MODE = "joined"  # joined | per_keyword_rrf
RETRIEVAL_MODE = "dense"  # dense | hybrid
KG_CHUNK_SELECTION_SOURCE = "truncated"  # truncated | untruncated; hybrid only
EXCLUDE_SYNONYM_EDGES = True

ANSWER_CONTEXT_MODE = "kg_prompt"  # hybrid only; ppr is always chunk_only_prompt
BYPASS_QUERY_CACHE = True
BYPASS_KEYWORDS_CACHE = False

PRINT_FULL_TRACE_JSON = False


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _print_section(title: str, payload: Any) -> None:
    print(f"\n[{title}]")
    if payload is None:
        print("None")
        return
    if isinstance(payload, str):
        print(payload)
        return
    print(_pretty(payload))


def _resolve_working_dir(
    run_root: str,
    workspace_id: str,
    working_dir: str | None,
) -> str:
    if working_dir:
        return str(working_dir)
    if str(run_root).startswith("/"):
        return str(PurePosixPath(run_root) / workspace_id)
    return str(Path(run_root) / workspace_id)


def _dataset_key(dataset: str) -> str:
    normalized = str(dataset or "").strip().lower()
    if normalized in {"docbench", "shared"}:
        return "docbench"
    if normalized == "surge":
        return "surge"
    raise ValueError(f"Unknown DATASET={dataset!r}; expected docbench or surge")


def _debug_group_matrix(dataset: str) -> list[dict[str, Any]]:
    dataset_key = _dataset_key(dataset)
    task = "shared" if dataset_key == "docbench" else "surge"
    return build_reduced_experiment_matrix(task)


def _single_group(dataset: str) -> dict[str, Any]:
    dataset_key = _dataset_key(dataset)
    item: dict[str, Any] = {
        "name": "single",
        "query_mode": QUERY_MODE,
        "keyword_fanout_mode": KEYWORD_FANOUT_MODE,
        "retrieval_mode": RETRIEVAL_MODE,
        "entity_retrieval_mode": RETRIEVAL_MODE,
        "chunk_retrieval_mode": RETRIEVAL_MODE,
        "exclude_synonym_edges": EXCLUDE_SYNONYM_EDGES,
    }
    if QUERY_MODE == "ppr":
        item["exclude_synonym_edges"] = False
        item["answer_context_mode"] = "chunk_only_prompt"
    else:
        item["kg_chunk_selection_source"] = KG_CHUNK_SELECTION_SOURCE
        if dataset_key == "docbench":
            item["answer_context_mode"] = ANSWER_CONTEXT_MODE
    return item


def _query_kwargs(group: dict[str, Any]) -> dict[str, Any]:
    dataset_key = _dataset_key(DATASET)
    query_mode = str(group["query_mode"])
    kwargs: dict[str, Any] = {
        "mode": query_mode,
        "top_k": TOP_K,
        "chunk_top_k": CHUNK_TOP_K if dataset_key == "docbench" else SURGE_CHUNK_TOP_K,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "recognition_top_k": RECOGNITION_TOP_K,
        "keyword_fanout_mode": str(group["keyword_fanout_mode"]),
        "entity_qdrant_retrieval_mode": str(group["entity_retrieval_mode"]),
        "chunk_qdrant_retrieval_mode": str(group["chunk_retrieval_mode"]),
        "exclude_synonym_edges": bool(group["exclude_synonym_edges"]),
        "bypass_query_cache": BYPASS_QUERY_CACHE,
        "bypass_keywords_cache": BYPASS_KEYWORDS_CACHE,
    }
    if "kg_chunk_selection_source" in group:
        kwargs["kg_chunk_selection_source"] = str(group["kg_chunk_selection_source"])
    if query_mode == "ppr":
        kwargs["answer_context_mode"] = "chunk_only_prompt"
    elif dataset_key == "docbench":
        kwargs["answer_context_mode"] = str(
            group.get("answer_context_mode", ANSWER_CONTEXT_MODE)
        )
    return kwargs


def _chunk_debug_summary(rerank_debug: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(rerank_debug, dict):
        return {}
    return {
        "count_input": rerank_debug.get("count_input"),
        "count_after_rerank": rerank_debug.get("count_after_rerank"),
        "count_after_threshold": rerank_debug.get("count_after_threshold"),
        "count_after_chunk_top_k": rerank_debug.get("count_after_chunk_top_k"),
        "count_final": rerank_debug.get("count_final"),
        "chunk_ids_after_chunk_top_k": rerank_debug.get(
            "chunk_ids_after_chunk_top_k"
        ),
        "chunk_ids_final": rerank_debug.get("chunk_ids_final"),
        "scores_final": rerank_debug.get("scores_final"),
    }


async def _run_group(
    *,
    service: LocalRagService,
    resolved_working_dir: str,
    group: dict[str, Any],
) -> None:
    kwargs = _query_kwargs(group)
    print("\n" + "=" * 88)
    print(f"[Group] {group['name']}")
    print(f"query_kwargs={_pretty(kwargs)}")
    response = await service.query_with_trace(
        workspace_id=WORKSPACE_ID,
        query=QUERY,
        working_dir=resolved_working_dir,
        **kwargs,
    )

    answer = response.get("answer", "")
    trace = response.get("trace", {}) if isinstance(response, dict) else {}
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else {}
    data = trace.get("data", {}) if isinstance(trace, dict) else {}
    retrieval_debug = metadata.get("retrieval_debug", {}) if isinstance(metadata, dict) else {}

    _print_section("Answer", answer)
    _print_section("Keywords", metadata.get("keywords"))
    _print_section("ProcessingInfo", metadata.get("processing_info"))

    if kwargs["mode"] == "hybrid":
        _print_section("LocalSearch", retrieval_debug.get("local_search"))
        _print_section("GlobalSearch", retrieval_debug.get("global_search"))
        _print_section("KGRerank", retrieval_debug.get("kg_rerank"))
    else:
        _print_section("PPR", retrieval_debug.get("ppr"))

    _print_section(
        "ChunkRerankSummary",
        _chunk_debug_summary(metadata.get("rerank_chunk_debug")),
    )
    _print_section("ChunkRerank", metadata.get("rerank_chunk_debug"))
    _print_section("FinalEntities", data.get("entities"))
    _print_section("FinalRelations", data.get("relationships"))
    _print_section("FinalChunks", data.get("chunks"))

    if PRINT_FULL_TRACE_JSON:
        _print_section("FullTrace", trace)


async def main() -> None:
    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)
    resolved_working_dir = _resolve_working_dir(RUN_ROOT, WORKSPACE_ID, WORKING_DIR)
    groups = _debug_group_matrix(DATASET) if RUN_GROUP_MATRIX else [_single_group(DATASET)]

    print("[Config]")
    print(f"run_root={RUN_ROOT}")
    print(f"workspace_id={WORKSPACE_ID}")
    print(f"working_dir={resolved_working_dir}")
    print(f"dataset={DATASET}")
    print(f"run_group_matrix={RUN_GROUP_MATRIX}")
    print(f"query={QUERY}")
    print(f"groups={[group['name'] for group in groups]}")

    for group in groups:
        await _run_group(
            service=service,
            resolved_working_dir=resolved_working_dir,
            group=group,
        )


if __name__ == "__main__":
    asyncio.run(main())
