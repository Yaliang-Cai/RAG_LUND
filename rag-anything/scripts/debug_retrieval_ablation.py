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


# ---------------------------------------------------------------------------
# Editable experiment config
# ---------------------------------------------------------------------------
RUN_ROOT = "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421"
WORKING_DIR = None  # None -> f"{RUN_ROOT}/{WORKSPACE_ID}"

WORKSPACE_ID = "docbench_shared_graphbm25_20260421_v0_v1_v2"
# WORKSPACE_ID = "surge_fast_graphbm25_20260421_v0_v1_v2"

QUERY = "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?"
QUERY_MODE = "hybrid"  # hybrid | ppr
TOP_K = 40
CHUNK_TOP_K = 20
RECOGNITION_TOP_K = 20

KEYWORD_FANOUT_MODE = "joined"  # joined | per_keyword_rrf
ENTITY_RETRIEVAL_MODE = "dense"  # dense | bm25 | hybrid
CHUNK_RETRIEVAL_MODE = "dense"  # dense | bm25 | hybrid
EXCLUDE_SYNONYM_EDGES = False

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


def _query_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "mode": QUERY_MODE,
        "top_k": TOP_K,
        "chunk_top_k": CHUNK_TOP_K,
        "recognition_top_k": RECOGNITION_TOP_K,
        "keyword_fanout_mode": KEYWORD_FANOUT_MODE,
        "entity_qdrant_retrieval_mode": ENTITY_RETRIEVAL_MODE,
        "chunk_qdrant_retrieval_mode": CHUNK_RETRIEVAL_MODE,
        "exclude_synonym_edges": EXCLUDE_SYNONYM_EDGES,
        "bypass_query_cache": BYPASS_QUERY_CACHE,
        "bypass_keywords_cache": BYPASS_KEYWORDS_CACHE,
    }
    if QUERY_MODE == "ppr":
        kwargs["answer_context_mode"] = "chunk_only_prompt"
    else:
        kwargs["answer_context_mode"] = ANSWER_CONTEXT_MODE
    return kwargs


async def main() -> None:
    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)
    resolved_working_dir = _resolve_working_dir(RUN_ROOT, WORKSPACE_ID, WORKING_DIR)

    print("[Config]")
    print(f"run_root={RUN_ROOT}")
    print(f"workspace_id={WORKSPACE_ID}")
    print(f"working_dir={resolved_working_dir}")
    print(f"query_mode={QUERY_MODE}")
    print(f"query={QUERY}")
    print(f"query_kwargs={_pretty(_query_kwargs())}")

    response = await service.query_with_trace(
        workspace_id=WORKSPACE_ID,
        query=QUERY,
        working_dir=resolved_working_dir,
        **_query_kwargs(),
    )

    answer = response.get("answer", "")
    trace = response.get("trace", {}) if isinstance(response, dict) else {}
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else {}
    data = trace.get("data", {}) if isinstance(trace, dict) else {}
    retrieval_debug = metadata.get("retrieval_debug", {}) if isinstance(metadata, dict) else {}

    _print_section("Answer", answer)
    _print_section("Keywords", metadata.get("keywords"))
    _print_section("ProcessingInfo", metadata.get("processing_info"))

    if QUERY_MODE == "hybrid":
        _print_section("LocalSearch", retrieval_debug.get("local_search"))
        _print_section("GlobalSearch", retrieval_debug.get("global_search"))
        _print_section("KGRerank", retrieval_debug.get("kg_rerank"))
    else:
        _print_section("PPR", retrieval_debug.get("ppr"))

    _print_section("ChunkRerank", metadata.get("rerank_chunk_debug"))
    _print_section("FinalEntities", data.get("entities"))
    _print_section("FinalRelations", data.get("relationships"))
    _print_section("FinalChunks", data.get("chunks"))

    if PRINT_FULL_TRACE_JSON:
        _print_section("FullTrace", trace)


if __name__ == "__main__":
    asyncio.run(main())
