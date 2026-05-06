#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick interactive test for PPR global-mode queries on an existing workspace.

Usage:
python scripts/query_ppr.py  -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode ppr

# LLM 自动路由（分类器决定 profile）
python scripts/query_ppr.py -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode auto --trace

# 强制指定 profile=precise（绕过分类器）
python scripts/query_ppr.py -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode auto --profile precise --trace

# agentic 模式：自适应复杂度路由（simple/medium/complex 三轨）
python scripts/query_ppr.py -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode agentic --trace

# agentic 模式 + Phoenix 监控（浏览器打开 http://localhost:6006）
python scripts/query_ppr.py -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode agentic --trace --phoenix

# ppr 模式对比基线
python scripts/query_ppr.py -w docbench_shared_ablation_20260417_v0_v1_v2 --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" --mode ppr --trace



Optional overrides:
  --mode ppr          # ppr (global PPR + recognition memory), global, hybrid, etc.
  --top-k 40
  --chunk-top-k 10
  --naive-top-k 20
  --ppr-top-k 50
  # LLM 自动路由（不指定 profile，让分类器决定）
  python scripts/query_ppr.py \
    -w My_Graph \
    -q "Describe the overall architecture of LightRAG." \
    --mode auto \
    --trace

  # 强制指定 profile（绕过 LLM 分类器，适合消融实验）
  python scripts/query_ppr.py \
    -w My_Graph \
    -q "What is the impact scope of CVE-2026-001?" \
    --mode auto \
    --profile precise \
    --trace

  # 与 ppr 模式对比
  python scripts/query_ppr.py \
    -w My_Graph \
    -q "..." \
    --mode ppr \
    --trace

  --ppr-damping 0.5
  --passage-node-weight 1.0
  --recognition-top-k 20   # 0 = disable recognition memory
  --linking-top-k 5        # max entity seeds from recognition memory (HippoRAG2 link_top_k)
  --ppr-qa-top-k 5         # chunks fed to LLM (HippoRAG2 qa_top_k)
  --no-rerank
  --trace               # print full trace JSON

Examples:
  # Default PPR global mode:
  python scripts/query_ppr.py -w My_Graph -q "What is the main contribution of this paper?"

  # Compare with hybrid mode:
  python scripts/query_ppr.py -w My_Graph -q "..." --mode hybrid

  # PPR without recognition memory:
  python scripts/query_ppr.py -w My_Graph -q "..." --recognition-top-k 0 --trace
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from raganything.services.local_rag import LocalRagService, LocalRagSettings
from raganything.constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_NAIVE_TOP_K,
    DEFAULT_PPR_DAMPING,
    DEFAULT_PPR_TOP_K,
    DEFAULT_PASSAGE_NODE_WEIGHT,
    DEFAULT_RECOGNITION_TOP_K,
    DEFAULT_LINKING_TOP_K,
    DEFAULT_PPR_QA_TOP_K,
    DEFAULT_ENABLE_RERANK,
    DEFAULT_ENABLE_KG_RERANK,
)


def _parse_args():
    p = argparse.ArgumentParser(description="PPR global-mode query tester")
    p.add_argument("-w", "--workspace", required=True, help="Workspace ID (must already be indexed)")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Writable directory for KV stores and LLM cache. "
             "Useful when the workspace path is read-only (e.g. another user's directory). "
             "Defaults to the workspace path.",
    )
    p.add_argument("-q", "--query", required=True, help="Question to ask")
    p.add_argument(
        "--mode",
        default="ppr",
        choices=["ppr", "ppr_local", "global", "local", "hybrid", "mix", "naive", "rrf", "bypass", "auto", "agentic"],
        help="Query mode (default: ppr = global PPR with recognition memory). "
             "Use 'auto' for LLM-based profile routing. "
             "Use 'agentic' for adaptive complexity-aware agentic retrieval.",
    )
    p.add_argument(
        "--profile",
        default=None,
        choices=["precise", "local", "multihop", "descriptive", "full"],
        help="(auto mode only) Force a specific retrieval profile, bypassing LLM classification.",
    )
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--chunk-top-k", type=int, default=DEFAULT_CHUNK_TOP_K,
                   help="Final chunk window size after reranking (default: %(default)s)")
    p.add_argument("--naive-top-k", type=int, default=DEFAULT_NAIVE_TOP_K,
                   help="Naive VDB retrieval count for mix/naive modes (default: %(default)s)")
    p.add_argument("--ppr-damping", type=float, default=DEFAULT_PPR_DAMPING)
    p.add_argument("--ppr-top-k", type=int, default=DEFAULT_PPR_TOP_K)
    p.add_argument("--passage-node-weight", type=float, default=DEFAULT_PASSAGE_NODE_WEIGHT)
    p.add_argument(
        "--recognition-top-k",
        type=int,
        default=DEFAULT_RECOGNITION_TOP_K,
        help="Recognition memory relation top-k (0 = disabled)",
    )
    p.add_argument(
        "--linking-top-k",
        type=int,
        default=DEFAULT_LINKING_TOP_K,
        help="Max entity seeds output by recognition memory (HippoRAG2 link_top_k, 0 = no cap)",
    )
    p.add_argument(
        "--ppr-qa-top-k",
        type=int,
        default=DEFAULT_PPR_QA_TOP_K,
        help="Chunks fed to LLM after PPR retrieval (HippoRAG2 qa_top_k)",
    )
    p.add_argument("--no-rerank", action="store_true", help="Disable chunk reranking")
    p.add_argument("--no-kg-rerank", action="store_true",
                   help="Disable entity/relation KG reranking (independent of --no-rerank)")
    p.add_argument("--trace", action="store_true", help="Print retrieval trace JSON")
    p.add_argument("--phoenix", action="store_true",
                   help="Enable Arize Phoenix local OTEL tracing (http://localhost:6006). "
                        "Requires: pip install 'raganything[agentic]'")
    return p.parse_args()


async def run(args):
    if args.phoenix:
        from raganything.observability import setup_phoenix
        setup_phoenix()
        print("[Phoenix] Tracing enabled → http://localhost:6006")

    print(f"\n[Init] Loading LocalRagSettings from env...")
    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)

    workspace_id = args.workspace
    query = args.query
    query_kwargs = dict(
        mode=args.mode,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        naive_top_k=args.naive_top_k,
        enable_rerank=not args.no_rerank,
        enable_kg_rerank=not args.no_kg_rerank,
        rerank_score_scope="all",
        ppr_damping=args.ppr_damping,
        ppr_top_k=args.ppr_top_k,
        passage_node_weight=args.passage_node_weight,
    )
    if args.mode == "ppr":
        query_kwargs["recognition_top_k"] = max(0, args.recognition_top_k)
        query_kwargs["linking_top_k"] = max(0, args.linking_top_k)
        query_kwargs["ppr_qa_top_k"] = max(1, args.ppr_qa_top_k)
    if args.mode == "auto" and args.profile:
        query_kwargs["profile"] = args.profile
    if args.mode == "agentic":
        query_kwargs["return_trace"] = True  # always fetch trace; display controlled by --trace flag

    print(f"\n[Query] workspace={workspace_id}")
    print(f"        mode={args.mode}  rerank={not args.no_rerank}  kg_rerank={not args.no_kg_rerank}")
    print(f"        top_k={args.top_k}  chunk_top_k={args.chunk_top_k}  naive_top_k={args.naive_top_k}")
    print(f"        ppr_damping={args.ppr_damping}  ppr_top_k={args.ppr_top_k}")
    print(f"        passage_node_weight={args.passage_node_weight}")
    if args.mode == "ppr":
        print(f"        recognition_top_k={query_kwargs['recognition_top_k']}")
        print(f"        linking_top_k={query_kwargs['linking_top_k']}  ppr_qa_top_k={query_kwargs['ppr_qa_top_k']}")
    print(f"\n[Question] {query}\n")
    print("-" * 72)

    response = await service.query_with_trace(
        workspace_id=workspace_id,
        query=query,
        working_dir=args.cache_dir or None,
        **query_kwargs,
    )

    answer = response.get("answer", "")
    trace = response.get("trace", {})

    print("\n[Answer]")
    print(answer)

    if trace:
        if args.mode == "agentic":
            # agentic mode: V4 trace fields
            confidence = response.get("confidence", "?")
            grounded = response.get("grounded", "?")
            profile = trace.get("profile", "?")
            cache_hit = trace.get("router_cache_hit", False)
            retrieve_cycles = trace.get("retrieve_cycles_used", 0)
            check_cycles = trace.get("check_cycles_used", 0)
            print(f"\n[Agentic trace]  confidence={confidence}  grounded={grounded}  profile={profile}  cache_hit={cache_hit}")
            print(f"  retrieve_cycles={retrieve_cycles}  check_cycles={check_cycles}")
            rewrite_history = trace.get("rewrite_history", [])
            if len(rewrite_history) > 1:
                print(f"  rewrite_history: {rewrite_history}")
            sub_qs = trace.get("sub_questions")
            if sub_qs:
                print(f"  sub_questions: {sub_qs}")
        elif "routing" in trace:
            # auto mode: routing trace
            rt = trace["routing"]
            print(f"\n[Routing trace]  profile={rt.get('profile')}  confidence={rt.get('confidence')}  paths={rt.get('paths_activated')}")
            print(f"  chunks_after_rrf={rt.get('chunks_after_rrf')}  chunks_after_rerank={rt.get('chunks_after_rerank')}  final={rt.get('chunks_after_threshold')}")
            lpp = rt.get("latency_per_path", {})
            print(f"  latency_per_path: {lpp}")
        else:
            trace_data = trace.get("data", trace)
            chunks = trace_data.get("chunks", [])
            entities = trace_data.get("entities", [])
            relations = trace_data.get("relations", [])
            print(f"\n[Trace summary]  chunks={len(chunks)}  entities={len(entities)}  relations={len(relations)}")
            if chunks:
                print(f"  Top chunk score: {chunks[0].get('score', 'n/a') if isinstance(chunks[0], dict) else 'n/a'}")
        if args.trace:
            print("\n[Trace JSON]")
            print(json.dumps(trace, ensure_ascii=False, indent=2)[:8000])  # cap at 8k chars


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(args))
