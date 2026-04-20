#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick interactive test for PPR global-mode queries on an existing workspace.

Usage:
  python scripts/query_ppr.py -w <workspace_id> -q "your question here"

Optional overrides:
  --mode ppr          # ppr (global PPR + recognition memory), global, hybrid, etc.
  --top-k 40
  --chunk-top-k 20
  --ppr-top-k 50
  --ppr-damping 0.5
  --passage-node-weight 1.0
  --recognition-top-k 20   # 0 = disable recognition memory
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
    DEFAULT_PPR_DAMPING,
    DEFAULT_PPR_TOP_K,
    DEFAULT_PASSAGE_NODE_WEIGHT,
    DEFAULT_RECOGNITION_TOP_K,
    DEFAULT_ENABLE_RERANK,
)


def _parse_args():
    p = argparse.ArgumentParser(description="PPR global-mode query tester")
    p.add_argument("-w", "--workspace", required=True, help="Workspace ID (must already be indexed)")
    p.add_argument("-q", "--query", required=True, help="Question to ask")
    p.add_argument(
        "--mode",
        default="ppr",
        choices=["ppr", "ppr_local", "global", "local", "hybrid", "mix", "naive", "rrf", "bypass"],
        help="Query mode (default: ppr = global PPR with recognition memory)",
    )
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--chunk-top-k", type=int, default=DEFAULT_CHUNK_TOP_K)
    p.add_argument("--ppr-damping", type=float, default=DEFAULT_PPR_DAMPING)
    p.add_argument("--ppr-top-k", type=int, default=DEFAULT_PPR_TOP_K)
    p.add_argument("--passage-node-weight", type=float, default=DEFAULT_PASSAGE_NODE_WEIGHT)
    p.add_argument(
        "--recognition-top-k",
        type=int,
        default=DEFAULT_RECOGNITION_TOP_K,
        help="Recognition memory relation top-k (0 = disabled)",
    )
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-multi-hop", action="store_true", help="Disable PPR multi-hop (V3 off)")
    p.add_argument("--trace", action="store_true", help="Print retrieval trace JSON")
    return p.parse_args()


async def run(args):
    print(f"\n[Init] Loading LocalRagSettings from env...")
    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)

    workspace_id = args.workspace
    query = args.query
    enable_multi_hop = not args.no_multi_hop

    query_kwargs = dict(
        mode=args.mode,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        enable_rerank=not args.no_rerank,
        rerank_score_scope="all",
        enable_multi_hop=enable_multi_hop,
        ppr_damping=args.ppr_damping,
        ppr_top_k=args.ppr_top_k,
        passage_node_weight=args.passage_node_weight,
    )
    if args.mode == "ppr":
        query_kwargs["recognition_top_k"] = max(0, args.recognition_top_k)

    print(f"\n[Query] workspace={workspace_id}")
    print(f"        mode={args.mode}  multi_hop={enable_multi_hop}  rerank={not args.no_rerank}")
    print(f"        top_k={args.top_k}  chunk_top_k={args.chunk_top_k}")
    print(f"        ppr_damping={args.ppr_damping}  ppr_top_k={args.ppr_top_k}")
    print(f"        passage_node_weight={args.passage_node_weight}")
    if args.mode == "ppr":
        print(f"        recognition_top_k={query_kwargs['recognition_top_k']}")
    print(f"\n[Question] {query}\n")
    print("-" * 72)

    response = await service.query_with_trace(
        workspace_id=workspace_id,
        query=query,
        **query_kwargs,
    )

    answer = response.get("answer", "")
    trace = response.get("trace", {})

    print("\n[Answer]")
    print(answer)

    if trace:
        chunks = trace.get("chunks", [])
        entities = trace.get("entities", [])
        relations = trace.get("relations", [])
        print(f"\n[Trace summary]  chunks={len(chunks)}  entities={len(entities)}  relations={len(relations)}")
        if chunks:
            print(f"  Top chunk score: {chunks[0].get('score', 'n/a') if isinstance(chunks[0], dict) else 'n/a'}")
        if args.trace:
            print("\n[Trace JSON]")
            print(json.dumps(trace, ensure_ascii=False, indent=2)[:8000])  # cap at 8k chars


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(args))
