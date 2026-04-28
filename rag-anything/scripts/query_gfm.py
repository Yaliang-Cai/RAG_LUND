#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick interactive test for GFM-RAG graph-neural queries on an existing workspace.

Prerequisites
-------------
1. Export the LightRAG workspace to GFM-RAG CSV format first:

   python scripts/export_lightrag_to_gfm.py \\
       --working-dir ./rag_storage/My_Graph \\
       --graph-name   My_Graph \\
       --data-dir     ./data

2. Set GFM_DATA_NAME (either in .env, or via --gfm-data-name below).
   GFM_DATA_DIR and GFM_MODEL_PATH have sensible defaults.

Usage examples
--------------
# GFM graph-neural retrieval (primary use case)
python scripts/query_gfm.py \\
    -w My_Graph \\
    -q "How did the network partition in region A cause the order service to fail?" \\
    --gfm-data-name My_Graph \\
    --trace

# GFM with a non-default model path
python scripts/query_gfm.py \\
    -w My_Graph \\
    -q "..." \\
    --gfm-data-name My_Graph \\
    --gfm-model-path /data/models/G-reasoner-34M \\
    --chunk-top-k 15 \\
    --trace

# Compare GFM against PPR baseline
python scripts/query_gfm.py \\
    -w My_Graph \\
    -q "Compare indexing strategies of LightRAG and HippoRAG2." \\
    --gfm-data-name My_Graph \\
    --mode ppr \\
    --trace

# Compare GFM against hybrid baseline
python scripts/query_gfm.py \\
    -w My_Graph \\
    -q "..." \\
    --gfm-data-name My_Graph \\
    --mode hybrid \\
    --trace

# Auto routing with gfm_multihop profile (enables PPR + hybrid; GFM commented out by default)
python scripts/query_gfm.py \\
    -w My_Graph \\
    -q "..." \\
    --mode auto --profile gfm_multihop \\
    --trace

# Full path for remote workspace (read-only cache dir)
python scripts/query_gfm.py \\
    -w docbench_shared_ablation_20260417_v0_v1_v2 \\
    --cache-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/ablation_20260417/v0_v1_v2/evaluate_shared/rag_workspaces/docbench_shared_ablation_20260417_v0_v1_v2 \\
    -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" \\
    --gfm-data-name docbench_shared_ablation_20260417_v0_v1_v2 \\
    --trace
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

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
    GFM_DATA_DIR as _DEFAULT_GFM_DATA_DIR,
    GFM_DATA_NAME as _DEFAULT_GFM_DATA_NAME,
    GFM_MODEL_PATH as _DEFAULT_GFM_MODEL_PATH,
)


def _parse_args():
    p = argparse.ArgumentParser(description="GFM graph-neural query tester")
    p.add_argument("-w", "--workspace", required=True,
                   help="Workspace ID (must already be indexed and exported via export_lightrag_to_gfm.py)")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Writable directory for KV stores. Useful when the workspace path is read-only. "
             "Defaults to the workspace path.",
    )
    p.add_argument("-q", "--query", required=True, help="Question to ask")
    p.add_argument(
        "--mode",
        default="gfm",
        choices=["gfm", "ppr", "hybrid", "mix", "naive", "auto"],
        help="Query mode (default: gfm). Use ppr/hybrid/mix/naive for baseline comparison, "
             "'auto' for LLM-based profile routing.",
    )
    p.add_argument(
        "--profile",
        default=None,
        choices=["precise", "semantic", "local", "multihop", "full", "gfm_multihop"],
        help="(auto mode only) Force a specific retrieval profile, bypassing LLM classification. "
             "'gfm_multihop' runs PPR + hybrid by default; uncomment 'gfm' lines in profiles.py to include GFM.",
    )

    # GFM-specific overrides
    gfm = p.add_argument_group("GFM overrides")
    gfm.add_argument(
        "--gfm-data-dir",
        default=None,
        help=f"GFM-RAG root data directory (default: '{_DEFAULT_GFM_DATA_DIR}' from constants). "
             "Override with the --data-dir value used in export_lightrag_to_gfm.py.",
    )
    gfm.add_argument(
        "--gfm-data-name",
        default=None,
        help=f"GFM graph name / GFM_DATA_NAME (default: '{_DEFAULT_GFM_DATA_NAME}' from constants). "
             "Must match the --graph-name used in export_lightrag_to_gfm.py.",
    )
    gfm.add_argument(
        "--gfm-model-path",
        default=None,
        help=f"HuggingFace model ID or local path for GFM reasoner "
             f"(default: '{_DEFAULT_GFM_MODEL_PATH}' from constants).",
    )

    # Shared retrieval knobs
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--chunk-top-k", type=int, default=DEFAULT_CHUNK_TOP_K,
                   help="Chunks retrieved by GFM / final window after reranking (default: %(default)s)")
    p.add_argument("--naive-top-k", type=int, default=DEFAULT_NAIVE_TOP_K,
                   help="Naive VDB retrieval count for non-GFM modes (default: %(default)s)")

    # PPR knobs (used when --mode ppr)
    ppr = p.add_argument_group("PPR options (--mode ppr only)")
    ppr.add_argument("--ppr-damping", type=float, default=DEFAULT_PPR_DAMPING)
    ppr.add_argument("--ppr-top-k", type=int, default=DEFAULT_PPR_TOP_K)
    ppr.add_argument("--passage-node-weight", type=float, default=DEFAULT_PASSAGE_NODE_WEIGHT)
    ppr.add_argument("--recognition-top-k", type=int, default=DEFAULT_RECOGNITION_TOP_K,
                     help="Recognition memory relation top-k (0 = disabled)")
    ppr.add_argument("--linking-top-k", type=int, default=DEFAULT_LINKING_TOP_K,
                     help="Max entity seeds from recognition memory (HippoRAG2 link_top_k)")
    ppr.add_argument("--ppr-qa-top-k", type=int, default=DEFAULT_PPR_QA_TOP_K,
                     help="Chunks fed to LLM after PPR retrieval (HippoRAG2 qa_top_k)")

    p.add_argument("--no-rerank", action="store_true", help="Disable chunk reranking")
    p.add_argument("--trace", action="store_true", help="Print full trace JSON")
    return p.parse_args()


def _apply_gfm_overrides(args):
    """Patch query.py's module-level GFM constants if CLI overrides are provided."""
    import raganything.query as _qmod

    if args.gfm_data_dir is not None:
        _qmod.GFM_DATA_DIR = args.gfm_data_dir
    if args.gfm_data_name is not None:
        _qmod.GFM_DATA_NAME = args.gfm_data_name
    if args.gfm_model_path is not None:
        _qmod.GFM_MODEL_PATH = args.gfm_model_path

    # Report effective GFM config
    return {
        "data_dir":   _qmod.GFM_DATA_DIR,
        "data_name":  _qmod.GFM_DATA_NAME,
        "model_path": _qmod.GFM_MODEL_PATH,
    }


async def run(args):
    from raganything.services.local_rag import LocalRagService, LocalRagSettings

    print(f"\n[Init] Loading LocalRagSettings from env...")
    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)

    workspace_id = args.workspace
    query = args.query

    # Apply GFM overrides before the singleton can be created
    gfm_cfg = _apply_gfm_overrides(args)

    query_kwargs = dict(
        mode=args.mode,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        naive_top_k=args.naive_top_k,
        enable_rerank=not args.no_rerank,
        rerank_score_scope="all",
    )
    if args.mode == "ppr":
        query_kwargs.update(
            ppr_damping=args.ppr_damping,
            ppr_top_k=args.ppr_top_k,
            passage_node_weight=args.passage_node_weight,
            recognition_top_k=max(0, args.recognition_top_k),
            linking_top_k=max(0, args.linking_top_k),
            ppr_qa_top_k=max(1, args.ppr_qa_top_k),
        )
    if args.mode == "auto" and args.profile:
        query_kwargs["profile"] = args.profile

    # Print run config
    print(f"\n[Query] workspace={workspace_id}")
    print(f"        mode={args.mode}  rerank={not args.no_rerank}")
    print(f"        top_k={args.top_k}  chunk_top_k={args.chunk_top_k}")
    if args.mode == "gfm":
        print(f"        gfm_data_dir={gfm_cfg['data_dir']}")
        print(f"        gfm_data_name={gfm_cfg['data_name'] or '(not set — will fail)'}")
        print(f"        gfm_model_path={gfm_cfg['model_path']}")
    if args.mode == "ppr":
        print(f"        ppr_damping={args.ppr_damping}  ppr_top_k={args.ppr_top_k}")
        print(f"        passage_node_weight={args.passage_node_weight}")
        print(f"        recognition_top_k={query_kwargs['recognition_top_k']}")
        print(f"        linking_top_k={query_kwargs['linking_top_k']}  ppr_qa_top_k={query_kwargs['ppr_qa_top_k']}")
    if args.mode == "auto" and args.profile:
        print(f"        profile={args.profile} (classifier bypassed)")
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
        if trace.get("mode") == "gfm":
            # GFM trace: {"mode": "gfm", "chunks_retrieved": N}
            print(f"\n[GFM trace]  chunks_retrieved={trace.get('chunks_retrieved')}")
        elif "routing" in trace:
            # auto mode routing trace
            rt = trace["routing"]
            print(f"\n[Routing trace]  profile={rt.get('profile')}  confidence={rt.get('confidence')}  "
                  f"paths={rt.get('paths_activated')}")
            print(f"  chunks_after_rrf={rt.get('chunks_after_rrf')}  "
                  f"chunks_after_rerank={rt.get('chunks_after_rerank')}  "
                  f"final={rt.get('chunks_after_threshold')}")
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
            print(json.dumps(trace, ensure_ascii=False, indent=2)[:8000])


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(args))
