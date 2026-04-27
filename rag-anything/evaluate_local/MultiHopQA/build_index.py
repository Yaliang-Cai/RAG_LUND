#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1 of the MultiHopQA evaluation pipeline — build a LightRAG workspace
from the subset corpus that corresponds to the sampled evaluation questions.

Run this BEFORE evaluate_multihop.py.  Use the same --n-samples and --seed
in both scripts so the evaluation queries match the indexed corpus.

Usage:
    python evaluate_local/MultiHopQA/build_index.py \\
        --dataset hotpotqa \\
        --n-samples 500 \\
        --workspace hotpotqa_500_seed42 \\
        --working-dir /data/y50056788/.../rag_workspaces/hotpotqa_500_seed42

    python evaluate_local/MultiHopQA/build_index.py \\
        --dataset musique \\
        --n-samples 500 \\
        --workspace musique_500_seed42 \\
        --working-dir /data/y50056788/.../rag_workspaces/musique_500_seed42
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
_lightrag_root = _project_root.parent / "lightrag"
for p in (_project_root, _lightrag_root):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv
load_dotenv()

VALID_DATASETS = ("hotpotqa", "musique", "2wiki")


async def main(args: argparse.Namespace) -> None:
    from raganything.services.local_rag import LocalRagService, LocalRagSettings
    from evaluate_local.MultiHopQA.dataset_adapters import (
        extract_corpus_hotpotqa,
        extract_corpus_musique,
        extract_corpus_2wiki,
    )

    extractors = {
        "hotpotqa": extract_corpus_hotpotqa,
        "musique":  extract_corpus_musique,
        "2wiki":    extract_corpus_2wiki,
    }

    print(f"[build_index] Extracting corpus: {args.dataset} n={args.n_samples} seed={args.seed}")
    corpus = extractors[args.dataset](n=args.n_samples, seed=args.seed)
    print(f"[build_index] Corpus size: {len(corpus)} unique paragraphs")

    # Format for LightRAG: "Title\nText" so the title becomes part of the chunk
    docs = [f"{p['title']}\n{p['text']}" for p in corpus]

    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)

    print(f"[build_index] Target workspace: {args.workspace!r}")
    print(f"[build_index] Working dir:      {args.working_dir}")

    batch = args.batch_size
    for i in range(0, len(docs), batch):
        chunk = docs[i : i + batch]
        await service.lightrag_ainsert(
            workspace_id=args.workspace,
            input=chunk,
            working_dir=args.working_dir,
        )
        done = min(i + batch, len(docs))
        print(f"[build_index]   {done}/{len(docs)} paragraphs indexed")

    print("\n[build_index] Indexing complete.")
    print("[build_index] Run evaluation with:")
    print(f"  python evaluate_local/MultiHopQA/evaluate_multihop.py \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --workspace {args.workspace} \\")
    print(f"    --working-dir {args.working_dir} \\")
    print(f"    --n-samples {args.n_samples} \\")
    print(f"    --seed {args.seed} \\")
    print(f"    --output-dir ./multihop_results \\")
    print(f"    --modes ppr hybrid mix")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Index a multi-hop QA corpus subset into a LightRAG workspace."
    )
    p.add_argument("--dataset",     required=True, choices=VALID_DATASETS,
                   help="Which dataset's context paragraphs to index")
    p.add_argument("--workspace",   required=True,
                   help="Workspace ID to create or append to")
    p.add_argument("--working-dir", required=True, dest="working_dir",
                   help="Directory where LightRAG stores its index files")
    p.add_argument("--n-samples",   type=int, default=500, dest="n_samples",
                   help="Number of questions to sample (must match evaluate_multihop.py)")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed (must match evaluate_multihop.py)")
    p.add_argument("--batch-size",  type=int, default=50, dest="batch_size",
                   help="Paragraphs per ainsert call")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
