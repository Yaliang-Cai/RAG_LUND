#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corpus size estimator for MultiHopQA datasets.

Counts unique paragraphs extracted for n sampled queries and estimates
chunk counts at a given tokens-per-chunk budget.

Usage:
    python evaluate_local/MultiHopQA/estimate_corpus_size.py
    python evaluate_local/MultiHopQA/estimate_corpus_size.py --n-samples 1000
    python evaluate_local/MultiHopQA/estimate_corpus_size.py --tokens-per-chunk 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_projects_root = Path(__file__).resolve().parents[3]
_raganything_root = Path(__file__).resolve().parents[2]
_lightrag_root = _projects_root / "lightrag"
for p in (_raganything_root, _lightrag_root, _projects_root):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv
load_dotenv()

# Rough tokens-per-word ratio for English Wikipedia text (GPT-style BPE).
_TOKENS_PER_WORD = 1.35


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _TOKENS_PER_WORD)


def _analyze_corpus(dataset: str, corpus: list[dict], tokens_per_chunk: int) -> dict:
    token_counts = [_estimate_tokens(p.get("text", "") + " " + p.get("title", "")) for p in corpus]
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    # Chunks: paragraphs that exceed tokens_per_chunk get split; smaller ones stay as-is.
    import math
    chunk_count = sum(max(1, math.ceil(t / tokens_per_chunk)) for t in token_counts)

    return {
        "dataset": dataset,
        "unique_paragraphs": len(corpus),
        "avg_tokens_per_paragraph": round(avg_tokens, 1),
        "max_tokens_per_paragraph": max_tokens,
        "total_tokens": total_tokens,
        "tokens_per_chunk": tokens_per_chunk,
        "estimated_chunks": chunk_count,
        "paragraphs_split": sum(1 for t in token_counts if t > tokens_per_chunk),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=500, dest="n_samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokens-per-chunk", type=int, default=1200, dest="tokens_per_chunk")
    p.add_argument("--datasets", nargs="+",
                   default=["hotpotqa", "musique", "2wiki"],
                   choices=["hotpotqa", "musique", "2wiki"])
    args = p.parse_args()

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

    # HippoRAG2 reference numbers (1000 queries)
    hipporag2_docs = {"hotpotqa": 9811, "musique": 11656, "2wiki": 6119}

    print(f"\nCorpus size analysis — n_samples={args.n_samples}, seed={args.seed}, "
          f"tokens_per_chunk={args.tokens_per_chunk}\n")
    print(f"{'Dataset':<12} {'Paragraphs':>12} {'AvgTok':>8} {'MaxTok':>8} "
          f"{'Est.Chunks':>12} {'Split(%)':>9} {'HR2@1k':>8} {'HR2@500est':>12}")
    print("-" * 90)

    for dataset in args.datasets:
        print(f"  Loading {dataset}...", end="", flush=True)
        corpus = extractors[dataset](n=args.n_samples, seed=args.seed)
        stats = _analyze_corpus(dataset, corpus, args.tokens_per_chunk)
        hr2 = hipporag2_docs.get(dataset, "?")
        # Rough linear estimate of what HippoRAG2 would have at our n
        hr2_scaled = round(hr2 * args.n_samples / 1000) if isinstance(hr2, int) else "?"
        split_pct = (
            f"{stats['paragraphs_split'] / stats['unique_paragraphs'] * 100:.1f}%"
            if stats['unique_paragraphs'] else "N/A"
        )
        print(f"\r  {dataset:<12} {stats['unique_paragraphs']:>12} "
              f"{stats['avg_tokens_per_paragraph']:>8} {stats['max_tokens_per_paragraph']:>8} "
              f"{stats['estimated_chunks']:>12} {split_pct:>9} "
              f"{hr2:>8} {hr2_scaled:>12}")

    print()
    print("HR2@1k = HippoRAG2 document count for 1000 queries")
    print("HR2@500est = linear-scaled estimate for comparison")
    print(f"Note: 'Est.Chunks' assumes each paragraph >{args.tokens_per_chunk} tokens is split.")
    print("      In this pipeline (split_by_character_only=True), each paragraph is always")
    print("      exactly 1 chunk regardless of length, so paragraph count = chunk count.")


if __name__ == "__main__":
    main()
