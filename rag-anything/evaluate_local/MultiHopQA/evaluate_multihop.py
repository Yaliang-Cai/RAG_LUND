#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Hop QA Query-Mode Comparison Evaluator
=============================================

Runs one or more LightRAG query modes against a pre-built workspace index and
computes EM / F1 / Recall@K for HotpotQA, MuSiQue, 2WikiMultiHopQA, SimpleQA.

Usage:
    python evaluate_local/MultiHopQA/evaluate_multihop.py \
        --dataset hotpotqa \
        --workspace my_hotpotqa_workspace \
        --working-dir /data/y50056788/.../rag_workspaces/my_hotpotqa_workspace \
        --output-dir /data/y50056788/.../multihop_results \
        --modes ppr hybrid mix
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[3]
_lightrag_root = _project_root.parent / "lightrag"
for p in (_project_root, _lightrag_root):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv
load_dotenv()

VALID_MODES = ("ppr", "ppr_local", "global", "local", "hybrid", "mix", "naive", "rrf", "bypass")
VALID_DATASETS = ("hotpotqa", "musique", "2wiki", "simpleqa")

_REFERENCES_RE = re.compile(r"#+\s*references?.*", re.IGNORECASE | re.DOTALL)


def _strip_references(text: str) -> str:
    return _REFERENCES_RE.sub("", text).strip()


def _load_existing_ids(jsonl_path: Path) -> set[str]:
    ids: set[str] = set()
    if not jsonl_path.exists():
        return ids
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _aggregate_jsonl(jsonl_path: Path, recall_ks: list[int]) -> dict[str, Any]:
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return {}

    em_vals = [r["em"] for r in records]
    f1_vals = [r["f1"] for r in records]
    metrics: dict[str, float] = {
        "em":  round(sum(em_vals) / len(em_vals), 4),
        "f1":  round(sum(f1_vals) / len(f1_vals), 4),
        "n":   len(records),
    }
    for k in recall_ks:
        key = f"recall@{k}"
        vals = [r[key] for r in records if r.get(key) is not None]
        if vals:
            metrics[key] = round(sum(vals) / len(vals), 4)
    return metrics


async def _run_mode(
    service: Any,
    workspace_id: str,
    working_dir: str,
    items: list[dict],
    mode: str,
    dataset: str,
    recall_ks: list[int],
    output_dir: Path,
    resume: bool,
    score_em: Any,
    score_f1: Any,
    score_recall_at_k: Any,
    get_eval_query_overrides: Any,
) -> dict[str, Any]:
    jsonl_path = output_dir / f"{dataset}_{mode}_results.jsonl"
    existing_ids = _load_existing_ids(jsonl_path) if resume else set()
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    query_overrides = get_eval_query_overrides(dataset)
    done = len(existing_ids)
    total = len(items)

    for item in items:
        if item["id"] in existing_ids:
            continue

        try:
            result = await service.query_with_trace(
                workspace_id=workspace_id,
                query=item["question"],
                working_dir=working_dir,
                mode=mode,
                **query_overrides,
            )
            raw_answer = result.get("answer", "")
            answer = _strip_references(raw_answer)
            chunks = result.get("trace", {}).get("data", {}).get("chunks", [])
        except Exception as e:
            print(f"  [WARN] query failed for id={item['id']}: {e}")
            answer = ""
            chunks = []

        gold = item["answer"]
        em = score_em(answer, gold)
        f1 = score_f1(answer, gold)

        record: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "gold": gold,
            "pred": answer,
            "em": em,
            "f1": f1,
        }
        for k in recall_ks:
            r = score_recall_at_k(chunks, item.get("supporting_facts"), k)
            record[f"recall@{k}"] = r

        _append_jsonl(jsonl_path, record)
        done += 1
        if done % 50 == 0 or done == total:
            print(f"  [{mode}] {done}/{total}")

    return _aggregate_jsonl(jsonl_path, recall_ks)


async def main(args: argparse.Namespace) -> None:
    # Defer heavy imports so --help works without lightrag/raganything installed
    from raganything.services.local_rag import LocalRagService, LocalRagSettings
    from evaluate_local.MultiHopQA.dataset_adapters import (
        load_hotpotqa, load_musique, load_2wiki, load_simpleqa,
        score_em, score_f1, score_recall_at_k,
        get_eval_query_overrides,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] Loading dataset: {args.dataset} (n={args.n_samples}, seed={args.seed})")
    loaders = {
        "hotpotqa": load_hotpotqa,
        "musique":  load_musique,
        "2wiki":    load_2wiki,
        "simpleqa": load_simpleqa,
    }
    items = loaders[args.dataset](n=args.n_samples, seed=args.seed)
    print(f"[eval] Loaded {len(items)} questions")

    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)

    results: dict[str, dict] = {}
    for mode in args.modes:
        print(f"\n[eval] Running mode: {mode}")
        metrics = await _run_mode(
            service=service,
            workspace_id=args.workspace,
            working_dir=args.working_dir,
            items=items,
            mode=mode,
            dataset=args.dataset,
            recall_ks=args.recall_k,
            output_dir=output_dir,
            resume=args.resume,
            score_em=score_em,
            score_f1=score_f1,
            score_recall_at_k=score_recall_at_k,
            get_eval_query_overrides=get_eval_query_overrides,
        )
        results[mode] = metrics
        print(f"  [{mode}] EM={metrics.get('em', 0):.4f}  F1={metrics.get('f1', 0):.4f}")

    summary_path = output_dir / f"{args.dataset}_summary.json"
    summary = {
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "recall_k": args.recall_k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}  n={args.n_samples}")
    print(f"{'Mode':<15} {'EM':>8} {'F1':>8}", end="")
    for k in args.recall_k:
        print(f" {'R@'+str(k):>8}", end="")
    print()
    print("-" * (15 + 8 + 8 + 9 * len(args.recall_k) + 4))
    for mode, m in results.items():
        print(f"{mode:<15} {m.get('em', 0):.4f}   {m.get('f1', 0):.4f}", end="")
        for k in args.recall_k:
            val = m.get(f"recall@{k}")
            print(f"   {val:.4f}" if val is not None else "      N/A", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-hop QA query-mode evaluator")
    p.add_argument("--dataset",     required=True, choices=VALID_DATASETS)
    p.add_argument("--workspace",   required=True, help="Pre-built workspace ID")
    p.add_argument("--working-dir", required=True, dest="working_dir")
    p.add_argument("--modes",       nargs="+", default=["ppr", "hybrid", "mix"],
                   choices=VALID_MODES, metavar="MODE")
    p.add_argument("--n-samples",   type=int, default=500, dest="n_samples")
    p.add_argument("--recall-k",    type=int, nargs="+", default=[5, 10, 20], dest="recall_k")
    p.add_argument("--output-dir",  required=True, dest="output_dir")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
