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
        --modes naive hybrid ppr auto
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_projects_root = Path(__file__).resolve().parents[3]
_raganything_root = Path(__file__).resolve().parents[2]
_lightrag_root = _projects_root / "lightrag"
for p in (_raganything_root, _lightrag_root, _projects_root):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv
load_dotenv()

VALID_MODES = ("ppr", "ppr_local", "global", "local", "hybrid", "mix", "naive", "rrf", "bypass", "auto", "full")
VALID_DATASETS = ("hotpotqa", "musique", "2wiki", "simpleqa")
SOURCE_MAP_FILENAME = "multihopqa_chunk_source_map.json"

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


def _load_chunk_source_map(working_dir: str | Path) -> dict[str, dict[str, Any]]:
    source_map_path = Path(working_dir) / SOURCE_MAP_FILENAME
    if not source_map_path.exists():
        return {}
    try:
        payload = json.loads(source_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    mapping = payload.get("map", {})
    if not isinstance(mapping, dict):
        return {}
    return {str(k): v for k, v in mapping.items() if isinstance(v, dict)}


def _trace_chunk_id(chunk: dict[str, Any]) -> str:
    for key in ("id", "chunk_id", "_id", "__id__", "key"):
        value = str(chunk.get(key) or "").strip()
        if value:
            return value
    content = str(chunk.get("content") or "").strip()
    if not content:
        return ""
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

    return compute_mdhash_id(sanitize_text_for_encoding(content).strip(), prefix="chunk-")


def _resolve_retrieved_sources(
    chunks: list[dict[str, Any]],
    chunk_source_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not chunk_source_map:
        return []
    sources: list[dict[str, Any]] = []
    for rank, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        chunk_id = _trace_chunk_id(chunk)
        if not chunk_id:
            continue
        source = chunk_source_map.get(chunk_id)
        if not source:
            continue
        sources.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "source_paragraph_id": source.get("source_paragraph_id"),
                "source_key": source.get("source_key"),
                "title": source.get("title"),
            }
        )
    return sources


def _score_recall_by_source_keys(
    retrieved_sources: list[dict[str, Any]],
    gold_source_keys: list[str] | None,
    k: int,
) -> float | None:
    if not gold_source_keys:
        return None
    gold = {str(key) for key in gold_source_keys if key}
    if not gold:
        return None
    retrieved = {
        str(source.get("source_key"))
        for source in retrieved_sources[:k]
        if source.get("source_key")
    }
    return len(gold & retrieved) / len(gold)


def _score_support_recall(
    *,
    chunks: list[dict[str, Any]],
    item: dict[str, Any],
    k: int,
    chunk_source_map: dict[str, dict[str, Any]],
    fallback_score_recall_at_k: Callable[[list[dict], list[str] | None, int], float | None],
) -> float | None:
    if chunk_source_map and item.get("gold_source_keys"):
        retrieved_sources = _resolve_retrieved_sources(chunks, chunk_source_map)
        return _score_recall_by_source_keys(
            retrieved_sources,
            item.get("gold_source_keys"),
            k,
        )
    return fallback_score_recall_at_k(chunks, item.get("supporting_facts"), k)


def _aggregate_jsonl(jsonl_path: Path, recall_ks: list[int]) -> dict[str, Any]:
    if not jsonl_path.exists():
        return {}
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
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
    score_em: Callable[[str, str | list[str]], float],
    score_f1: Callable[[str, str | list[str]], float],
    score_recall_at_k: Callable[[list[dict], list[str] | None, int], float | None],
    get_eval_query_overrides: Callable[[str], dict[str, str]],
    chunk_source_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    jsonl_path = output_dir / f"{dataset}_{mode}_results.jsonl"
    existing_ids = _load_existing_ids(jsonl_path) if resume else set()
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    query_overrides = get_eval_query_overrides(dataset)
    done = len(existing_ids)
    total = len(items)

    # "full" is a pseudo-mode: forces the router's "full" profile (all paths, RRF fusion).
    # "auto" lets the router classify per query and pick the best profile.
    # Both use mode="auto" on the wire; only "full" pins a profile.
    wire_mode = "auto" if mode in ("auto", "full") else mode
    wire_profile = "full" if mode == "full" else None

    for item in items:
        if item["id"] in existing_ids:
            continue

        try:
            call_kwargs = dict(query_overrides)
            if wire_profile is not None:
                call_kwargs["profile"] = wire_profile
            result = await service.query_with_trace(
                workspace_id=workspace_id,
                query=item["question"],
                working_dir=working_dir,
                mode=wire_mode,
                **call_kwargs,
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
        if item.get("gold_source_keys"):
            record["gold_source_keys"] = item["gold_source_keys"]
        retrieved_sources = _resolve_retrieved_sources(chunks, chunk_source_map)
        for k in recall_ks:
            r = _score_support_recall(
                chunks=chunks,
                item=item,
                k=k,
                chunk_source_map=chunk_source_map,
                fallback_score_recall_at_k=score_recall_at_k,
            )
            record[f"recall@{k}"] = r

        if retrieved_sources:
            record["retrieved_source_paragraph_ids"] = [
                str(s["source_paragraph_id"])
                for s in retrieved_sources
                if s.get("source_paragraph_id")
            ]
            record["retrieved_source_keys"] = [
                str(s["source_key"])
                for s in retrieved_sources
                if s.get("source_key")
            ]
            record["retrieved_sources"] = retrieved_sources

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
    chunk_source_map = _load_chunk_source_map(args.working_dir)
    if chunk_source_map:
        print(f"[eval] Loaded chunk source map: {len(chunk_source_map)} chunks")
    else:
        print("[eval] No chunk source map found; JSONL will not include retrieved source ids")

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
            chunk_source_map=chunk_source_map,
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
    p.add_argument("--modes",       nargs="+", default=["naive", "hybrid", "ppr", "auto", "full"],
                   choices=VALID_MODES, metavar="MODE")
    p.add_argument("--n-samples",   type=int, default=500, dest="n_samples")
    p.add_argument("--recall-k",    type=int, nargs="+", default=[5, 10, 20], dest="recall_k")
    p.add_argument("--output-dir",  required=True, dest="output_dir")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
