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
PPR_MODES = {"ppr", "ppr_local"}

_REFERENCES_RE = re.compile(r"#+\s*references?.*", re.IGNORECASE | re.DOTALL)


class _TeeStream:
    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        primary = self._streams[0] if self._streams else None
        return bool(getattr(primary, "isatty", lambda: False)())


class _TeeOutput:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self._file = None
        self._stdout = None
        self._stderr = None

    def __enter__(self) -> "_TeeOutput":
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_file.open("w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _TeeStream(self._stdout, self._file)
        sys.stderr = _TeeStream(self._stderr, self._file)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._file is not None:
            self._file.close()


def _resolve_log_file(output_dir: str | Path, log_file: str | None, dataset: str) -> Path:
    output_dir_path = Path(output_dir)
    if log_file:
        candidate = Path(log_file).expanduser()
        return candidate if candidate.is_absolute() else output_dir_path / candidate
    return output_dir_path / f"{dataset}_evaluate_multihop.log"


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


def _load_chunk_source_map(
    working_dir: str | Path,
    *,
    dataset: str | None = None,
    workspace: str | None = None,
    n_samples: int | None = None,
    seed: int | None = None,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    source_map_path = Path(working_dir) / SOURCE_MAP_FILENAME
    if not source_map_path.exists():
        if strict:
            raise FileNotFoundError(f"Missing chunk source map: {source_map_path}")
        return {}
    try:
        payload = json.loads(source_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if strict:
            raise ValueError(f"Invalid chunk source map JSON: {source_map_path}") from exc
        return {}

    if strict:
        expected = {
            "workspace_id": workspace,
            "dataset": dataset,
            "n_samples": n_samples,
            "seed": seed,
        }
        mismatches = []
        for key, expected_value in expected.items():
            if expected_value is None:
                continue
            actual_value = payload.get(key)
            if actual_value != expected_value:
                mismatches.append(f"{key}: expected={expected_value!r}, actual={actual_value!r}")
        if mismatches:
            raise ValueError(
                "Chunk source map identity mismatch. "
                f"Use the workspace built for this dataset/sample/seed. Details: {'; '.join(mismatches)}"
            )

    mapping = payload.get("map", {})
    if not isinstance(mapping, dict):
        if strict:
            raise ValueError(f"Chunk source map has no object 'map': {source_map_path}")
        return {}
    if strict and "map_size" in payload and int(payload["map_size"]) != len(mapping):
        raise ValueError(
            "Chunk source map size mismatch. "
            f"map_size={payload['map_size']!r}, actual={len(mapping)}"
        )
    return {str(k): v for k, v in mapping.items() if isinstance(v, dict)}


def _build_query_kwargs(
    *,
    query_overrides: dict[str, Any],
    wire_profile: str | None,
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    max_total_tokens: int | None = None,
    **extra_query_kwargs: Any,
) -> dict[str, Any]:
    kwargs = dict(query_overrides)
    if wire_profile is not None:
        kwargs["profile"] = wire_profile
    if top_k is not None:
        kwargs["top_k"] = int(top_k)
    if chunk_top_k is not None:
        kwargs["chunk_top_k"] = int(chunk_top_k)
    if max_total_tokens is not None:
        kwargs["max_total_tokens"] = int(max_total_tokens)
    for key, value in extra_query_kwargs.items():
        if value is not None:
            kwargs[key] = value
    return kwargs


def _mode_query_kwargs(
    query_kwargs: dict[str, Any] | None,
    mode: str,
    *,
    hybrid_enable_rerank: bool = True,
    ppr_enable_rerank: bool = False,
) -> dict[str, Any]:
    kwargs = {k: v for k, v in dict(query_kwargs or {}).items() if v is not None}
    if mode in PPR_MODES:
        kwargs["enable_rerank"] = bool(ppr_enable_rerank)
        kwargs["answer_context_mode"] = "chunk_only_prompt"
        kwargs["ppr_post_rerank_fusion"] = str(
            kwargs.get("ppr_post_rerank_fusion", "none")
        ).strip().lower()
        kwargs["ppr_post_rerank_rrf_k"] = int(
            kwargs.get("ppr_post_rerank_rrf_k", 60)
        )
    else:
        kwargs["enable_rerank"] = bool(hybrid_enable_rerank)
        if mode == "hybrid":
            kwargs.setdefault("answer_context_mode", "kg_prompt")
    return kwargs


def _trace_chunk_id(chunk: dict[str, Any]) -> str:
    for key in ("chunk_id", "_id", "__id__", "key"):
        value = str(chunk.get(key) or "").strip()
        if value:
            return value
    value = str(chunk.get("id") or "").strip()
    if value and not re.fullmatch(r"DC\d+", value, flags=re.IGNORECASE):
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
    query_kwargs: dict[str, Any] | None = None,
    concurrency: int = 1,
    hybrid_enable_rerank: bool = True,
    ppr_enable_rerank: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{dataset}_{mode}_results.jsonl"
    existing_ids = _load_existing_ids(jsonl_path) if resume else set()
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    query_overrides = get_eval_query_overrides(dataset)
    done = len(existing_ids)
    total = len(items)
    last_reported = done
    effective_concurrency = max(1, int(concurrency))
    query_kwargs = _mode_query_kwargs(
        query_kwargs,
        mode,
        hybrid_enable_rerank=hybrid_enable_rerank,
        ppr_enable_rerank=ppr_enable_rerank,
    )

    # "full" is a pseudo-mode: forces the router's "full" profile (all paths, RRF fusion).
    # "auto" lets the router classify per query and pick the best profile.
    # Both use mode="auto" on the wire; only "full" pins a profile.
    wire_mode = "auto" if mode in ("auto", "full") else mode
    wire_profile = "full" if mode == "full" else None

    async def _evaluate_item(item: dict) -> dict[str, Any]:
        try:
            call_kwargs = _build_query_kwargs(
                query_overrides=query_overrides,
                wire_profile=wire_profile,
                **query_kwargs,
            )
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

        return record

    pending_items = [item for item in items if item["id"] not in existing_ids]
    for start in range(0, len(pending_items), effective_concurrency):
        batch = pending_items[start : start + effective_concurrency]
        records = await asyncio.gather(*(_evaluate_item(item) for item in batch))

        for record in records:
            _append_jsonl(jsonl_path, record)
        done += len(records)
        if done == total or done - last_reported >= 50:
            print(f"  [{mode}] {done}/{total}")
            last_reported = done

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
    strict_source_map = args.dataset != "simpleqa" and not args.allow_missing_source_map
    chunk_source_map = _load_chunk_source_map(
        args.working_dir,
        dataset=args.dataset,
        workspace=args.workspace,
        n_samples=args.n_samples,
        seed=args.seed,
        strict=strict_source_map,
    )
    if args.dataset == "simpleqa":
        print("[eval] SimpleQA has no supporting facts/source corpus; Recall@K will be N/A")
    if chunk_source_map:
        print(f"[eval] Loaded chunk source map: {len(chunk_source_map)} chunks")
    else:
        print("[eval] No chunk source map found; JSONL will not include retrieved source ids")

    base_query_kwargs = {
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "max_total_tokens": args.max_total_tokens,
        "qdrant_retrieval_mode": args.qdrant_retrieval_mode,
        "entity_qdrant_retrieval_mode": args.qdrant_retrieval_mode,
        "chunk_qdrant_retrieval_mode": args.qdrant_retrieval_mode,
        "keyword_fanout_mode": args.keyword_fanout_mode,
        "keyword_entity_rrf_k": args.keyword_entity_rrf_k,
        "keyword_relation_rrf_k": args.keyword_relation_rrf_k,
        "answer_context_mode": args.answer_context_mode,
        "kg_chunk_selection_source": args.kg_chunk_selection_source,
        "enable_kg_rerank": args.enable_kg_rerank,
        "rerank_score_scope": "all",
        "ppr_damping": args.ppr_damping,
        "ppr_top_k": args.ppr_top_k,
        "ppr_qa_top_k": args.ppr_qa_top_k,
        "ppr_post_rerank_fusion": args.ppr_post_rerank_fusion,
        "ppr_post_rerank_rrf_k": args.ppr_post_rerank_rrf_k,
        "passage_node_weight": args.passage_node_weight,
        "recognition_top_k": args.recognition_top_k,
        "linking_top_k": args.linking_top_k,
        "ppr_synonym_weight_mode": args.ppr_synonym_weight_mode,
        "exclude_synonym_edges": args.exclude_synonym_edges,
        "bypass_query_cache": args.bypass_query_cache,
        "bypass_keywords_cache": args.bypass_keywords_cache,
        "vlm_enhanced": args.vlm_enhanced,
    }
    print(
        "[eval] Query controls: "
        f"top_k={args.top_k}, chunk_top_k={args.chunk_top_k}, "
        f"max_total_tokens={args.max_total_tokens}, concurrency={args.concurrency}, "
        f"qdrant_retrieval_mode={args.qdrant_retrieval_mode}, "
        f"ppr_top_k={args.ppr_top_k}, ppr_qa_top_k={args.ppr_qa_top_k}, "
        f"enable_kg_rerank={args.enable_kg_rerank}, "
        f"bypass_query_cache={args.bypass_query_cache}, vlm_enhanced={args.vlm_enhanced}"
    )

    results: dict[str, dict] = {}
    query_kwargs_by_mode: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        print(f"\n[eval] Running mode: {mode}")
        mode_query_kwargs = _mode_query_kwargs(
            base_query_kwargs,
            mode,
            hybrid_enable_rerank=args.hybrid_enable_rerank,
            ppr_enable_rerank=args.ppr_enable_rerank,
        )
        query_kwargs_by_mode[mode] = mode_query_kwargs
        print(
            f"  [{mode}] enable_rerank={mode_query_kwargs.get('enable_rerank')}, "
            f"answer_context_mode={mode_query_kwargs.get('answer_context_mode')}"
        )
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
            query_kwargs=base_query_kwargs,
            concurrency=args.concurrency,
            hybrid_enable_rerank=args.hybrid_enable_rerank,
            ppr_enable_rerank=args.ppr_enable_rerank,
        )
        results[mode] = metrics
        print(f"  [{mode}] EM={metrics.get('em', 0):.4f}  F1={metrics.get('f1', 0):.4f}")

    summary_path = output_dir / f"{args.dataset}_summary.json"
    summary = {
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "recall_k": args.recall_k,
        "concurrency": args.concurrency,
        "base_query_kwargs": {k: v for k, v in base_query_kwargs.items() if v is not None},
        "query_kwargs_by_mode": query_kwargs_by_mode,
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


async def main_with_logging(args: argparse.Namespace) -> None:
    log_path = _resolve_log_file(args.output_dir, args.log_file, args.dataset)
    with _TeeOutput(log_path):
        print(f"[eval] Log file: {log_path}")
        print(f"[eval] Started at: {datetime.now(timezone.utc).isoformat()}")
        print(f"[eval] CLI args: {json.dumps(vars(args), ensure_ascii=False, sort_keys=True)}")
        await main(args)


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
    p.add_argument("--log-file", default=None, dest="log_file")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--top-k", type=int, default=10, dest="top_k")
    p.add_argument("--chunk-top-k", type=int, default=5, dest="chunk_top_k")
    p.add_argument("--max-total-tokens", type=int, default=45000, dest="max_total_tokens")
    p.add_argument("--qdrant-retrieval-mode", default="dense", choices=["dense", "bm25", "hybrid"], dest="qdrant_retrieval_mode")
    p.add_argument("--keyword-fanout-mode", default="joined", choices=["joined", "per_keyword_rrf"], dest="keyword_fanout_mode")
    p.add_argument("--keyword-entity-rrf-k", type=int, default=10, dest="keyword_entity_rrf_k")
    p.add_argument("--keyword-relation-rrf-k", type=int, default=20, dest="keyword_relation_rrf_k")
    p.add_argument("--answer-context-mode", default="kg_prompt", choices=["kg_prompt", "chunk_only_prompt"], dest="answer_context_mode")
    p.add_argument("--kg-chunk-selection-source", default="truncated", choices=["truncated", "untruncated"], dest="kg_chunk_selection_source")
    p.add_argument("--enable-kg-rerank", action=argparse.BooleanOptionalAction, default=False, dest="enable_kg_rerank")
    p.add_argument("--hybrid-enable-rerank", action=argparse.BooleanOptionalAction, default=True, dest="hybrid_enable_rerank")
    p.add_argument("--ppr-enable-rerank", action=argparse.BooleanOptionalAction, default=False, dest="ppr_enable_rerank")
    p.add_argument("--ppr-damping", type=float, default=0.5, dest="ppr_damping")
    p.add_argument("--ppr-top-k", type=int, default=50, dest="ppr_top_k")
    p.add_argument("--ppr-qa-top-k", type=int, default=5, dest="ppr_qa_top_k")
    p.add_argument("--ppr-post-rerank-fusion", "--ppr_post_rerank_fusion", default="none", choices=["none", "raw_rrf"], dest="ppr_post_rerank_fusion")
    p.add_argument("--ppr-post-rerank-rrf-k", "--ppr_post_rerank_rrf_k", type=int, default=60, dest="ppr_post_rerank_rrf_k")
    p.add_argument("--passage-node-weight", type=float, default=0.05, dest="passage_node_weight")
    p.add_argument("--recognition-top-k", type=int, default=20, dest="recognition_top_k")
    p.add_argument("--linking-top-k", type=int, default=5, dest="linking_top_k")
    p.add_argument("--ppr-synonym-weight-mode", default="raw", choices=["raw", "plus_one"], dest="ppr_synonym_weight_mode")
    p.add_argument("--exclude-synonym-edges", action=argparse.BooleanOptionalAction, default=None, dest="exclude_synonym_edges")
    p.add_argument("--bypass-query-cache", "--bypass_query_cache", action=argparse.BooleanOptionalAction, default=True, dest="bypass_query_cache")
    p.add_argument("--bypass-keywords-cache", "--bypass_keywords_cache", action=argparse.BooleanOptionalAction, default=False, dest="bypass_keywords_cache")
    p.add_argument("--vlm-enhanced", action=argparse.BooleanOptionalAction, default=False, dest="vlm_enhanced")
    p.add_argument("--allow-missing-source-map", action="store_true", dest="allow_missing_source_map")
    args = p.parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.chunk_top_k <= 0:
        raise SystemExit("--chunk-top-k must be > 0")
    if args.max_total_tokens <= 0:
        raise SystemExit("--max-total-tokens must be > 0")
    if args.ppr_top_k <= 0:
        raise SystemExit("--ppr-top-k must be > 0")
    if args.ppr_qa_top_k <= 0:
        raise SystemExit("--ppr-qa-top-k must be > 0")
    if args.ppr_qa_top_k > args.ppr_top_k:
        raise SystemExit("--ppr-qa-top-k must be <= --ppr-top-k")
    if args.ppr_post_rerank_rrf_k <= 0:
        raise SystemExit("--ppr-post-rerank-rrf-k must be > 0")
    if not (0.0 < args.ppr_damping < 1.0):
        raise SystemExit("--ppr-damping must be in (0,1)")
    if args.keyword_entity_rrf_k <= 0:
        raise SystemExit("--keyword-entity-rrf-k must be > 0")
    if args.keyword_relation_rrf_k <= 0:
        raise SystemExit("--keyword-relation-rrf-k must be > 0")
    if args.passage_node_weight < 0:
        raise SystemExit("--passage-node-weight must be >= 0")
    if args.recognition_top_k < 0:
        raise SystemExit("--recognition-top-k must be >= 0")
    if args.linking_top_k < 0:
        raise SystemExit("--linking-top-k must be >= 0")
    return args


if __name__ == "__main__":
    asyncio.run(main_with_logging(_parse_args()))
