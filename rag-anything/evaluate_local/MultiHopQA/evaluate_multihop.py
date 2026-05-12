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
import random
import re
import sys
from collections import Counter, defaultdict
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

VALID_MODES = ("ppr", "ppr_local", "global", "local", "hybrid", "mix", "naive", "rrf", "bypass", "auto", "full", "agentic")
VALID_DATASETS = ("hotpotqa", "musique", "2wiki", "simpleqa")
VALID_QA_PROMPT_STYLES = ("lightrag", "semantic_cot", "kg_semantic_cot")
VALID_ANSWER_PARSE_MODES = ("strip_references", "answer_marker")
SOURCE_MAP_FILENAME = "multihopqa_chunk_source_map.json"
PPR_MODES = {"ppr", "ppr_local"}

_REFERENCES_RE = re.compile(r"#+\s*references?.*", re.IGNORECASE | re.DOTALL)
_ANSWER_MARKER_RE = re.compile(r"\banswer\s*:\s*", re.IGNORECASE)

_SEMANTIC_COT_QA_SYSTEM = (
    'As an advanced reading comprehension assistant, your task is to analyze the provided passages and answer the corresponding question meticulously. '
    'Use only the information in the provided passages. '
    'Your response should start after "Thought: ", where you briefly identify the evidence needed to answer the question. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
)

_KG_SEMANTIC_COT_QA_SYSTEM = (
    "As an advanced multi-hop reading comprehension assistant, analyze the "
    "provided knowledge graph data and document passages to answer the question. "
    "Use only the provided context. Start your response after \"Thought: \" by "
    "briefly identifying the evidence chain, and conclude with \"Answer: \" "
    "followed by a concise final answer."
)

_SEMANTIC_COT_ONE_SHOT_QA_DOCS = (
    "Passage 1:\n"
    "Title: The Last Horse\n"
    "Text: The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n\n"
    "Passage 2:\n"
    "Title: Southampton\n"
    "Text: The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. "
    "The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. "
    "In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. "
    "The university considers itself one of the top 5 research universities in the UK. "
    "The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, "
    "computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) "
    "It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n\n"
    "Passage 3:\n"
    "Title: Stanton Township, Champaign County, Illinois\n"
    "Text: Stanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n\n"
    "Passage 4:\n"
    "Title: Neville A. Stanton\n"
    "Text: Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. "
    "Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). "
    "He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. "
    "Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. "
    'He has been published in academic journals including "Nature". '
    "He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n\n"
    "Passage 5:\n"
    "Title: Finding Nemo\n"
    "Text: Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds "
    "Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan "
    "Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution "
    "Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"
)

_SEMANTIC_COT_ONE_SHOT_QA_INPUT = (
    f"{_SEMANTIC_COT_ONE_SHOT_QA_DOCS}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    "\nThought: "
)

_SEMANTIC_COT_ONE_SHOT_QA_OUTPUT = (
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)


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


def _parse_answer_text(text: str, answer_parse_mode: str) -> str:
    stripped = _strip_references(text)
    if answer_parse_mode == "strip_references":
        return stripped
    if answer_parse_mode == "answer_marker":
        matches = list(_ANSWER_MARKER_RE.finditer(stripped))
        if not matches:
            return stripped
        return stripped[matches[-1].end():].strip()
    raise ValueError(f"Unknown answer_parse_mode: {answer_parse_mode!r}")


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


def _select_hipporag2_eval_items(items: list[dict], n_samples: int, seed: int) -> list[dict]:
    """Select a deterministic query subset without changing the indexed corpus identity."""
    if n_samples <= 0 or n_samples >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n_samples)


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
    naive_top_k: int | None = None,
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
    if naive_top_k is not None:
        kwargs["naive_top_k"] = int(naive_top_k)
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


def _semantic_cot_passage_text(
    rank: int,
    chunk: dict[str, Any],
    chunk_source_map: dict[str, dict[str, Any]],
) -> str:
    chunk_id = _trace_chunk_id(chunk)
    source = chunk_source_map.get(chunk_id) if chunk_id else None
    if source:
        title = str(source.get("title") or "").strip()
        text = str(source.get("text") or "").strip()
        if title and text:
            return f"Passage {rank}:\nTitle: {title}\nText: {text}"
        content = str(source.get("content") or "").strip()
        if title:
            return f"Passage {rank}:\nTitle: {title}"
        if text:
            return f"Passage {rank}:\nText: {text}"
        if content:
            return f"Passage {rank}:\n{content}"
    content = str(chunk.get("content") or "").strip()
    return f"Passage {rank}:\n{content}" if content else ""


def _build_semantic_cot_user_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    chunk_source_map: dict[str, dict[str, Any]],
    *,
    qa_top_k: int | None = None,
) -> str:
    limit = len(chunks) if qa_top_k is None else max(0, int(qa_top_k))
    passages = []
    for rank, chunk in enumerate(chunks[:limit], start=1):
        if not isinstance(chunk, dict):
            continue
        passage = _semantic_cot_passage_text(rank, chunk, chunk_source_map)
        if passage:
            passages.append(passage)
    prefix = "".join(f"{passage}\n\n" for passage in passages)
    return f"{prefix}Question: {question}\nThought: "


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _jsonl_context_block(rows: list[Any]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)


def _kg_semantic_reference_list(references: list[Any]) -> str:
    lines = []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        reference_id = str(ref.get("reference_id") or "").strip()
        title = str(ref.get("file_path") or ref.get("title") or "").strip()
        if reference_id:
            lines.append(f"[{reference_id}] {title}".rstrip())
    return "\n".join(lines)


def _build_kg_semantic_cot_user_prompt(
    question: str,
    raw_data: dict[str, Any],
    chunk_source_map: dict[str, dict[str, Any]],
    *,
    qa_top_k: int | None = None,
) -> str:
    data = raw_data.get("data") if isinstance(raw_data.get("data"), dict) else raw_data
    entities = _as_list(data.get("entities"))
    relationships = _as_list(data.get("relationships"))
    chunks = [chunk for chunk in _as_list(data.get("chunks")) if isinstance(chunk, dict)]
    references = _as_list(data.get("references"))

    limit = len(chunks) if qa_top_k is None else max(0, int(qa_top_k))
    passages = []
    for rank, chunk in enumerate(chunks[:limit], start=1):
        passage = _semantic_cot_passage_text(rank, chunk, chunk_source_map)
        if passage:
            passages.append(passage)

    passage_block = "\n\n".join(passages)
    reference_list = _kg_semantic_reference_list(references)
    return (
        "Knowledge Graph Data (Entity):\n\n"
        "```json\n"
        f"{_jsonl_context_block(entities)}\n"
        "```\n\n"
        "Knowledge Graph Data (Relationship):\n\n"
        "```json\n"
        f"{_jsonl_context_block(relationships)}\n"
        "```\n\n"
        "Document Chunks (Title/Text passages):\n\n"
        f"{passage_block}\n\n"
        "Reference Document List:\n\n"
        "```\n"
        f"{reference_list}\n"
        "```\n\n"
        f"Question: {question}\nThought: "
    )


async def _call_semantic_cot_qa(
    service: Any,
    *,
    question: str,
    chunks: list[dict[str, Any]],
    chunk_source_map: dict[str, dict[str, Any]],
    qa_top_k: int | None,
) -> str:
    llm_model_func = getattr(service, "llm_model_func", None)
    if not callable(llm_model_func):
        raise RuntimeError("Semantic CoT QA prompt requires service.llm_model_func")
    prompt = _build_semantic_cot_user_prompt(
        question,
        chunks,
        chunk_source_map,
        qa_top_k=qa_top_k,
    )
    answer = await llm_model_func(
        prompt,
        system_prompt=_SEMANTIC_COT_QA_SYSTEM,
        history_messages=[
            {"role": "user", "content": _SEMANTIC_COT_ONE_SHOT_QA_INPUT},
            {"role": "assistant", "content": _SEMANTIC_COT_ONE_SHOT_QA_OUTPUT},
        ],
        enable_cot=True,
        stream=False,
    )
    return answer if isinstance(answer, str) else str(answer)


async def _call_kg_semantic_cot_qa(
    service: Any,
    *,
    question: str,
    raw_data: dict[str, Any],
    chunk_source_map: dict[str, dict[str, Any]],
    qa_top_k: int | None,
) -> str:
    llm_model_func = getattr(service, "llm_model_func", None)
    if not callable(llm_model_func):
        raise RuntimeError("KG semantic CoT QA prompt requires service.llm_model_func")
    prompt = _build_kg_semantic_cot_user_prompt(
        question,
        raw_data,
        chunk_source_map,
        qa_top_k=qa_top_k,
    )
    answer = await llm_model_func(
        prompt,
        system_prompt=_KG_SEMANTIC_COT_QA_SYSTEM,
        enable_cot=True,
        stream=False,
    )
    return answer if isinstance(answer, str) else str(answer)


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
) -> float | None:
    """Passage-level Recall@K via source_map.

    Returns None when source_map or gold_source_keys are absent — callers
    should record this as N/A rather than substitute a chunk-level proxy,
    which would not be comparable to HippoRAG2 / GFM-RAG passage recall.
    """
    if not chunk_source_map or not item.get("gold_source_keys"):
        return None
    retrieved_sources = _resolve_retrieved_sources(chunks, chunk_source_map)
    return _score_recall_by_source_keys(
        retrieved_sources,
        item.get("gold_source_keys"),
        k,
    )


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


def _aggregate_agentic_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    profile_counts: Counter[str] = Counter()
    retrieval_cycle_counts: Counter[str] = Counter()
    check_cycle_counts: Counter[str] = Counter()
    paths_activated_counts: Counter[str] = Counter()
    first_paths_counts: Counter[str] = Counter()
    chunks_per_path_sum: defaultdict[str, float] = defaultdict(float)
    chunks_per_path_count: Counter[str] = Counter()

    semantic_selected_count = 0
    semantic_fallback_count = 0
    rewrite_count = 0
    decompose_count = 0
    targeted_retrieval_count = 0
    grounded_count = 0
    empty_final_chunks_count = 0
    query_fail_count = 0

    for record in records:
        record_failed = bool(record.get("agentic_query_failed"))
        if record_failed:
            query_fail_count += 1
        trace = record.get("agentic_trace")
        if not isinstance(trace, dict) or not trace:
            if not record_failed:
                query_fail_count += 1
            continue

        classifier = trace.get("classifier") if isinstance(trace.get("classifier"), dict) else {}
        selected_profile = str(
            classifier.get("selected_profile") or trace.get("profile") or "unknown"
        )
        profile_counts[selected_profile] += 1
        if selected_profile == "semantic":
            semantic_selected_count += 1
            if classifier.get("fallback_used") is True:
                semantic_fallback_count += 1

        retrieval_cycle_counts[str(int(trace.get("retrieve_cycles_used") or 0))] += 1
        check_cycle_counts[str(int(trace.get("check_cycles_used") or 0))] += 1

        steps = trace.get("retrieval_steps") if isinstance(trace.get("retrieval_steps"), list) else []
        step_types = {str(step.get("type") or "") for step in steps if isinstance(step, dict)}
        rewrite_history = trace.get("rewrite_history") if isinstance(trace.get("rewrite_history"), list) else []
        sub_questions = trace.get("sub_questions") if isinstance(trace.get("sub_questions"), list) else []
        if len(rewrite_history) > 1 or "rewrite" in step_types:
            rewrite_count += 1
        if sub_questions or "decompose" in step_types:
            decompose_count += 1
        if "targeted" in step_types:
            targeted_retrieval_count += 1

        hallucination_events = (
            trace.get("hallucination_events")
            if isinstance(trace.get("hallucination_events"), list)
            else []
        )
        if trace.get("grounded") is True or any(
            isinstance(event, dict) and event.get("grounded") is True
            for event in hallucination_events
        ):
            grounded_count += 1

        final_chunks = trace.get("data", {}).get("chunks", []) if isinstance(trace.get("data"), dict) else []
        if not final_chunks:
            empty_final_chunks_count += 1

        first_step_seen = False
        for step in steps:
            if not isinstance(step, dict):
                continue
            paths = [
                str(path)
                for path in step.get("paths_activated", [])
                if str(path).strip()
            ]
            for path in paths:
                paths_activated_counts[path] += 1
                if not first_step_seen:
                    first_paths_counts[path] += 1
            first_step_seen = first_step_seen or bool(paths)
            chunks_per_path = step.get("chunks_per_path", {})
            if isinstance(chunks_per_path, dict):
                for path, count in chunks_per_path.items():
                    try:
                        chunks_per_path_sum[str(path)] += float(count)
                        chunks_per_path_count[str(path)] += 1
                    except (TypeError, ValueError):
                        continue

    chunks_per_path_avg = {
        path: round(chunks_per_path_sum[path] / chunks_per_path_count[path], 4)
        for path in sorted(chunks_per_path_count)
        if chunks_per_path_count[path]
    }
    return {
        "n": len(records),
        "profile_counts": dict(sorted(profile_counts.items())),
        "semantic_selected_count": semantic_selected_count,
        "semantic_fallback_count": semantic_fallback_count,
        "retrieval_cycle_counts": dict(sorted(retrieval_cycle_counts.items())),
        "check_cycle_counts": dict(sorted(check_cycle_counts.items())),
        "rewrite_count": rewrite_count,
        "decompose_count": decompose_count,
        "targeted_retrieval_count": targeted_retrieval_count,
        "grounded_count": grounded_count,
        "empty_final_chunks_count": empty_final_chunks_count,
        "paths_activated_counts": dict(sorted(paths_activated_counts.items())),
        "first_paths_counts": dict(sorted(first_paths_counts.items())),
        "chunks_per_path_avg": chunks_per_path_avg,
        "query_fail_count": query_fail_count,
    }


def _aggregate_agentic_jsonl(jsonl_path: Path) -> dict[str, Any]:
    if not jsonl_path.exists():
        return {}
    records: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return _aggregate_agentic_records(records) if records else {}


def _format_mapping_for_summary(mapping: Any) -> str:
    if not isinstance(mapping, dict) or not mapping:
        return "{}"
    return ", ".join(
        f"{key}={value}" for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    )


def _format_agentic_stats_lines(agentic_stats_by_mode: dict[str, dict[str, Any]]) -> list[str]:
    if not agentic_stats_by_mode:
        return []

    lines = ["", "Agentic trace summary:"]
    for mode, stats in sorted(agentic_stats_by_mode.items()):
        if not isinstance(stats, dict) or not stats:
            continue
        lines.append(
            f"  [{mode}] n={stats.get('n', 0)} "
            f"semantic_selected={stats.get('semantic_selected_count', 0)} "
            f"semantic_fallback={stats.get('semantic_fallback_count', 0)} "
            f"query_fail={stats.get('query_fail_count', 0)} "
            f"empty_final_chunks={stats.get('empty_final_chunks_count', 0)}"
        )
        lines.append(
            f"  [{mode}] rewrite={stats.get('rewrite_count', 0)} "
            f"decompose={stats.get('decompose_count', 0)} "
            f"targeted={stats.get('targeted_retrieval_count', 0)} "
            f"grounded={stats.get('grounded_count', 0)}"
        )
        lines.append(
            f"  [{mode}] profiles: "
            f"{_format_mapping_for_summary(stats.get('profile_counts'))}"
        )
        lines.append(
            f"  [{mode}] retrieval_cycles: "
            f"{_format_mapping_for_summary(stats.get('retrieval_cycle_counts'))}"
        )
        lines.append(
            f"  [{mode}] check_cycles: "
            f"{_format_mapping_for_summary(stats.get('check_cycle_counts'))}"
        )
        lines.append(
            f"  [{mode}] paths_activated: "
            f"{_format_mapping_for_summary(stats.get('paths_activated_counts'))}"
        )
        lines.append(
            f"  [{mode}] first_paths: "
            f"{_format_mapping_for_summary(stats.get('first_paths_counts'))}"
        )
        lines.append(
            f"  [{mode}] chunks_per_path_avg: "
            f"{_format_mapping_for_summary(stats.get('chunks_per_path_avg'))}"
        )

    return lines if len(lines) > 2 else []


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
    get_eval_query_overrides: Callable[[str], dict[str, str]],
    chunk_source_map: dict[str, dict[str, Any]],
    query_kwargs: dict[str, Any] | None = None,
    concurrency: int = 1,
    hybrid_enable_rerank: bool = True,
    ppr_enable_rerank: bool = False,
    qa_prompt_style: str = "lightrag",
    answer_parse_mode: str = "strip_references",
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
        raw_answer = ""
        chunks: list[dict[str, Any]] = []
        agentic_internal_answer: str | None = None
        agentic_trace: dict[str, Any] | None = None
        agentic_query_failed = False
        try:
            call_kwargs = _build_query_kwargs(
                query_overrides=query_overrides,
                wire_profile=wire_profile,
                **query_kwargs,
            )
            if mode == "agentic" and qa_prompt_style == "semantic_cot":
                result = await service.query_with_trace(
                    workspace_id=workspace_id,
                    query=item["question"],
                    working_dir=working_dir,
                    mode=wire_mode,
                    **call_kwargs,
                )
                agentic_internal_answer = str(result.get("answer") or "")
                raw_trace = result.get("trace", {})
                agentic_trace = raw_trace if isinstance(raw_trace, dict) else {}
                if agentic_trace is not None:
                    agentic_trace.setdefault("grounded", result.get("grounded"))
                trace_data = agentic_trace.get("data", {}) if isinstance(agentic_trace, dict) else {}
                raw_chunks = trace_data.get("chunks", []) if isinstance(trace_data, dict) else []
                chunks = raw_chunks if isinstance(raw_chunks, list) else []
                qa_top_k = call_kwargs.get("ppr_qa_top_k") or call_kwargs.get("chunk_top_k")
                raw_answer = await _call_semantic_cot_qa(
                    service,
                    question=item["question"],
                    chunks=chunks,
                    chunk_source_map=chunk_source_map,
                    qa_top_k=int(qa_top_k) if qa_top_k is not None else None,
                )
            elif qa_prompt_style in ("semantic_cot", "kg_semantic_cot"):
                retrieval_kwargs = dict(call_kwargs)
                retrieval_kwargs["only_need_context"] = True
                result = await service.query_with_trace(
                    workspace_id=workspace_id,
                    query=item["question"],
                    working_dir=working_dir,
                    mode=wire_mode,
                    **retrieval_kwargs,
                )
                trace_data = result.get("trace", {}).get("data", {})
                raw_chunks = trace_data.get("chunks", []) if isinstance(trace_data, dict) else []
                chunks = raw_chunks if isinstance(raw_chunks, list) else []
                qa_top_k = call_kwargs.get("ppr_qa_top_k") or call_kwargs.get("chunk_top_k")
                if qa_prompt_style == "kg_semantic_cot":
                    raw_answer = await _call_kg_semantic_cot_qa(
                        service,
                        question=item["question"],
                        raw_data=trace_data if isinstance(trace_data, dict) else {},
                        chunk_source_map=chunk_source_map,
                        qa_top_k=int(qa_top_k) if qa_top_k is not None else None,
                    )
                else:
                    raw_answer = await _call_semantic_cot_qa(
                        service,
                        question=item["question"],
                        chunks=chunks,
                        chunk_source_map=chunk_source_map,
                        qa_top_k=int(qa_top_k) if qa_top_k is not None else None,
                    )
            else:
                result = await service.query_with_trace(
                    workspace_id=workspace_id,
                    query=item["question"],
                    working_dir=working_dir,
                    mode=wire_mode,
                    **call_kwargs,
                )
                raw_answer = result.get("answer", "")
                raw_chunks = result.get("trace", {}).get("data", {}).get("chunks", [])
                chunks = raw_chunks if isinstance(raw_chunks, list) else []
            answer = _parse_answer_text(raw_answer, answer_parse_mode)
        except Exception as e:
            print(f"  [WARN] query failed for id={item['id']}: {e}")
            answer = ""
            agentic_query_failed = mode == "agentic"

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
        if qa_prompt_style in ("semantic_cot", "kg_semantic_cot"):
            record["raw_pred"] = raw_answer
        if mode == "agentic":
            record["agentic_internal_answer"] = agentic_internal_answer or ""
            record["agentic_trace"] = agentic_trace or {}
            record["agentic_query_failed"] = agentic_query_failed
        if item.get("gold_source_keys"):
            record["gold_source_keys"] = item["gold_source_keys"]
        retrieved_sources = _resolve_retrieved_sources(chunks, chunk_source_map)
        if chunks:
            hits = sum(1 for c in chunks if _trace_chunk_id(c) in chunk_source_map)
            record["source_map_coverage"] = round(hits / len(chunks), 4)
        for k in recall_ks:
            r = _score_support_recall(
                chunks=chunks,
                item=item,
                k=k,
                chunk_source_map=chunk_source_map,
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

    metrics = _aggregate_jsonl(jsonl_path, recall_ks)
    if mode == "agentic":
        metrics["agentic_stats"] = _aggregate_agentic_jsonl(jsonl_path)
    return metrics


async def main(args: argparse.Namespace) -> None:
    # Defer heavy imports so --help works without lightrag/raganything installed
    from raganything.services.local_rag import LocalRagService, LocalRagSettings
    from evaluate_local.MultiHopQA.dataset_adapters import (
        load_hotpotqa, load_musique, load_2wiki, load_simpleqa,
        load_hotpotqa_hipporag2, load_musique_hipporag2, load_2wiki_hipporag2,
        score_em, score_f1,
        get_eval_query_overrides,
    )

    hipporag2_dir = Path(args.hipporag2_data_dir).resolve() if args.hipporag2_data_dir else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hipporag2_dir:
        hr2_loaders = {
            "hotpotqa": lambda **_: load_hotpotqa_hipporag2(hipporag2_dir),
            "musique":  lambda **_: load_musique_hipporag2(hipporag2_dir),
            "2wiki":    lambda **_: load_2wiki_hipporag2(hipporag2_dir),
        }
        if args.dataset not in hr2_loaders:
            raise SystemExit(
                f"--hipporag2-data-dir is not supported for dataset={args.dataset!r}. "
                "Supported: hotpotqa, musique, 2wiki"
            )
        print(f"[eval] HippoRAG2 mode: loading {args.dataset} from {hipporag2_dir}")
        all_items = hr2_loaders[args.dataset]()
        items = _select_hipporag2_eval_items(all_items, args.n_samples, args.seed)
        if len(items) != len(all_items):
            print(
                "[eval] HippoRAG2 query subset: "
                f"n={len(items)} of {len(all_items)}, seed={args.seed}"
            )
        effective_n_samples = 0
        effective_seed = 0
    else:
        loaders = {
            "hotpotqa": load_hotpotqa,
            "musique":  load_musique,
            "2wiki":    load_2wiki,
            "simpleqa": load_simpleqa,
        }
        print(f"[eval] Loading dataset: {args.dataset} (n={args.n_samples}, seed={args.seed})")
        items = loaders[args.dataset](n=args.n_samples, seed=args.seed)
        effective_n_samples = args.n_samples
        effective_seed = args.seed

    print(f"[eval] Loaded {len(items)} questions")

    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings)
    strict_source_map = args.dataset != "simpleqa" and not args.allow_missing_source_map
    chunk_source_map = _load_chunk_source_map(
        args.working_dir,
        dataset=args.dataset,
        workspace=args.workspace,
        n_samples=effective_n_samples,
        seed=effective_seed,
        strict=strict_source_map,
    )
    if args.dataset == "simpleqa":
        print("[eval] SimpleQA has no supporting facts/source corpus; Recall@K will be N/A")
    if chunk_source_map:
        print(f"[eval] Loaded chunk source map: {len(chunk_source_map)} chunks")
    else:
        print("[eval] No chunk source map found; Recall@K will be N/A (passage-level recall requires source map)")

    base_query_kwargs = {
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "naive_top_k": args.naive_top_k,
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
        "hub_penalty_threshold": args.hub_penalty_threshold,
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
        f"naive_top_k={args.naive_top_k}, "
        f"max_total_tokens={args.max_total_tokens}, concurrency={args.concurrency}, "
        f"qdrant_retrieval_mode={args.qdrant_retrieval_mode}, "
        f"ppr_top_k={args.ppr_top_k}, ppr_qa_top_k={args.ppr_qa_top_k}, "
        f"hub_penalty_threshold={args.hub_penalty_threshold}, "
        f"enable_kg_rerank={args.enable_kg_rerank}, "
        f"bypass_query_cache={args.bypass_query_cache}, "
        f"bypass_keywords_cache={args.bypass_keywords_cache}, "
        f"vlm_enhanced={args.vlm_enhanced}, "
        f"qa_prompt_style={args.qa_prompt_style}, answer_parse_mode={args.answer_parse_mode}"
    )

    results: dict[str, dict] = {}
    query_kwargs_by_mode: dict[str, dict[str, Any]] = {}
    agentic_stats_by_mode: dict[str, dict[str, Any]] = {}
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
            get_eval_query_overrides=get_eval_query_overrides,
            chunk_source_map=chunk_source_map,
            query_kwargs=base_query_kwargs,
            concurrency=args.concurrency,
            hybrid_enable_rerank=args.hybrid_enable_rerank,
            ppr_enable_rerank=args.ppr_enable_rerank,
            qa_prompt_style=args.qa_prompt_style,
            answer_parse_mode=args.answer_parse_mode,
        )
        agentic_stats = metrics.pop("agentic_stats", None)
        if agentic_stats:
            agentic_stats_by_mode[mode] = agentic_stats
        results[mode] = metrics
        print(f"  [{mode}] EM={metrics.get('em', 0):.4f}  F1={metrics.get('f1', 0):.4f}")

    summary_path = output_dir / f"{args.dataset}_summary.json"
    summary = {
        "dataset": args.dataset,
        "corpus_source": "hipporag2" if hipporag2_dir else "huggingface",
        "hipporag2_data_dir": str(hipporag2_dir) if hipporag2_dir else None,
        "n_queries": len(items),
        "n_samples": effective_n_samples,
        "seed": effective_seed,
        "query_n_samples_arg": args.n_samples,
        "query_seed_arg": args.seed,
        "recall_k": args.recall_k,
        "concurrency": args.concurrency,
        "qa_prompt_style": args.qa_prompt_style,
        "answer_parse_mode": args.answer_parse_mode,
        "base_query_kwargs": {k: v for k, v in base_query_kwargs.items() if v is not None},
        "query_kwargs_by_mode": query_kwargs_by_mode,
        "agentic_stats_by_mode": agentic_stats_by_mode,
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
    for line in _format_agentic_stats_lines(agentic_stats_by_mode):
        print(line)
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
    p.add_argument(
        "--hipporag2-data-dir",
        default=None,
        dest="hipporag2_data_dir",
        help=(
            "Path to HippoRAG2 dataset directory (from download_hipporag2_datasets.py). "
            "When set, --n-samples/--seed select a fixed query subset only; "
            "the workspace/source-map identity remains the full HippoRAG2 corpus."
        ),
    )
    p.add_argument("--modes",       nargs="+", default=["naive", "hybrid", "ppr", "auto", "full"],
                   choices=VALID_MODES, metavar="MODE")
    p.add_argument("--n-samples",   type=int, default=1000, dest="n_samples",
                   help="Questions to sample when NOT using --hipporag2-data-dir (default 1000)")
    p.add_argument("--recall-k",    type=int, nargs="+", default=[2, 5], dest="recall_k")
    p.add_argument("--output-dir",  required=True, dest="output_dir")
    p.add_argument("--log-file", default=None, dest="log_file")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--top-k", type=int, default=10, dest="top_k")
    p.add_argument("--chunk-top-k", type=int, default=5, dest="chunk_top_k")
    p.add_argument(
        "--naive-top-k",
        type=int,
        default=None,
        dest="naive_top_k",
        help="Naive chunks VDB retrieval count before reranking. Omit to use LightRAG default.",
    )
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
    p.add_argument("--hub-penalty-threshold", type=int, default=50, dest="hub_penalty_threshold")
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
    p.add_argument("--qa-prompt-style", default="lightrag", choices=VALID_QA_PROMPT_STYLES, dest="qa_prompt_style")
    p.add_argument("--answer-parse-mode", default=None, choices=VALID_ANSWER_PARSE_MODES, dest="answer_parse_mode")
    args = p.parse_args()
    if args.answer_parse_mode is None:
        args.answer_parse_mode = (
            "answer_marker"
            if args.qa_prompt_style in ("semantic_cot", "kg_semantic_cot")
            else "strip_references"
        )
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.n_samples < 0:
        raise SystemExit("--n-samples must be >= 0")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.chunk_top_k <= 0:
        raise SystemExit("--chunk-top-k must be > 0")
    if args.naive_top_k is not None and args.naive_top_k <= 0:
        raise SystemExit("--naive-top-k must be > 0")
    if args.max_total_tokens <= 0:
        raise SystemExit("--max-total-tokens must be > 0")
    if args.ppr_top_k <= 0:
        raise SystemExit("--ppr-top-k must be > 0")
    if args.ppr_qa_top_k <= 0:
        raise SystemExit("--ppr-qa-top-k must be > 0")
    if args.ppr_qa_top_k > args.ppr_top_k:
        raise SystemExit("--ppr-qa-top-k must be <= --ppr-top-k")
    if args.hub_penalty_threshold < 0:
        raise SystemExit("--hub-penalty-threshold must be >= 0")
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
