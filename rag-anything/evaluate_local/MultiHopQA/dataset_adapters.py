from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Answer normalization (standard HotpotQA / SQuAD normalization)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

def score_em(pred: str, gold: str | list[str]) -> float:
    pred_norm = normalize_answer(pred)
    if isinstance(gold, list):
        return float(any(pred_norm == normalize_answer(g) for g in gold))
    return float(pred_norm == normalize_answer(gold))


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

def _f1_single(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_f1(pred: str, gold: str | list[str]) -> float:
    if isinstance(gold, list):
        if not gold:
            return 0.0
        return max(_f1_single(pred, g) for g in gold)
    return _f1_single(pred, gold)


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def score_recall_at_k(
    chunks: list[dict[str, Any]],
    supporting_facts: list[str] | None,
    k: int,
) -> float | None:
    """Return fraction of supporting facts covered by top-k chunks, or None if no facts."""
    if not supporting_facts:
        return None
    top_chunks = chunks[:k]
    top_texts = [normalize_answer(c.get("content", "")) for c in top_chunks]
    covered = sum(
        any(
            " " + normalize_answer(fact) + " " in " " + t + " "
            for t in top_texts
        )
        for fact in supporting_facts
    )
    return covered / len(supporting_facts)


# ---------------------------------------------------------------------------
# Dataset-specific prompt overrides
# ---------------------------------------------------------------------------

_SHORT_ANSWER_INSTRUCTION = (
    "Answer with a short phrase or entity name only. "
    "For yes/no questions, reply only 'yes' or 'no'. "
    "Do NOT include reasoning, citations, markdown, or a References section."
)

_SHORT_ANSWER_FACTUAL = (
    "Answer with a concise factual phrase only. "
    "Do NOT include reasoning, citations, markdown, or a References section."
)

_DATASET_OVERRIDES: dict[str, dict[str, str]] = {
    "hotpotqa": {"response_type": "Short Answer", "user_prompt": _SHORT_ANSWER_INSTRUCTION},
    "musique":  {"response_type": "Short Answer", "user_prompt": _SHORT_ANSWER_INSTRUCTION},
    "2wiki":    {"response_type": "Short Answer", "user_prompt": _SHORT_ANSWER_INSTRUCTION},
    "simpleqa": {"response_type": "Short Answer", "user_prompt": _SHORT_ANSWER_FACTUAL},
}


def get_eval_query_overrides(dataset: str) -> dict[str, str]:
    key = dataset.lower()
    if key not in _DATASET_OVERRIDES:
        raise ValueError(f"Unknown dataset: {dataset!r}. Valid: {sorted(_DATASET_OVERRIDES)}")
    return dict(_DATASET_OVERRIDES[key])


# ---------------------------------------------------------------------------
# Dataset load functions
# ---------------------------------------------------------------------------
import random


def _sample(items: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def load_hotpotqa(n: int = 500, seed: int = 42) -> list[dict]:
    """Load HotpotQA distractor dev set. supporting_facts = list of sentence strings."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    raw = list(ds)
    sampled = _sample(raw, n, seed)
    result = []
    for row in sampled:
        # Build a mapping: title -> list[sentence_text]
        ctx = {title: sents for title, sents in zip(row["context"]["title"], row["context"]["sentences"])}
        facts = []
        for title, sent_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
            sents = ctx.get(title, [])
            if sent_id < len(sents):
                facts.append(sents[sent_id])
        result.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": facts,
        })
    return result


def load_musique(n: int = 500, seed: int = 42) -> list[dict]:
    """Load MuSiQue answerable dev set. supporting_facts = list of supporting paragraph texts."""
    from datasets import load_dataset
    ds = load_dataset("dgslibisey/MuSiQue", split="validation", trust_remote_code=True)
    raw = [row for row in ds if row.get("answerable", True)]
    sampled = _sample(raw, n, seed)
    result = []
    for row in sampled:
        facts = [
            p["paragraph_text"]
            for p in row.get("paragraphs", [])
            if p.get("is_supporting", False)
        ]
        result.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": facts,
        })
    return result


def load_2wiki(n: int = 500, seed: int = 42) -> list[dict]:
    """Load 2WikiMultiHopQA dev set. supporting_facts = list of sentence strings."""
    from datasets import load_dataset
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    raw = list(ds)
    sampled = _sample(raw, n, seed)
    result = []
    for row in sampled:
        # context: {"title": [...], "sentences": [[...]]}
        ctx = {title: sents for title, sents in zip(row["context"]["title"], row["context"]["sentences"])}
        facts = []
        for title, sent_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
            sents = ctx.get(title, [])
            if sent_id < len(sents):
                facts.append(sents[sent_id])
        result.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": facts,
        })
    return result


def load_simpleqa(n: int = 500, seed: int = 42) -> list[dict]:
    """Load SimpleQA test set. No supporting facts."""
    from datasets import load_dataset
    ds = load_dataset("basicv8vc/SimpleQA", split="test", trust_remote_code=True)
    raw = list(ds)
    sampled = _sample(raw, n, seed)
    result = []
    for row in sampled:
        result.append({
            "id": row.get("id", row["problem"]),
            "question": row["problem"],
            "answer": row["answer"],
            "supporting_facts": None,
        })
    return result
