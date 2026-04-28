from __future__ import annotations

import hashlib
import random
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


def _format_paragraph(title: Any, text: Any) -> str:
    title_text = str(title or "").strip()
    body_text = str(text or "").strip()
    if title_text and body_text:
        return f"{title_text}\n{body_text}"
    return title_text or body_text


def paragraph_source_key(dataset: str, title: Any, text: Any) -> str:
    """Stable paragraph identity used by indexing and retrieval metrics.

    This intentionally keys on the full formatted paragraph, not only title, so
    same-title passages from MuSiQue/2Wiki remain distinct.
    """
    content = _format_paragraph(title, text).strip()
    digest = hashlib.sha1(f"{dataset}\n{content}".encode("utf-8")).hexdigest()
    return f"{dataset}:{digest}"


def dedupe_corpus_paragraphs(
    *,
    dataset: str,
    paragraphs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate exact paragraphs without collapsing same-title variants."""
    seen: set[str] = set()
    corpus: list[dict[str, Any]] = []
    for row in paragraphs:
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        if not _format_paragraph(title, text).strip():
            continue
        source_key = paragraph_source_key(dataset, title, text)
        if source_key in seen:
            continue
        seen.add(source_key)
        corpus.append({"title": title, "text": text, "source_key": source_key})
    return corpus


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

def _fact_key(fact: str, n_words: int = 10) -> str:
    """First n_words of fact lowercased — used as a fingerprint for recall matching.

    Avoids applying normalize_answer (which strips punctuation) to long paragraph
    texts, where aggressive normalization destroys substring match signal.
    """
    return " ".join(fact.lower().split()[:n_words])


def score_recall_at_k(
    chunks: list[dict[str, Any]],
    supporting_facts: list[str] | None,
    k: int,
) -> float | None:
    """Return fraction of supporting facts covered by top-k chunks, or None if no facts.

    supporting_facts are paragraph-level texts (not individual sentences).
    Matching uses the first 10 words of each fact as a fingerprint so that
    minor whitespace / punctuation differences don't cause false misses.
    """
    if not supporting_facts:
        return None
    top_texts = [c.get("content", "").lower() for c in chunks[:k]]
    covered = sum(
        any(_fact_key(fact) in t for t in top_texts)
        for fact in supporting_facts
    )
    return covered / len(supporting_facts)


def score_recall_at_k_by_source_keys(
    retrieved_sources: list[dict[str, Any]],
    gold_source_keys: list[str] | None,
    k: int,
) -> float | None:
    """Passage recall from stable source keys."""
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

def _sample(items: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def _unique_preserve_order(values: list[Any]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        key = str(value)
        seen[key] = None
    return list(seen)


def _join_sentences(sentences: Any) -> str:
    if hasattr(sentences, "tolist"):
        sentences = sentences.tolist()
    return " ".join(str(s) for s in sentences)


def _source_record(dataset: str, title: Any, text: Any) -> dict[str, str]:
    title_text = str(title or "").strip()
    body_text = str(text or "").strip()
    return {
        "source_key": paragraph_source_key(dataset, title_text, body_text),
        "title": title_text,
        "text": body_text,
    }


def load_hotpotqa(n: int = 500, seed: int = 42) -> list[dict]:
    """Load HotpotQA distractor dev set. supporting_facts = full paragraph texts for gold titles."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    raw = list(ds)
    sampled = _sample(raw, n, seed)
    result = []
    for row in sampled:
        ctx = {
            str(title): _join_sentences(sents)
            for title, sents in zip(row["context"]["title"], row["context"]["sentences"])
        }
        # Collect unique gold titles (preserving order) and join their sentences into paragraphs.
        # Full-paragraph facts give reliable Recall@K signal at chunk granularity.
        facts = []
        gold_sources = []
        for title in _unique_preserve_order(list(row["supporting_facts"]["title"])):
            text = ctx.get(title, "")
            if text:
                facts.append(text)
                gold_sources.append(_source_record("hotpotqa", title, text))
        result.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": facts,
            "gold_source_keys": [source["source_key"] for source in gold_sources],
            "gold_sources": gold_sources,
        })
    return result


def extract_corpus_hotpotqa(n: int = 500, seed: int = 42) -> list[dict]:
    """Return unique context paragraphs for n sampled HotpotQA questions (for indexing)."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    raw = list(ds)
    sampled = _sample(raw, n, seed)
    paragraphs: list[dict] = []
    for row in sampled:
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            paragraphs.append({"title": title, "text": _join_sentences(sents)})
    return dedupe_corpus_paragraphs(dataset="hotpotqa", paragraphs=paragraphs)


def load_musique(n: int = 500, seed: int = 42) -> list[dict]:
    """Load MuSiQue answerable dev set. supporting_facts = list of supporting paragraph texts."""
    from datasets import load_dataset
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    raw = [row for row in ds if row.get("answerable", True)]
    sampled = _sample(raw, n, seed)
    result = []
    for row in sampled:
        facts = []
        gold_sources = []
        seen_gold: set[str] = set()
        for p in row.get("paragraphs", []):
            if not p.get("is_supporting", False):
                continue
            source = _source_record("musique", p.get("title", ""), p.get("paragraph_text", ""))
            if source["source_key"] in seen_gold:
                continue
            seen_gold.add(source["source_key"])
            facts.append(source["text"])
            gold_sources.append(source)
        result.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": facts,
            "gold_source_keys": [source["source_key"] for source in gold_sources],
            "gold_sources": gold_sources,
        })
    return result


def extract_corpus_musique(n: int = 500, seed: int = 42) -> list[dict]:
    """Return unique context paragraphs for n sampled MuSiQue questions (for indexing)."""
    from datasets import load_dataset
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    raw = [row for row in ds if row.get("answerable", True)]
    sampled = _sample(raw, n, seed)
    paragraphs: list[dict] = []
    for row in sampled:
        for p in row.get("paragraphs", []):
            title = p.get("title", "")
            text = p.get("paragraph_text", "")
            paragraphs.append({"title": title, "text": text})
    return dedupe_corpus_paragraphs(dataset="musique", paragraphs=paragraphs)


def load_2wiki(n: int = 500, seed: int = 42) -> list[dict]:
    """Load 2WikiMultiHopQA dev set. supporting_facts = full paragraph texts for gold titles."""
    from datasets import load_dataset
    # framolfese/ is the working public mirror; original plan used voidful/ which has a broken loading script
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    raw = list(ds)
    sampled = _sample(raw, n, seed)
    result = []
    for row in sampled:
        ctx = {
            str(title): _join_sentences(sents)
            for title, sents in zip(row["context"]["title"], row["context"]["sentences"])
        }
        facts = []
        gold_sources = []
        for title in _unique_preserve_order(list(row["supporting_facts"]["title"])):
            text = ctx.get(title, "")
            if text:
                facts.append(text)
                gold_sources.append(_source_record("2wiki", title, text))
        result.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": facts,
            "gold_source_keys": [source["source_key"] for source in gold_sources],
            "gold_sources": gold_sources,
        })
    return result


def extract_corpus_2wiki(n: int = 500, seed: int = 42) -> list[dict]:
    """Return unique context paragraphs for n sampled 2WikiMultiHopQA questions (for indexing)."""
    from datasets import load_dataset
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    raw = list(ds)
    sampled = _sample(raw, n, seed)
    paragraphs: list[dict] = []
    for row in sampled:
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            paragraphs.append({"title": title, "text": _join_sentences(sents)})
    return dedupe_corpus_paragraphs(dataset="2wiki", paragraphs=paragraphs)


def load_simpleqa(n: int = 500, seed: int = 42) -> list[dict]:
    """Load SimpleQA test set. No supporting facts."""
    from datasets import load_dataset
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
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
