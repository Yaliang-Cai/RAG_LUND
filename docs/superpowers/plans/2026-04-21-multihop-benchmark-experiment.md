# Multi-Hop QA Benchmark Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation pipeline that compares PPR vs hybrid vs mix query modes on HotpotQA, MuSIQue, 2WikiMultiHopQA, and SimpleQA using EM + F1 + Recall@K metrics.

**Architecture:** One shared index per dataset (pre-built, not in scope); `evaluate_multihop.py` loads the dataset, runs each query mode sequentially against the index via `LocalRagService.query_with_trace()`, writes per-question JSONL results in real-time, then aggregates into a summary JSON. Dataset-specific `user_prompt` + `response_type="Short Answer"` overrides are injected into each query to suppress verbose markdown output that breaks EM/F1.

**Tech Stack:** Python 3.12, `datasets` (HuggingFace), `asyncio`, existing `LocalRagService` / `LocalRagSettings` from `raganything/services/local_rag.py`, `pytest` + `pytest-asyncio` for tests.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `rag-anything/evaluate_local/MultiHopQA/__init__.py` | Package marker |
| Create | `rag-anything/evaluate_local/MultiHopQA/dataset_adapters.py` | Load functions, score functions, prompt overrides |
| Create | `rag-anything/evaluate_local/MultiHopQA/download_datasets.py` | One-time HF download script |
| Create | `rag-anything/evaluate_local/MultiHopQA/evaluate_multihop.py` | Main eval script |
| Create | `rag-anything/evaluate_local/MultiHopQA/tests/__init__.py` | Test package marker |
| Create | `rag-anything/evaluate_local/MultiHopQA/tests/test_dataset_adapters.py` | Unit tests for scoring + adapters |
| Create | `rag-anything/run_multihop_evals.py` | Orchestrator |

---

## Task 1: Scoring functions in `dataset_adapters.py`

**Files:**
- Create: `rag-anything/evaluate_local/MultiHopQA/__init__.py`
- Create: `rag-anything/evaluate_local/MultiHopQA/tests/__init__.py`
- Create: `rag-anything/evaluate_local/MultiHopQA/dataset_adapters.py` (scoring section only)
- Create: `rag-anything/evaluate_local/MultiHopQA/tests/test_dataset_adapters.py`

- [ ] **Step 1: Create package markers**

```bash
touch rag-anything/evaluate_local/MultiHopQA/__init__.py
mkdir -p rag-anything/evaluate_local/MultiHopQA/tests
touch rag-anything/evaluate_local/MultiHopQA/tests/__init__.py
```

- [ ] **Step 2: Write failing tests for scoring functions**

Create `rag-anything/evaluate_local/MultiHopQA/tests/test_dataset_adapters.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "rag-anything"))

import pytest
from evaluate_local.MultiHopQA.dataset_adapters import (
    normalize_answer,
    score_em,
    score_f1,
    score_recall_at_k,
    get_eval_query_overrides,
)


def test_normalize_strips_articles():
    assert normalize_answer("the Berlin") == "berlin"
    assert normalize_answer("a dog") == "dog"
    assert normalize_answer("an apple") == "apple"


def test_normalize_strips_punctuation():
    assert normalize_answer("yes.") == "yes"
    assert normalize_answer("New York, City") == "new york city"


def test_score_em_exact():
    assert score_em("berlin", "Berlin") == 1.0


def test_score_em_mismatch():
    assert score_em("paris", "Berlin") == 0.0


def test_score_em_multiple_gold():
    assert score_em("yes", ["Yes", "no"]) == 1.0
    assert score_em("maybe", ["Yes", "no"]) == 0.0


def test_score_f1_perfect():
    assert score_f1("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_score_f1_partial():
    f1 = score_f1("cat sat", "the cat sat on mat")
    assert 0.0 < f1 < 1.0


def test_score_f1_no_overlap():
    assert score_f1("dog", "cat") == 0.0


def test_score_f1_multiple_gold_takes_max():
    f1_a = score_f1("Berlin is a city", "Berlin")
    f1_b = score_f1("Berlin is a city", "London")
    result = score_f1("Berlin is a city", ["Berlin", "London"])
    assert result == pytest.approx(max(f1_a, f1_b))


def test_recall_at_k_all_covered():
    chunks = [{"content": "The capital of Germany is Berlin."}]
    facts = ["Berlin"]
    assert score_recall_at_k(chunks, facts, k=1) == pytest.approx(1.0)


def test_recall_at_k_none_covered():
    chunks = [{"content": "Paris is in France."}]
    facts = ["Berlin"]
    assert score_recall_at_k(chunks, facts, k=1) == pytest.approx(0.0)


def test_recall_at_k_respects_k():
    chunks = [
        {"content": "Irrelevant text."},
        {"content": "Berlin is the answer."},
    ]
    facts = ["Berlin"]
    assert score_recall_at_k(chunks, facts, k=1) == pytest.approx(0.0)
    assert score_recall_at_k(chunks, facts, k=2) == pytest.approx(1.0)


def test_recall_at_k_none_facts_returns_none():
    assert score_recall_at_k([{"content": "x"}], None, k=5) is None


def test_get_eval_query_overrides_hotpotqa():
    overrides = get_eval_query_overrides("hotpotqa")
    assert overrides["response_type"] == "Short Answer"
    assert "yes" in overrides["user_prompt"].lower() or "short" in overrides["user_prompt"].lower()


def test_get_eval_query_overrides_simpleqa():
    overrides = get_eval_query_overrides("simpleqa")
    assert overrides["response_type"] == "Short Answer"


def test_get_eval_query_overrides_unknown_raises():
    with pytest.raises(ValueError):
        get_eval_query_overrides("unknown_dataset")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd rag-anything
python -m pytest evaluate_local/MultiHopQA/tests/test_dataset_adapters.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError` or `ImportError` (file doesn't exist yet).

- [ ] **Step 4: Create `dataset_adapters.py` with scoring functions**

Create `rag-anything/evaluate_local/MultiHopQA/dataset_adapters.py`:

```python
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Answer normalization (standard HotpotQA / SQuAD normalization)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


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
        any(normalize_answer(fact) in t for t in top_texts)
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd rag-anything
python -m pytest evaluate_local/MultiHopQA/tests/test_dataset_adapters.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd rag-anything
git add evaluate_local/MultiHopQA/__init__.py \
        evaluate_local/MultiHopQA/tests/__init__.py \
        evaluate_local/MultiHopQA/dataset_adapters.py \
        evaluate_local/MultiHopQA/tests/test_dataset_adapters.py
git commit -m "feat: add dataset_adapters scoring functions and tests"
```

---

## Task 2: Load functions in `dataset_adapters.py`

**Files:**
- Modify: `rag-anything/evaluate_local/MultiHopQA/dataset_adapters.py` (append load functions)
- Modify: `rag-anything/evaluate_local/MultiHopQA/tests/test_dataset_adapters.py` (append load tests)

**Prerequisites:** `pip install datasets` on the server. The load functions call HuggingFace `datasets.load_dataset()`; they do NOT download at import time — only when called.

- [ ] **Step 1: Write failing load tests**

Append to `rag-anything/evaluate_local/MultiHopQA/tests/test_dataset_adapters.py`:

```python
# ---------------------------------------------------------------------------
# Load function tests — use n=3 to keep fast; requires internet/HF cache
# ---------------------------------------------------------------------------
from evaluate_local.MultiHopQA.dataset_adapters import (
    load_hotpotqa,
    load_musique,
    load_2wiki,
    load_simpleqa,
)


def _check_items(items, expect_supporting: bool):
    assert len(items) > 0
    for item in items:
        assert "id" in item
        assert "question" in item and item["question"]
        assert "answer" in item and item["answer"]
        if expect_supporting:
            assert "supporting_facts" in item
            assert isinstance(item["supporting_facts"], list)
            assert all(isinstance(f, str) for f in item["supporting_facts"])
        else:
            assert item.get("supporting_facts") is None


def test_load_hotpotqa_returns_correct_shape():
    items = load_hotpotqa(n=3, seed=42)
    assert len(items) == 3
    _check_items(items, expect_supporting=True)


def test_load_musique_returns_correct_shape():
    items = load_musique(n=3, seed=42)
    assert len(items) == 3
    _check_items(items, expect_supporting=True)


def test_load_2wiki_returns_correct_shape():
    items = load_2wiki(n=3, seed=42)
    assert len(items) == 3
    _check_items(items, expect_supporting=True)


def test_load_simpleqa_returns_correct_shape():
    items = load_simpleqa(n=3, seed=42)
    assert len(items) == 3
    _check_items(items, expect_supporting=False)


def test_load_hotpotqa_seed_reproducible():
    a = load_hotpotqa(n=5, seed=42)
    b = load_hotpotqa(n=5, seed=42)
    assert [x["id"] for x in a] == [x["id"] for x in b]


def test_load_hotpotqa_different_seeds_differ():
    a = load_hotpotqa(n=5, seed=42)
    b = load_hotpotqa(n=5, seed=99)
    assert [x["id"] for x in a] != [x["id"] for x in b]
```

- [ ] **Step 2: Run to verify failure**

```bash
cd rag-anything
python -m pytest evaluate_local/MultiHopQA/tests/test_dataset_adapters.py::test_load_hotpotqa_returns_correct_shape -v
```

Expected: `ImportError` (functions not defined yet).

- [ ] **Step 3: Append load functions to `dataset_adapters.py`**

Append to the bottom of `rag-anything/evaluate_local/MultiHopQA/dataset_adapters.py`:

```python
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
    ds = load_dataset("voidful/2WikiMultihopQA", split="validation", trust_remote_code=True)
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
```

- [ ] **Step 4: Run load tests (requires HF internet access)**

```bash
cd rag-anything
python -m pytest evaluate_local/MultiHopQA/tests/test_dataset_adapters.py -v -k "load"
```

Expected: all 6 load tests PASS. If a HuggingFace dataset ID is wrong, the error message will show the correct name — update the `load_dataset()` call accordingly.

- [ ] **Step 5: Run full test suite**

```bash
cd rag-anything
python -m pytest evaluate_local/MultiHopQA/tests/test_dataset_adapters.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd rag-anything
git add evaluate_local/MultiHopQA/dataset_adapters.py \
        evaluate_local/MultiHopQA/tests/test_dataset_adapters.py
git commit -m "feat: add dataset load functions (hotpotqa/musique/2wiki/simpleqa)"
```

---

## Task 3: `download_datasets.py` — one-time cache script

**Files:**
- Create: `rag-anything/evaluate_local/MultiHopQA/download_datasets.py`

No unit tests (pure I/O side-effect script).

- [ ] **Step 1: Create download script**

Create `rag-anything/evaluate_local/MultiHopQA/download_datasets.py`:

```python
#!/usr/bin/env python
"""One-time script to download all benchmark datasets to HuggingFace local cache.

Usage:
    python evaluate_local/MultiHopQA/download_datasets.py
    python evaluate_local/MultiHopQA/download_datasets.py --data-dir /data/hf_cache
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="HuggingFace cache dir (sets HF_DATASETS_CACHE)")
    args = p.parse_args()

    if args.data_dir:
        os.environ["HF_DATASETS_CACHE"] = args.data_dir
        print(f"[download] HF cache dir: {args.data_dir}")

    from datasets import load_dataset

    configs = [
        ("hotpot_qa",             "distractor",  "validation"),
        ("dgslibisey/MuSiQue",    None,           "validation"),
        ("voidful/2WikiMultihopQA", None,         "validation"),
        ("basicv8vc/SimpleQA",    None,           "test"),
    ]

    for name, config, split in configs:
        label = f"{name}[{config or 'default'}]/{split}"
        print(f"[download] Downloading {label} ...")
        try:
            kwargs = {"trust_remote_code": True}
            if config:
                load_dataset(name, config, split=split, **kwargs)
            else:
                load_dataset(name, split=split, **kwargs)
            print(f"[download] OK: {label}")
        except Exception as e:
            print(f"[download] FAILED: {label} — {e}")

    print("[download] Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run download script on server**

```bash
cd rag-anything
python evaluate_local/MultiHopQA/download_datasets.py \
    --data-dir /data/y50056788/Yaliang/datasets_for_eval/hf_cache
```

Expected: four `[download] OK:` lines. If any fails, the error message will show the correct HF dataset name — update `configs` list in the script and re-run.

- [ ] **Step 3: Commit**

```bash
cd rag-anything
git add evaluate_local/MultiHopQA/download_datasets.py
git commit -m "feat: add one-time dataset download script"
```

---

## Task 4: `evaluate_multihop.py` — main evaluation script

**Files:**
- Create: `rag-anything/evaluate_local/MultiHopQA/evaluate_multihop.py`

- [ ] **Step 1: Create the evaluation script**

Create `rag-anything/evaluate_local/MultiHopQA/evaluate_multihop.py`:

```python
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

from raganything.services.local_rag import LocalRagService, LocalRagSettings
from evaluate_local.MultiHopQA.dataset_adapters import (
    load_hotpotqa, load_musique, load_2wiki, load_simpleqa,
    score_em, score_f1, score_recall_at_k,
    get_eval_query_overrides,
)

VALID_MODES = ("ppr", "ppr_local", "global", "local", "hybrid", "mix", "naive", "rrf", "bypass")
VALID_DATASETS = ("hotpotqa", "musique", "2wiki", "simpleqa")

_REFERENCES_RE = re.compile(r"#+\s*references?.*", re.IGNORECASE | re.DOTALL)


def _strip_references(text: str) -> str:
    return _REFERENCES_RE.sub("", text).strip()


def _load_dataset(dataset: str, n: int, seed: int) -> list[dict]:
    loaders = {
        "hotpotqa": load_hotpotqa,
        "musique":  load_musique,
        "2wiki":    load_2wiki,
        "simpleqa": load_simpleqa,
    }
    return loaders[dataset](n=n, seed=seed)


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
    service: LocalRagService,
    workspace_id: str,
    working_dir: str,
    items: list[dict],
    mode: str,
    dataset: str,
    recall_ks: list[int],
    output_dir: Path,
    resume: bool,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] Loading dataset: {args.dataset} (n={args.n_samples}, seed={args.seed})")
    items = _load_dataset(args.dataset, args.n_samples, args.seed)
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
        )
        results[mode] = metrics
        print(f"  [{mode}] EM={metrics.get('em'):.4f}  F1={metrics.get('f1'):.4f}")

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
```

- [ ] **Step 2: Smoke-test the script (no network, just verify CLI parses)**

```bash
cd rag-anything
python evaluate_local/MultiHopQA/evaluate_multihop.py --help
```

Expected: prints help with all arguments, exits 0.

- [ ] **Step 3: Commit**

```bash
cd rag-anything
git add evaluate_local/MultiHopQA/evaluate_multihop.py
git commit -m "feat: add evaluate_multihop.py main eval script"
```

---

## Task 5: `run_multihop_evals.py` — orchestrator

**Files:**
- Create: `rag-anything/run_multihop_evals.py`

- [ ] **Step 1: Create orchestrator**

Create `rag-anything/run_multihop_evals.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Hop QA Evaluation Orchestrator
=====================================

Sequentially runs evaluate_multihop.py for each requested dataset and mode set.

Usage:
    python run_multihop_evals.py \
        --workspace my_wiki_workspace \
        --working-dir /data/y50056788/.../rag_workspaces/my_wiki_workspace \
        --output-dir /data/y50056788/.../multihop_results \
        --datasets hotpotqa musique 2wiki simpleqa \
        --modes ppr hybrid mix
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parent / "evaluate_local" / "MultiHopQA" / "evaluate_multihop.py"

VALID_DATASETS = ("hotpotqa", "musique", "2wiki", "simpleqa")
VALID_MODES = ("ppr", "ppr_local", "global", "local", "hybrid", "mix", "naive", "rrf", "bypass")


def _run_one(dataset: str, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, str(_SCRIPT),
        "--dataset",     dataset,
        "--workspace",   args.workspace,
        "--working-dir", args.working_dir,
        "--output-dir",  args.output_dir,
        "--modes",       *args.modes,
        "--n-samples",   str(args.n_samples),
        "--seed",        str(args.seed),
        "--recall-k",    *[str(k) for k in args.recall_k],
    ]
    if args.resume:
        cmd.append("--resume")

    print(f"\n{'='*60}")
    print(f"[orchestrator] Dataset: {dataset}  modes: {args.modes}")
    print(f"[orchestrator] cmd: {' '.join(cmd)}")
    print("="*60)

    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-hop QA eval orchestrator")
    p.add_argument("--workspace",   required=True)
    p.add_argument("--working-dir", required=True, dest="working_dir")
    p.add_argument("--output-dir",  required=True, dest="output_dir")
    p.add_argument("--datasets",    nargs="+", default=list(VALID_DATASETS),
                   choices=VALID_DATASETS, metavar="DATASET")
    p.add_argument("--modes",       nargs="+", default=["ppr", "hybrid", "mix"],
                   choices=VALID_MODES, metavar="MODE")
    p.add_argument("--n-samples",   type=int, default=500, dest="n_samples")
    p.add_argument("--recall-k",    type=int, nargs="+", default=[5, 10, 20], dest="recall_k")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    failed = []
    for dataset in args.datasets:
        rc = _run_one(dataset, args)
        if rc != 0:
            print(f"[orchestrator] WARN: {dataset} exited with code {rc}")
            failed.append(dataset)

    print(f"\n[orchestrator] Done. Results in: {args.output_dir}")
    if failed:
        print(f"[orchestrator] Failed datasets: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test**

```bash
cd rag-anything
python run_multihop_evals.py --help
```

Expected: prints help, exits 0.

- [ ] **Step 3: Commit**

```bash
cd rag-anything
git add run_multihop_evals.py
git commit -m "feat: add run_multihop_evals.py orchestrator"
```

---

## Task 6: End-to-end smoke test on server

Run against a live workspace with 5 questions to verify the full pipeline works before launching the full 500-question run.

- [ ] **Step 1: Run 5-question smoke test with one mode**

```bash
cd rag-anything
python evaluate_local/MultiHopQA/evaluate_multihop.py \
    --dataset hotpotqa \
    --workspace <YOUR_HOTPOTQA_WORKSPACE_ID> \
    --working-dir <PATH_TO_WORKSPACE_DIR> \
    --output-dir /tmp/multihop_smoke \
    --modes hybrid \
    --n-samples 5 \
    --seed 42
```

Expected output ends with a metrics table. Verify:
- `hybrid_results.jsonl` has 5 lines
- `hotpotqa_summary.json` is valid JSON with `results.hybrid.em` and `results.hybrid.f1`
- Answers are short phrases (not long markdown blocks with References)

- [ ] **Step 2: Test resume behavior**

```bash
# Interrupt after a few questions with Ctrl+C, then re-run with --resume
python evaluate_local/MultiHopQA/evaluate_multihop.py \
    --dataset hotpotqa \
    --workspace <YOUR_HOTPOTQA_WORKSPACE_ID> \
    --working-dir <PATH_TO_WORKSPACE_DIR> \
    --output-dir /tmp/multihop_smoke \
    --modes hybrid \
    --n-samples 5 \
    --seed 42 \
    --resume
```

Expected: skips already-answered questions (counter starts from > 0), final JSONL still has exactly 5 lines.

- [ ] **Step 3: Launch full evaluation**

```bash
cd rag-anything
nohup python run_multihop_evals.py \
    --workspace <YOUR_WORKSPACE_ID> \
    --working-dir <PATH_TO_WORKSPACE_DIR> \
    --output-dir /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/MultiHopQA/results \
    --datasets hotpotqa musique 2wiki simpleqa \
    --modes ppr hybrid mix \
    --n-samples 500 \
    --seed 42 \
    > /tmp/multihop_eval.log 2>&1 &
echo "PID: $!"
```

Monitor:
```bash
tail -f /tmp/multihop_eval.log
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Query-mode only (no indexing) — Task 4 uses `query_with_trace()` on existing workspace
- [x] 4 datasets with load functions — Task 2
- [x] HF download script — Task 3
- [x] EM + F1 — `score_em`, `score_f1` in Task 1
- [x] Recall@K (5, 10, 20) — `score_recall_at_k` in Task 1; SimpleQA returns None
- [x] Dataset-specific prompt overrides — `get_eval_query_overrides` + injected in Task 4
- [x] References stripping post-processing — `_strip_references` in Task 4
- [x] JSONL real-time write (crash-safe) — `_append_jsonl` in Task 4
- [x] `--resume` skip by question `id` — `_load_existing_ids` in Task 4
- [x] Configurable modes via CLI — `--modes` nargs in Tasks 4 & 5
- [x] Orchestrator for all datasets — Task 5
- [x] Summary JSON + printed table — `_aggregate_jsonl` + print block in Task 4

**Type consistency:**
- `score_recall_at_k` returns `float | None`; Task 4 checks `if r.get(key) is not None` before averaging — consistent
- `get_eval_query_overrides` returns `dict[str, str]`; Task 4 unpacks via `**query_overrides` — consistent
- `load_*` functions return `list[dict]` with keys `id/question/answer/supporting_facts` — consistent with Task 4 access
