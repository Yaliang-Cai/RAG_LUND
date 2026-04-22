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


def test_score_f1_empty_gold_list():
    assert score_f1("berlin", []) == 0.0


def test_get_eval_query_overrides_hotpotqa():
    overrides = get_eval_query_overrides("hotpotqa")
    assert overrides["response_type"] == "Short Answer"
    assert "yes" in overrides["user_prompt"].lower() and "no" in overrides["user_prompt"].lower()


def test_get_eval_query_overrides_simpleqa():
    overrides = get_eval_query_overrides("simpleqa")
    assert overrides["response_type"] == "Short Answer"


def test_get_eval_query_overrides_musique():
    overrides = get_eval_query_overrides("musique")
    assert overrides["response_type"] == "Short Answer"
    assert "yes" in overrides["user_prompt"].lower() and "no" in overrides["user_prompt"].lower()


def test_get_eval_query_overrides_2wiki():
    overrides = get_eval_query_overrides("2wiki")
    assert overrides["response_type"] == "Short Answer"
    assert "yes" in overrides["user_prompt"].lower() and "no" in overrides["user_prompt"].lower()


def test_get_eval_query_overrides_unknown_raises():
    with pytest.raises(ValueError):
        get_eval_query_overrides("unknown_dataset")


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
