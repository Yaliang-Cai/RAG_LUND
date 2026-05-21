import pytest
from raganything.retrieval.router_cache import RouterCache


def test_miss_returns_none():
    cache = RouterCache()
    assert cache.get("some query") is None


def test_put_then_get():
    cache = RouterCache()
    cache.put("what is BERT?", "semantic")
    entry = cache.get("what is BERT?")
    assert entry is not None
    assert entry["profile"] == "semantic"
    assert entry["outcome"] == "unknown"


def test_normalisation_ignores_case_and_whitespace():
    cache = RouterCache()
    cache.put("  What IS bert?  ", "semantic")
    assert cache.get("what is bert?") is not None


def test_mark_success():
    cache = RouterCache()
    cache.put("q", "semantic")
    cache.mark_success("q")
    assert cache.get("q")["outcome"] == "success"


def test_mark_failed_twice_marks_entry_failed():
    cache = RouterCache()
    cache.put("q", "multihop")
    cache.mark_failed("q")
    assert cache.get("q")["outcome"] == "unknown"  # not yet failed
    cache.mark_failed("q")
    assert cache.get("q")["outcome"] == "failed"


def test_mark_failed_three_times_evicts():
    cache = RouterCache()
    cache.put("q", "multihop")
    for _ in range(3):
        cache.mark_failed("q")
    assert cache.get("q") is None


def test_get_avoid_profiles_returns_empty_when_not_failed():
    cache = RouterCache()
    cache.put("q", "multihop")
    assert cache.get_avoid_profiles("q") == []


def test_get_avoid_profiles_returns_failed_profile():
    cache = RouterCache()
    cache.put("q", "multihop")
    cache.mark_failed("q")
    cache.mark_failed("q")
    avoid = cache.get_avoid_profiles("q")
    assert "multihop" in avoid


def test_lru_eviction():
    cache = RouterCache(maxsize=2)
    cache.put("q1", "semantic")
    cache.put("q2", "multihop")
    cache.put("q3", "full")  # evicts q1
    assert cache.get("q1") is None
    assert cache.get("q2") is not None
    assert cache.get("q3") is not None


def test_prompt_hash_isolates_keys():
    c1 = RouterCache(prompt_hash="abc")
    c2 = RouterCache(prompt_hash="xyz")
    c1.put("q", "semantic")
    assert c2.get("q") is None
