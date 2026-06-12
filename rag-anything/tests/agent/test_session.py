# tests/agent/test_session.py
from raganything.agent.session import SessionMemory, SessionStore


def test_active_entities_cap_evicts_oldest():
    s = SessionMemory(session_id="s1", workspace_id="w1")
    for i in range(14):
        s.register_entities([{"name": f"e{i}", "note": "", "last_turn": i}])
    assert len(s.active_entities) == 12
    assert all(e["name"] != "e0" for e in s.active_entities)


def test_chunk_cache_lru():
    s = SessionMemory(session_id="s1", workspace_id="w1", cache_max=2)
    s.cache_chunks([{"chunk_id": "c1", "content": "a"}, {"chunk_id": "c2", "content": "b"}])
    s.get_cached(["c1"])  # touch c1
    s.cache_chunks([{"chunk_id": "c3", "content": "c"}])
    assert set(s.chunk_cache) == {"c1", "c3"}


def test_store_get_create_and_ttl(monkeypatch):
    store = SessionStore(ttl_seconds=100, max_sessions=2)
    a = store.get("w1", "s1")
    assert store.get("w1", "s1") is a
    now = [1000.0]
    monkeypatch.setattr("raganything.agent.session._now", lambda: now[0])
    a.touch()
    now[0] += 200
    store.sweep()
    assert store.get("w1", "s1") is not a  # 过期重建


def test_drop_chunks_by_workspace():
    store = SessionStore()
    s = store.get("w1", "s1")
    s.cache_chunks([{"chunk_id": "c1", "content": "x"}])
    other = store.get("w2", "s2")
    other.cache_chunks([{"chunk_id": "c1", "content": "x"}])
    store.drop_chunks("w1", ["c1"])  # 治理删除联动 §5.6
    assert "c1" not in s.chunk_cache and "c1" in other.chunk_cache


def test_dump_load_roundtrip():
    s = SessionMemory(session_id="s1", workspace_id="w1")
    s.history_summary = "摘要"
    s.recent_turns.append({"q": "a", "a": "b", "cancelled": False})
    data = s.dump()
    s2 = SessionMemory.load(data)
    assert s2.history_summary == "摘要" and len(s2.recent_turns) == 1
