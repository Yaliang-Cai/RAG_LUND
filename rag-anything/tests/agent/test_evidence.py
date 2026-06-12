from raganything.agent.evidence import EvidencePool, PoolEntry


def chunk(cid, content="text", score=0.5, file_path="a.md"):
    return {"chunk_id": cid, "content": content, "rrf_score": score, "file_path": file_path}


def test_dedup_appends_provenance_and_hit_count():
    pool = EvidencePool()
    new = pool.add([chunk("c1")], step=0, tool="search_dense", sub_query="q")
    assert len(new) == 1
    new2 = pool.add([chunk("c1"), chunk("c2")], step=1, tool="search_hybrid", sub_query="q2")
    assert [e.chunk_id for e in new2] == ["c2"]  # 仅新增需 rerank §5.1
    e1 = pool.entries["c1"]
    assert e1.hit_count == 2 and len(e1.provenance) == 2
    assert pool.last_dup_rate == 0.5  # 2 进 1 重


def test_synthetic_id_is_content_hash():
    pool = EvidencePool()
    a = pool.add([{"content": "same text", "rrf_score": 0.1}], step=0, tool="t", sub_query="q")
    b = pool.add([{"content": "same text", "rrf_score": 0.2}], step=1, tool="t", sub_query="q")
    assert len(a) == 1 and len(b) == 0  # 同内容不同批次判同条 §5.2


def test_image_paths_parsed_on_admission():
    pool = EvidencePool()
    pool.add([chunk("c1", content="说明\nImage Path: img/fig1.jpg\n后文")], step=0, tool="t", sub_query="q")
    assert pool.entries["c1"].image_paths == ["img/fig1.jpg"]


def test_eviction_protects_fact_supporters():
    pool = EvidencePool(max_entries=2)
    pool.add([chunk("c1"), chunk("c2"), chunk("c3")], step=0, tool="t", sub_query="q")
    pool.set_scores({"c1": 0.1, "c2": 0.9, "c3": 0.5})
    pool.entries["c1"].supports.add("f1")  # 低分但支撑事实 → 豁免 §5.5
    pool.evict()
    assert set(pool.entries) == {"c1", "c2"}


def test_top_sorted_by_canonical_then_hits():
    pool = EvidencePool()
    pool.add([chunk("c1"), chunk("c2")], step=0, tool="t", sub_query="q")
    pool.set_scores({"c1": 0.3, "c2": 0.8})
    assert [e.chunk_id for e in pool.top(2)] == ["c2", "c1"]
