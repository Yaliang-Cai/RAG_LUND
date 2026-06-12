from raganything.agent.evidence import FactLedger


PAYLOAD = {"facts": [
    {"id": "f1", "text": "A 是 B", "status": "found", "chunks": ["c1"]},
    {"id": "f2", "text": "C 的数值", "status": "missing", "chunks": []},
    {"id": "f3", "text": "D 的来源", "status": "missing", "chunks": []},
]}


def test_update_and_effective_coverage():
    led = FactLedger()
    led.update(PAYLOAD)
    assert led.coverage == 1 / 3
    assert [f["id"] for f in led.missing()] == ["f2", "f3"]


def test_unverifiable_after_two_distinct_tools():
    led = FactLedger()
    led.update(PAYLOAD)
    led.record_attempt("f2", "search_dense")
    assert led.facts["f2"]["status"] == "missing"
    led.record_attempt("f2", "search_dense")  # 同工具重复不计第二次
    assert led.facts["f2"]["status"] == "missing"
    led.record_attempt("f2", "search_hybrid")  # 第二个不同工具 → 放弃 §5.3
    assert led.facts["f2"]["status"] == "unverifiable"
    assert led.coverage == 1 / 2  # 分母剔除 unverifiable


def test_found_marks_supports_back():
    led = FactLedger()
    led.update(PAYLOAD)
    assert led.supported_chunks() == {"c1": {"f1"}}


def test_update_merge_keeps_unverifiable():
    led = FactLedger()
    led.update(PAYLOAD)
    led.record_attempt("f2", "a"); led.record_attempt("f2", "b")
    led.update({"facts": [{"id": "f2", "text": "C 的数值", "status": "missing", "chunks": []}]})
    assert led.facts["f2"]["status"] == "unverifiable"  # grader 不能复活已放弃事实


def test_synthetic_id_no_collision_with_grader_ids():
    led = FactLedger()
    led.update({"facts": [
        {"id": "f1", "text": "a", "status": "found", "chunks": []},
        {"id": "f3", "text": "b", "status": "missing", "chunks": []},
        {"text": "无 id 的事实", "status": "missing", "chunks": []},  # 不得覆盖 f3
    ]})
    assert len(led.facts) == 3
    assert led.facts["f3"]["text"] == "b"
    # 同文本无 id 事实再次出现 → 同一条目，不重复
    led.update({"facts": [{"text": "无 id 的事实", "status": "found", "chunks": ["c9"]}]})
    assert len(led.facts) == 3
