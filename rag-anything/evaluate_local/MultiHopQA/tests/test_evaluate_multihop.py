import sys
import hashlib
import asyncio
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "rag-anything"))

from evaluate_local.MultiHopQA.evaluate_multihop import (
    _build_query_kwargs,
    _load_chunk_source_map,
    _parse_args,
    _resolve_log_file,
    _resolve_retrieved_sources,
    _run_mode,
    _score_support_recall,
)


def test_resolve_retrieved_sources_uses_chunk_id_map():
    source_map = {
        "chunk-abc": {
            "source_paragraph_id": "hotpotqa_000001",
            "title": "Article A",
        }
    }
    chunks = [{"id": "chunk-abc", "content": "Article A\nText"}]

    sources = _resolve_retrieved_sources(chunks, source_map)

    assert sources == [
        {
            "rank": 1,
            "chunk_id": "chunk-abc",
            "source_paragraph_id": "hotpotqa_000001",
            "source_key": None,
            "title": "Article A",
        }
    ]


def test_resolve_retrieved_sources_prefers_lightrag_chunk_id_over_display_id():
    source_map = {
        "chunk-abc": {
            "source_paragraph_id": "hotpotqa_000001",
            "title": "Article A",
        }
    }
    chunks = [{"id": "DC1", "chunk_id": "chunk-abc", "content": "Article A\nText"}]

    sources = _resolve_retrieved_sources(chunks, source_map)

    assert sources == [
        {
            "rank": 1,
            "chunk_id": "chunk-abc",
            "source_paragraph_id": "hotpotqa_000001",
            "source_key": None,
            "title": "Article A",
        }
    ]


def test_resolve_retrieved_sources_falls_back_to_content_hash():
    chunk_id = "chunk-" + hashlib.md5("Article C\nText".encode("utf-8")).hexdigest()
    source_map = {
        chunk_id: {
            "source_paragraph_id": "2wiki_000003",
            "title": "Article C",
        }
    }
    chunks = [{"content": "Article C\nText"}]

    sources = _resolve_retrieved_sources(chunks, source_map)

    assert sources[0]["chunk_id"] == chunk_id
    assert sources[0]["source_paragraph_id"] == "2wiki_000003"


def test_score_support_recall_prefers_source_keys_over_text_fingerprint():
    chunks = [
        {"id": "chunk-a", "content": "Maurice Elvey( 11 November 1887 was a director."},
        {"id": "chunk-b", "content": "Other content."},
    ]
    source_map = {
        "chunk-a": {
            "source_paragraph_id": "2wiki_000001",
            "source_key": "source-a",
            "title": "Maurice Elvey",
        }
    }
    item = {
        "gold_source_keys": ["source-a"],
        "supporting_facts": ["Maurice Elvey (11 November 1887 was a director."],
    }

    assert _score_support_recall(
        chunks=chunks,
        item=item,
        k=1,
        chunk_source_map=source_map,
        fallback_score_recall_at_k=lambda _chunks, _facts, _k: 0.0,
    ) == 1.0


def test_build_query_kwargs_pins_eval_retrieval_controls():
    kwargs = _build_query_kwargs(
        query_overrides={"response_type": "Short Answer"},
        wire_profile=None,
        top_k=10,
        chunk_top_k=5,
        max_total_tokens=45000,
    )

    assert kwargs["response_type"] == "Short Answer"
    assert kwargs["top_k"] == 10
    assert kwargs["chunk_top_k"] == 5
    assert kwargs["max_total_tokens"] == 45000


def test_parse_args_defaults_match_shared_retrieval_ablation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_multihop.py",
            "--dataset",
            "2wiki",
            "--workspace",
            "2wiki_500_seed42_0428",
            "--working-dir",
            "/tmp/2wiki_500_seed42_0428",
            "--output-dir",
            "/tmp/out",
        ],
    )

    args = _parse_args()

    assert args.concurrency == 16
    assert args.top_k == 10
    assert args.chunk_top_k == 5
    assert args.ppr_qa_top_k == 5
    assert args.enable_kg_rerank is False
    assert args.hybrid_enable_rerank is True
    assert args.ppr_enable_rerank is False
    assert args.bypass_query_cache is True
    assert args.bypass_keywords_cache is False
    assert args.vlm_enhanced is False
    assert args.log_file is None


def test_resolve_log_file_defaults_to_dataset_log_in_output_dir(tmp_path):
    absolute_log = tmp_path / "custom.log"
    assert _resolve_log_file(tmp_path, None, "2wiki") == tmp_path / "2wiki_evaluate_multihop.log"
    assert _resolve_log_file(tmp_path, "logs/custom.log", "2wiki") == tmp_path / "logs/custom.log"
    assert _resolve_log_file(tmp_path, str(absolute_log), "2wiki") == absolute_log


def test_load_chunk_source_map_rejects_identity_mismatch(tmp_path):
    payload = {
        "workspace_id": "hotpotqa_500_seed42_0428",
        "dataset": "hotpotqa",
        "n_samples": 500,
        "seed": 42,
        "map_size": 1,
        "map": {"chunk-a": {"source_key": "source-a"}},
    }
    (tmp_path / "multihopqa_chunk_source_map.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset"):
        _load_chunk_source_map(
            tmp_path,
            dataset="2wiki",
            workspace="hotpotqa_500_seed42_0428",
            n_samples=500,
            seed=42,
            strict=True,
        )


def test_run_mode_uses_bounded_concurrency_and_preserves_jsonl_order(tmp_path):
    class FakeService:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.calls = []

        async def query_with_trace(self, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            self.calls.append(kwargs)
            return {
                "answer": "gold",
                "trace": {"data": {"chunks": [{"content": "support paragraph"}]}},
            }

    items = [
        {
            "id": f"q{i}",
            "question": f"question {i}",
            "answer": "gold",
            "supporting_facts": ["support paragraph"],
        }
        for i in range(4)
    ]
    service = FakeService()

    metrics = asyncio.run(
        _run_mode(
            service=service,
            workspace_id="2wiki_500_seed42_0428",
            working_dir="/tmp/ws",
            items=items,
            mode="hybrid",
            dataset="2wiki",
            recall_ks=[1],
            output_dir=tmp_path,
            resume=False,
            score_em=lambda pred, gold: 1.0 if pred == gold else 0.0,
            score_f1=lambda pred, gold: 1.0 if pred == gold else 0.0,
            score_recall_at_k=lambda chunks, facts, k: 1.0,
            get_eval_query_overrides=lambda dataset: {"response_type": "Short Answer"},
            chunk_source_map={},
            query_kwargs={"top_k": 10, "chunk_top_k": 5, "max_total_tokens": 45000},
            concurrency=2,
        )
    )

    assert metrics["n"] == 4
    assert service.max_active == 2
    assert all(call["top_k"] == 10 for call in service.calls)
    assert all(call["chunk_top_k"] == 5 for call in service.calls)
    assert all(call["max_total_tokens"] == 45000 for call in service.calls)

    rows = [
        json.loads(line)
        for line in (tmp_path / "2wiki_hybrid_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["q0", "q1", "q2", "q3"]


def test_run_mode_applies_mode_specific_rerank_and_cache_defaults(tmp_path):
    class FakeService:
        def __init__(self):
            self.calls = []

        async def query_with_trace(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "answer": "gold",
                "trace": {"data": {"chunks": [{"content": "support paragraph"}]}},
            }

    async def run_once(mode: str, output_dir: Path) -> dict:
        output_dir.mkdir()
        service = FakeService()
        await _run_mode(
            service=service,
            workspace_id="2wiki_500_seed42_0428",
            working_dir="/tmp/ws",
            items=[
                {
                    "id": f"q-{mode}",
                    "question": "question",
                    "answer": "gold",
                    "supporting_facts": ["support paragraph"],
                }
            ],
            mode=mode,
            dataset="2wiki",
            recall_ks=[1],
            output_dir=output_dir,
            resume=False,
            score_em=lambda pred, gold: 1.0 if pred == gold else 0.0,
            score_f1=lambda pred, gold: 1.0 if pred == gold else 0.0,
            score_recall_at_k=lambda chunks, facts, k: 1.0,
            get_eval_query_overrides=lambda dataset: {"response_type": "Short Answer"},
            chunk_source_map={},
            query_kwargs={
                "top_k": 10,
                "chunk_top_k": 5,
                "max_total_tokens": 45000,
                "ppr_qa_top_k": 5,
                "enable_kg_rerank": False,
                "ppr_post_rerank_fusion": "none",
                "ppr_post_rerank_rrf_k": 60,
                "bypass_query_cache": True,
                "bypass_keywords_cache": False,
                "vlm_enhanced": False,
            },
            concurrency=1,
        )
        return service.calls[0]

    hybrid_call = asyncio.run(run_once("hybrid", tmp_path / "hybrid"))
    ppr_call = asyncio.run(run_once("ppr", tmp_path / "ppr"))

    assert hybrid_call["enable_rerank"] is True
    assert ppr_call["enable_rerank"] is False
    assert hybrid_call["enable_kg_rerank"] is False
    assert ppr_call["enable_kg_rerank"] is False
    assert ppr_call["ppr_qa_top_k"] == 5
    assert ppr_call["ppr_post_rerank_fusion"] == "none"
    assert ppr_call["bypass_query_cache"] is True
    assert ppr_call["bypass_keywords_cache"] is False
    assert ppr_call["vlm_enhanced"] is False
