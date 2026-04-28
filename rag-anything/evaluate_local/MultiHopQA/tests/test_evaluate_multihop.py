import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "rag-anything"))

from evaluate_local.MultiHopQA.evaluate_multihop import (
    _resolve_retrieved_sources,
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
