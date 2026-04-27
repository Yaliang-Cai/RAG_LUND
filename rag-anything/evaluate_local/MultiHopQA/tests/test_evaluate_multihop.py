import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "rag-anything"))

from evaluate_local.MultiHopQA.evaluate_multihop import _resolve_retrieved_sources


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
