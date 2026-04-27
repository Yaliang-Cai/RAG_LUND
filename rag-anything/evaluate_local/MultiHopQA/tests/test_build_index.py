import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "rag-anything"))

import pytest

from evaluate_local.MultiHopQA.build_index import (
    MULTIHOPQA_NEVER_SPLIT_DELIMITER,
    build_virtual_batches,
    prepare_source_records,
    resolve_safe_split_delimiter,
    validate_existing_manifest_for_resume,
)


def test_prepare_source_records_builds_stable_chunk_source_map():
    corpus = [
        {"title": "Title A", "text": "First paragraph."},
        {"title": "Title B", "text": "Second paragraph."},
    ]

    source_records, chunk_source_map, stats = prepare_source_records(
        dataset="hotpotqa",
        corpus=corpus,
    )

    assert list(source_records) == ["hotpotqa_000001", "hotpotqa_000002"]
    first = source_records["hotpotqa_000001"]
    assert first["source_paragraph_id"] == "hotpotqa_000001"
    assert first["title"] == "Title A"
    assert first["content"] == "Title A\nFirst paragraph."
    assert first["lightrag_chunk_id"].startswith("chunk-")
    assert chunk_source_map[first["lightrag_chunk_id"]]["source_paragraph_id"] == "hotpotqa_000001"
    assert chunk_source_map[first["lightrag_chunk_id"]]["title"] == "Title A"
    assert stats == {
        "source_paragraph_count": 2,
        "source_chunk_count": 2,
        "empty_paragraph_count": 0,
    }


def test_prepare_source_records_rejects_duplicate_lightrag_chunk_ids():
    corpus = [
        {"title": "Same", "text": "Same text."},
        {"title": "Same", "text": "Same text."},
    ]

    with pytest.raises(ValueError, match="duplicate LightRAG chunk id"):
        prepare_source_records(dataset="hotpotqa", corpus=corpus)


def test_resolve_safe_split_delimiter_uses_fallback_when_needed():
    texts = [f"alpha {MULTIHOPQA_NEVER_SPLIT_DELIMITER} beta"]

    delimiter = resolve_safe_split_delimiter(texts)

    assert delimiter != MULTIHOPQA_NEVER_SPLIT_DELIMITER
    assert delimiter.startswith(MULTIHOPQA_NEVER_SPLIT_DELIMITER + "_")
    assert all(delimiter not in text for text in texts)


def test_build_virtual_batches_preserves_expected_chunk_mapping():
    corpus = [
        {"title": "T1", "text": "A"},
        {"title": "T2", "text": "B"},
        {"title": "T3", "text": "C"},
    ]
    source_records, _, _ = prepare_source_records(dataset="2wiki", corpus=corpus)

    batches = build_virtual_batches(
        source_records=source_records,
        ingest_batch_size=2,
    )

    assert [b["batch_doc_id"] for b in batches] == [
        "multihopqa_batch_000001",
        "multihopqa_batch_000002",
    ]
    assert batches[0]["source_paragraph_ids"] == ["2wiki_000001", "2wiki_000002"]
    assert batches[0]["expected_chunk_ids"] == [
        source_records["2wiki_000001"]["lightrag_chunk_id"],
        source_records["2wiki_000002"]["lightrag_chunk_id"],
    ]
    assert batches[0]["expected_chunk_count"] == 2
    assert batches[0]["content"] == batches[0]["delimiter"].join(
        [
            source_records["2wiki_000001"]["content"],
            source_records["2wiki_000002"]["content"],
        ]
    )
    assert batches[1]["source_paragraph_ids"] == ["2wiki_000003"]


def test_validate_existing_manifest_for_resume_rejects_mismatched_seed(tmp_path):
    manifest_path = tmp_path / "multihopqa_ingest_manifest.json"
    manifest_path.write_text(
        """{
  "workspace_id": "hotpotqa_500_seed42",
  "dataset": "hotpotqa",
  "n_samples": 500,
  "seed": 13
}""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seed"):
        validate_existing_manifest_for_resume(
            manifest_path=manifest_path,
            workspace="hotpotqa_500_seed42",
            dataset="hotpotqa",
            n_samples=500,
            seed=42,
        )


def test_validate_existing_manifest_for_resume_accepts_matching_manifest(tmp_path):
    manifest_path = tmp_path / "multihopqa_ingest_manifest.json"
    manifest_path.write_text(
        """{
  "workspace_id": "hotpotqa_500_seed42",
  "dataset": "hotpotqa",
  "n_samples": 500,
  "seed": 42
}""",
        encoding="utf-8",
    )

    validate_existing_manifest_for_resume(
        manifest_path=manifest_path,
        workspace="hotpotqa_500_seed42",
        dataset="hotpotqa",
        n_samples=500,
        seed=42,
    )
