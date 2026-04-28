import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "rag-anything"))

import pytest

from evaluate_local.MultiHopQA.build_index import (
    MULTIHOPQA_NEVER_SPLIT_DELIMITER,
    _TeeOutput,
    _validate_batch_doc_status,
    apply_multihopqa_index_profile,
    build_virtual_batches,
    prepare_source_records,
    resolve_safe_split_delimiter,
    resolve_log_file,
    validate_existing_manifest_for_resume,
    validate_or_write_index_profile,
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
    assert first["source_key"].startswith("hotpotqa:")
    assert first["title"] == "Title A"
    assert first["content"] == "Title A\nFirst paragraph."
    assert first["lightrag_chunk_id"].startswith("chunk-")
    assert chunk_source_map[first["lightrag_chunk_id"]]["source_paragraph_id"] == "hotpotqa_000001"
    assert chunk_source_map[first["lightrag_chunk_id"]]["source_key"] == first["source_key"]
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


class _DummySettings:
    enable_entity_disambiguation = False
    enable_synonym_linking = False
    enable_multi_hop = True
    multi_hop_depth = 2
    ppr_damping = 0.5
    ppr_top_k = 50
    ppr_qa_top_k = 5
    passage_node_weight = 0.05
    synonymy_threshold = 0.8
    synonymy_topk = 2048
    synonymy_min_entity_len = 2
    enable_entity_surface_normalization = False
    enable_keyword_case_normalization = False
    strict_relation_endpoint_entity_match = False


def test_apply_multihopqa_index_profile_matches_v0_v1_v2_build_settings():
    settings = _DummySettings()

    metadata = apply_multihopqa_index_profile(settings)

    assert settings.enable_entity_disambiguation is True
    assert settings.enable_synonym_linking is True
    assert settings.enable_multi_hop is False
    assert settings.enable_entity_surface_normalization is True
    assert settings.enable_keyword_case_normalization is True
    assert settings.strict_relation_endpoint_entity_match is True
    assert metadata["ablation_group"] == "DB+V1+V2"
    assert metadata["index_profile"]["enable_entity_disambiguation"] is True
    assert metadata["index_profile"]["enable_synonym_linking"] is True


def test_validate_batch_doc_status_rejects_failed_lightrag_doc_status():
    batch = {
        "batch_doc_id": "multihopqa_batch_000001",
        "expected_chunk_count": 2,
        "expected_chunk_ids": ["chunk-a", "chunk-b"],
    }

    with pytest.raises(RuntimeError, match="LightRAG marked .* failed"):
        _validate_batch_doc_status(
            {
                "status": "failed",
                "error_msg": "Chunk token length 1236 exceeds chunk_token_size 1200",
                "chunks_list": [],
            },
            batch,
        )


def test_validate_batch_doc_status_rejects_missing_expected_chunks():
    batch = {
        "batch_doc_id": "multihopqa_batch_000001",
        "expected_chunk_count": 2,
        "expected_chunk_ids": ["chunk-a", "chunk-b"],
    }

    with pytest.raises(RuntimeError, match="missing expected chunks"):
        _validate_batch_doc_status(
            {
                "status": "processed",
                "chunks_list": ["chunk-a"],
            },
            batch,
        )


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


def test_validate_existing_manifest_for_resume_rejects_mismatched_index_profile(tmp_path):
    manifest_path = tmp_path / "multihopqa_ingest_manifest.json"
    manifest_path.write_text(
        """{
  "workspace_id": "hotpotqa_500_seed42",
  "dataset": "hotpotqa",
  "n_samples": 500,
  "seed": 42,
  "index_profile": {
    "profile_version": 1,
    "enable_entity_disambiguation": true,
    "enable_synonym_linking": false
  }
}""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index_profile"):
        validate_existing_manifest_for_resume(
            manifest_path=manifest_path,
            workspace="hotpotqa_500_seed42",
            dataset="hotpotqa",
            n_samples=500,
            seed=42,
            expected_index_profile={
                "profile_version": 1,
                "enable_entity_disambiguation": True,
                "enable_synonym_linking": True,
            },
        )


def test_validate_or_write_index_profile_persists_profile(tmp_path):
    metadata = {
        "ablation_profile": "v0_v1_v2",
        "ablation_group": "DB+V1+V2",
        "ablation_flags": {"enable_synonym_linking": True},
        "index_profile": {
            "profile_version": 1,
            "enable_entity_disambiguation": True,
            "enable_synonym_linking": True,
        },
    }

    validate_or_write_index_profile(
        working_dir=tmp_path,
        index_profile_metadata=metadata,
    )

    payload = json.loads((tmp_path / "multihopqa_index_profile.json").read_text())
    assert payload["index_profile"] == metadata["index_profile"]
    assert payload["ablation_group"] == "DB+V1+V2"


def test_validate_or_write_index_profile_rejects_existing_artifacts_without_profile(
    tmp_path,
):
    (tmp_path / "kv_store_doc_status.json").write_text("{}", encoding="utf-8")
    metadata = {
        "ablation_profile": "v0_v1_v2",
        "ablation_group": "DB+V1+V2",
        "ablation_flags": {"enable_synonym_linking": True},
        "index_profile": {
            "profile_version": 1,
            "enable_entity_disambiguation": True,
            "enable_synonym_linking": True,
        },
    }

    with pytest.raises(ValueError, match="existing artifacts"):
        validate_or_write_index_profile(
            working_dir=tmp_path,
            index_profile_metadata=metadata,
        )


def test_validate_or_write_index_profile_rejects_mismatched_profile(tmp_path):
    (tmp_path / "multihopqa_index_profile.json").write_text(
        json.dumps(
            {
                "index_profile": {
                    "profile_version": 1,
                    "enable_entity_disambiguation": True,
                    "enable_synonym_linking": False,
                }
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "ablation_profile": "v0_v1_v2",
        "ablation_group": "DB+V1+V2",
        "ablation_flags": {"enable_synonym_linking": True},
        "index_profile": {
            "profile_version": 1,
            "enable_entity_disambiguation": True,
            "enable_synonym_linking": True,
        },
    }

    with pytest.raises(ValueError, match="index profile"):
        validate_or_write_index_profile(
            working_dir=tmp_path,
            index_profile_metadata=metadata,
        )


def test_resolve_log_file_defaults_inside_working_dir(tmp_path):
    assert resolve_log_file(tmp_path, None) == tmp_path / "multihopqa_build_index.log"


def test_resolve_log_file_accepts_explicit_path(tmp_path):
    explicit = tmp_path / "logs" / "hotpotqa.log"
    assert resolve_log_file(tmp_path, str(explicit)) == explicit


def test_tee_output_writes_prints_to_log_file(tmp_path):
    log_file = tmp_path / "build.log"

    with _TeeOutput(log_file):
        print("hello build log")

    assert "hello build log" in log_file.read_text(encoding="utf-8")
