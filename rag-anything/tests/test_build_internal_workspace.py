from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_internal_workspace as build_internal
from raganything.processor import ProcessorMixin


class _FakeEnum(Enum):
    VALUE = "value"


class _DummyProcessor(ProcessorMixin):
    def __init__(self) -> None:
        self.config = SimpleNamespace(parser="mineru")
        self.logger = logging.getLogger("dummy_processor")

    def _resolve_mineru_method(self, parse_method, **kwargs):
        return parse_method or "auto"


class _FakeDocStatus:
    def __init__(self) -> None:
        self.docs = {
            "doc-1": {
                "status": "processed",
                "file_path": "/raw/a.pdf",
                "chunks_count": 2,
                "chunks_list": ["chunk-1", "chunk-mm-1"],
                "multimodal_processed": False,
                "multimodal_stage": "chunks_stored",
                "multimodal_failed_items": [{"index": 0, "error": "timeout"}],
                "multimodal_chunk_ids": ["chunk-mm-1"],
                "created_at": "2026-05-14T00:00:00+00:00",
                "updated_at": "2026-05-14T00:01:00+00:00",
                "metadata": {
                    "enum_value": _FakeEnum.VALUE,
                    "enum_class": _FakeEnum,
                },
            },
            "doc-2": {
                "status": "processed",
                "file_path": "/raw/b.pdf",
                "chunks_count": 1,
                "chunks_list": ["chunk-2"],
                "multimodal_processed": True,
                "multimodal_stage": "completed",
                "multimodal_failed_items": [],
                "multimodal_chunk_ids": [],
            },
        }

    async def get_docs_paginated(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
        status_filter=None,
    ):
        rows = [(doc_id, payload) for doc_id, payload in sorted(self.docs.items())]
        start = (page - 1) * page_size
        return rows[start : start + page_size], len(rows)

    async def get_by_id(self, doc_id: str):
        return self.docs.get(doc_id)


class _FakeKV:
    def __init__(self, data=None) -> None:
        self.data = data or {}

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(key) for key in keys]


class _FakeVector:
    async def get_by_ids(self, keys):
        return [{"id": key} for key in keys if key]


class _FakeGraph:
    async def get_all_labels(self):
        return ["EntityA"]

    async def get_nodes_batch(self, labels):
        return {
            "EntityA": {
                "entity_id": "EntityA",
                "entity_name": "EntityA",
                "entity_type": "TECH",
                "description": "desc",
                "source_id": "chunk-1",
                "file_path": "/raw/a.pdf",
            }
        }

    async def get_all_edges(self):
        return [
            {
                "source": "EntityA",
                "target": "EntityB",
                "description": "related",
                "keywords": "k",
                "weight": 1.0,
                "source_id": "chunk-1",
                "file_path": "/raw/a.pdf",
            }
        ]


class _FakeRagWrapper:
    def __init__(self) -> None:
        self.lightrag = SimpleNamespace(
            doc_status=_FakeDocStatus(),
            full_docs=_FakeKV({"doc-1": {"content": "a"}}),
            full_entities=_FakeKV({"doc-1": {"entity_names": ["EntityA"]}}),
            full_relations=_FakeKV({"doc-1": {"relation_pairs": [["EntityA", "EntityB"]]}}),
            text_chunks=_FakeKV({"chunk-1": {"content": "text"}, "chunk-mm-1": {"content": "image"}}),
            chunks_vdb=_FakeVector(),
            chunk_entity_relation_graph=_FakeGraph(),
        )

    async def _ensure_lightrag_initialized(self):
        return None


def _stub_sentence_transformers(monkeypatch):
    stub = types.ModuleType("sentence_transformers")

    class _DummyCrossEncoder:
        pass

    class _DummySentenceTransformer:
        pass

    stub.CrossEncoder = _DummyCrossEncoder
    stub.SentenceTransformer = _DummySentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)


class _FakeService:
    def __init__(self, settings=None, fail_names=None) -> None:
        self.settings = settings or SimpleNamespace(enable_synonym_linking=True)
        self.fail_names = set(fail_names or [])
        self.ingest_calls = []
        self.synonym_calls = []
        self.delete_calls = []
        self.rag = _FakeRagWrapper()

    async def ingest(self, *, file_path, output_dir, workspace_id, serialize_by_workspace_id):
        file_name = Path(file_path).name
        self.ingest_calls.append(
            {
                "file_path": file_path,
                "output_dir": output_dir,
                "workspace_id": workspace_id,
                "serialize_by_workspace_id": serialize_by_workspace_id,
            }
        )
        if file_name in self.fail_names:
            raise RuntimeError(f"forced ingest failure: {file_name}")
        return file_path

    async def finalize_workspace_synonyms(self, workspace_id, *, force=False, reset_existing=True):
        self.synonym_calls.append((workspace_id, force, reset_existing))
        return {"success": True, "cleared_edges": 1, "created_edges": 2}

    async def get_rag(self, workspace_id):
        return self.rag

    async def lightrag_adelete_by_doc_id(self, workspace_id, doc_id, *, delete_llm_cache=False):
        self.delete_calls.append((workspace_id, doc_id, delete_llm_cache))
        return SimpleNamespace(
            status="success",
            doc_id=doc_id,
            message="deleted",
            status_code=200,
            file_path="/raw/a.pdf",
        )

    async def cleanup_workspace_instance(self, workspace_id):
        return None


def _patch_fake_service(monkeypatch, *, fail_names=None):
    fake = _FakeService(fail_names=fail_names)
    monkeypatch.setattr(
        build_internal,
        "preparse_mineru_files",
        lambda profile, files, report_dir: {
            "enabled": True,
            "output_dir": str(profile.output_dir / profile.workspace_id),
            "candidate_count": len(files),
            "skipped_count": len(files),
            "parsed_count": 0,
            "pending_count": 0,
        },
    )
    monkeypatch.setattr(
        build_internal,
        "build_local_settings",
        lambda profile: SimpleNamespace(
            working_dir_root=str(profile.working_dir_root),
            output_dir=str(profile.output_dir),
            log_dir=str(profile.log_dir),
            uploads_dir=str(profile.uploads_dir),
            enable_entity_disambiguation=False,
            enable_synonym_linking=True,
            enable_resilience=True,
            enable_entity_surface_normalization=True,
            enable_keyword_case_normalization=True,
            strict_relation_endpoint_entity_match=True,
        ),
    )
    monkeypatch.setattr(build_internal, "create_local_rag_service", lambda settings: fake)
    monkeypatch.setattr(
        build_internal,
        "ensure_workspace_index_profile",
        lambda **kwargs: dict(kwargs["index_profile"]),
    )
    return fake


def test_test_profile_resolves_internal_test_paths():
    profile = build_internal.resolve_profile("test")

    assert profile.raw_dir == Path("/data/y50056788/Yaliang/datasets_raw_test")
    assert profile.storage_root == Path("/data/y50056788/Yaliang/internal_test")
    assert profile.workspace_id == "internal_test"
    assert profile.output_dir == profile.storage_root / "output"
    assert profile.working_dir_root == profile.storage_root / "rag_workspace"
    assert profile.log_dir == profile.storage_root / "logs"


def test_prod_profile_resolves_internal_paths():
    profile = build_internal.resolve_profile("prod")

    assert profile.raw_dir == Path("/data/y50056788/Yaliang/datasets_raw")
    assert profile.storage_root == Path("/data/y50056788/Yaliang/internal")
    assert profile.workspace_id == "internal"


def test_ran2_133_bis_profile_resolves_internal_paths():
    profile = build_internal.resolve_profile("ran2_133_bis")

    assert profile.raw_dir == Path(
        "/data/y50056788/Yaliang/datasets_raw_RAN2_133_BIS"
    )
    assert profile.storage_root == Path(
        "/data/y50056788/Yaliang/internal_RAN2_133_BIS"
    )
    assert profile.workspace_id == "internal_RAN2_133_BIS"
    assert profile.output_dir == profile.storage_root / "output"
    assert profile.working_dir_root == profile.storage_root / "rag_workspace"
    assert profile.log_dir == profile.storage_root / "logs"
    assert profile.reports_dir == profile.storage_root / "reports"

    env = build_internal.build_local_env(profile, base_env={})

    assert env["RAGANYTHING_WORKDIR_ROOT"] == (
        "/data/y50056788/Yaliang/internal_RAN2_133_BIS/rag_workspace"
    )
    assert env["RAGANYTHING_OUTPUT_DIR"] == (
        "/data/y50056788/Yaliang/internal_RAN2_133_BIS/output"
    )
    assert env["RAGANYTHING_UPLOADS_DIR"] == (
        "/data/y50056788/Yaliang/internal_RAN2_133_BIS/uploads"
    )
    assert env["RAGANYTHING_LOG_DIR"] == (
        "/data/y50056788/Yaliang/internal_RAN2_133_BIS/logs"
    )


def test_sa_175_profile_resolves_internal_paths():
    profile = build_internal.resolve_profile("sa_175")

    assert profile.raw_dir == Path("/data/y50056788/Yaliang/datasets_raw_SA_175")
    assert profile.storage_root == Path("/data/y50056788/Yaliang/internal_SA_175")
    assert profile.workspace_id == "internal_SA_175"
    assert profile.output_dir == profile.storage_root / "output"
    assert profile.working_dir_root == profile.storage_root / "rag_workspace"
    assert profile.log_dir == profile.storage_root / "logs"
    assert profile.reports_dir == profile.storage_root / "reports"

    env = build_internal.build_local_env(profile, base_env={})

    assert env["RAGANYTHING_WORKDIR_ROOT"] == (
        "/data/y50056788/Yaliang/internal_SA_175/rag_workspace"
    )
    assert env["RAGANYTHING_OUTPUT_DIR"] == (
        "/data/y50056788/Yaliang/internal_SA_175/output"
    )
    assert env["RAGANYTHING_UPLOADS_DIR"] == (
        "/data/y50056788/Yaliang/internal_SA_175/uploads"
    )
    assert env["RAGANYTHING_LOG_DIR"] == (
        "/data/y50056788/Yaliang/internal_SA_175/logs"
    )


def test_local_env_enables_internal_build_defaults():
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=Path("/tmp/raw"),
        storage_root=Path("/tmp/internal"),
        workspace_id="demo_ws",
    )

    env = build_internal.build_local_env(profile, base_env={"EXISTING": "1"})

    assert env["EXISTING"] == "1"
    assert env["RAGANYTHING_WORKDIR_ROOT"] == "/tmp/internal/rag_workspace"
    assert env["RAGANYTHING_OUTPUT_DIR"] == "/tmp/internal/output"
    assert env["RAGANYTHING_LOG_DIR"] == "/tmp/internal/logs"
    assert env["RAGANYTHING_ENABLE_ENTITY_DISAMBIGUATION"] == "false"
    assert env["RAGANYTHING_ENABLE_SYNONYM_LINKING"] == "true"
    assert env["RAGANYTHING_ENABLE_RESILIENCE"] == "true"
    assert env["RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION"] == "true"
    assert env["RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION"] == "true"
    assert env["RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH"] == "true"
    assert env["ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE"] == "true"
    assert env["CONTEXT_ZERO_WINDOW_CONTENT_TYPES"]
    assert env["RAGANYTHING_SERIALIZE_MINERU"] == "true"
    assert env["MINERU_VLLM_GPU_MEMORY_UTILIZATION"] == "0.1"
    assert env["LIBREOFFICE_CONVERT_TIMEOUT_SECONDS"] == "900"
    assert env["RAGANYTHING_PRELOAD_RERANKER_MODEL"] == "false"
    assert env["RAGANYTHING_PRESERVE_EXISTING_LOGGING"] == "true"
    assert env["RAGANYTHING_DISABLE_LOCAL_RUN_LOG"] == "true"
    assert env["MAX_SOURCE_IDS_PER_ENTITY"] == "99999"
    assert env["MAX_SOURCE_IDS_PER_RELATION"] == "99999"
    assert env["MAX_CONCURRENT_FILES"] == "4"
    assert env["RAGANYTHING_LLM_CONTEXT_MAX_TOKENS"] == "65536"
    assert env["RAGANYTHING_LLM_CONTEXT_RESERVED_TOKENS"] == "512"
    assert env["RAGANYTHING_MULTIMODAL_ITEM_PARALLELISM"] == "3"


def test_local_env_allows_max_async_ingest_override():
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=Path("/tmp/raw"),
        storage_root=Path("/tmp/internal"),
        workspace_id="demo_ws",
    )

    env = build_internal.build_local_env(
        profile,
        base_env={},
        max_async_ingest=1,
    )

    assert env["MAX_CONCURRENT_FILES"] == "1"


def test_local_env_preserves_explicit_libreoffice_timeout():
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=Path("/tmp/raw"),
        storage_root=Path("/tmp/internal"),
        workspace_id="demo_ws",
    )

    env = build_internal.build_local_env(
        profile,
        base_env={"LIBREOFFICE_CONVERT_TIMEOUT_SECONDS": "1200"},
    )

    assert env["LIBREOFFICE_CONVERT_TIMEOUT_SECONDS"] == "1200"


def test_local_rag_settings_reads_preload_reranker_env(monkeypatch):
    _stub_sentence_transformers(monkeypatch)
    from raganything.services.local_rag import LocalRagSettings

    monkeypatch.setenv("RAGANYTHING_PRELOAD_RERANKER_MODEL", "false")

    settings = LocalRagSettings.from_env()

    assert settings.preload_reranker_model is False


def test_local_rag_settings_reads_llm_context_and_multimodal_parallelism_env(
    monkeypatch,
):
    _stub_sentence_transformers(monkeypatch)
    from raganything.services.local_rag import LocalRagSettings

    monkeypatch.setenv("RAGANYTHING_LLM_CONTEXT_MAX_TOKENS", "32768")
    monkeypatch.setenv("RAGANYTHING_LLM_CONTEXT_RESERVED_TOKENS", "256")
    monkeypatch.setenv("RAGANYTHING_MULTIMODAL_ITEM_PARALLELISM", "3")

    settings = LocalRagSettings.from_env()

    assert settings.llm_context_max_tokens == 32768
    assert settings.llm_context_reserved_tokens == 256
    assert settings.multimodal_item_parallelism == 3


def test_multimodal_guardrails_can_override_item_parallelism():
    processor = _DummyProcessor()
    processor.lightrag = SimpleNamespace(
        addon_params={"multimodal_item_parallelism": "3"},
        max_parallel_insert=4,
    )

    guardrails = processor._resolve_multimodal_batch_guardrails(total_items=10)

    assert guardrails["parallelism"] == 3


def test_local_rag_preserves_internal_build_logging_without_run_log(
    monkeypatch, tmp_path
):
    _stub_sentence_transformers(monkeypatch)
    from raganything.services.local_rag import LocalRagSettings, configure_logging

    monkeypatch.setenv("RAGANYTHING_PRESERVE_EXISTING_LOGGING", "true")
    settings = LocalRagSettings(log_dir=str(tmp_path / "logs"))

    logger = configure_logging(settings)

    assert logger.name == "raganything.services.local_rag"
    assert not list((tmp_path / "logs").glob("run_*.log"))


def test_rerank_func_lazy_loads_when_model_not_preloaded(monkeypatch):
    _stub_sentence_transformers(monkeypatch)
    from raganything.services.local_rag import build_rerank_func

    class _RecordingReranker:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, pairs, batch_size=32):
            self.calls.append({"pairs": len(pairs), "batch_size": int(batch_size)})
            return [0.1, 0.7]

    reranker = _RecordingReranker()
    load_calls = []
    settings = SimpleNamespace(
        rerank_batch_size=32,
        rerank_enable_oom_backoff=True,
        rerank_min_batch_size=4,
    )

    def _load_model(load_settings):
        load_calls.append(load_settings)
        return reranker

    rerank_func = build_rerank_func(
        settings,
        None,
        logging.getLogger(__name__),
        model_loader=_load_model,
    )

    results = asyncio.run(rerank_func("query", ["a", "b"], top_n=1))

    assert load_calls == [settings]
    assert reranker.calls == [{"pairs": 2, "batch_size": 32}]
    assert results == [{"index": 1, "relevance_score": 0.7}]


def test_supported_files_are_top_level_only(tmp_path):
    (tmp_path / "a.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt", encoding="utf-8")
    (tmp_path / "ignore.tmp").write_text("tmp", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.pdf").write_text("nested", encoding="utf-8")

    files = build_internal.collect_supported_files(tmp_path)

    assert [p.name for p in files] == ["a.pdf", "b.txt"]


def test_mineru_preparse_skips_valid_new_artifact(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "a.pdf"
    source.write_text("pdf", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    artifact = (
        storage_root
        / "output"
        / "ws"
        / "a"
        / "auto"
        / "a_content_list.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps([{"type": "text", "text": "ok"}]), encoding="utf-8")
    os.utime(artifact, (source.stat().st_mtime + 1, source.stat().st_mtime + 1))
    called = []
    monkeypatch.setattr(
        build_internal,
        "_run_mineru_preparse_command",
        lambda input_dir, output_root: called.append((input_dir, output_root)),
    )

    summary = build_internal.preparse_mineru_files(
        profile,
        [source],
        tmp_path / "reports",
    )

    assert called == []
    assert summary["skipped_count"] == 1
    assert summary["parsed_count"] == 0


def test_mineru_preparse_skips_valid_legacy_safe_stem_artifact(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "A Test.pdf"
    source.write_text("pdf", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    artifact = (
        storage_root
        / "output"
        / "ws"
        / "A_Test"
        / "A Test"
        / "auto"
        / "A Test_content_list.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps([{"type": "text", "text": "ok"}]), encoding="utf-8")
    os.utime(artifact, (source.stat().st_mtime + 1, source.stat().st_mtime + 1))
    called = []
    monkeypatch.setattr(
        build_internal,
        "_run_mineru_preparse_command",
        lambda input_dir, output_root: called.append((input_dir, output_root)),
    )

    summary = build_internal.preparse_mineru_files(
        profile,
        [source],
        tmp_path / "reports",
    )

    assert called == []
    assert summary["skipped_count"] == 1
    assert summary["parsed_count"] == 0


def test_mineru_preparse_skips_nonempty_artifact_even_if_older(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "a.pdf"
    source.write_text("pdf", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    artifact = (
        storage_root
        / "output"
        / "ws"
        / "a"
        / "auto"
        / "a_content_list.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps([{"type": "text", "text": "ok"}]), encoding="utf-8")
    os.utime(artifact, (source.stat().st_mtime - 10, source.stat().st_mtime - 10))
    called = []
    monkeypatch.setattr(
        build_internal,
        "_run_mineru_preparse_command",
        lambda input_dir, output_root: called.append((input_dir, output_root)),
    )

    summary = build_internal.preparse_mineru_files(
        profile,
        [source],
        tmp_path / "reports",
    )

    assert called == []
    assert summary["skipped_count"] == 1


def test_mineru_preparse_recovers_nested_artifact_by_fallback_search(
    monkeypatch, tmp_path
):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "R2-2601385 Use cases.docx"
    source.write_text("docx", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    output_root = storage_root / "output" / "ws"
    artifact = (
        output_root
        / "legacy_mineru_output"
        / source.stem
        / "hybrid_auto"
        / f"{source.stem}_content_list.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps([{"type": "text", "text": "ok"}]), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        build_internal,
        "_run_mineru_preparse_command",
        lambda input_dir, output_root: calls.append((input_dir, output_root)),
    )

    summary = build_internal.preparse_mineru_files(
        profile,
        [source],
        tmp_path / "reports",
    )

    assert calls == []
    assert summary["skipped_count"] == 1
    assert summary["recovered_by_fallback_count"] == 1
    assert summary["skipped"][0]["artifact"] == build_internal._path_env(artifact)


def test_mineru_preparse_reuses_bracketed_stem_artifact(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "R2-2601385 Report of [POST133][010][6G AI] Use cases.docx"
    source.write_text("docx", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    output_root = storage_root / "output" / "ws"
    artifact = (
        output_root
        / source.stem
        / "hybrid_auto"
        / f"{source.stem}_content_list.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps([{"type": "text", "text": "ok"}]), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        build_internal,
        "_run_mineru_preparse_command",
        lambda input_dir, output_root: calls.append((input_dir, output_root)),
    )

    summary = build_internal.preparse_mineru_files(
        profile,
        [source],
        tmp_path / "reports",
    )

    assert calls == []
    assert summary["skipped_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["skipped"][0]["artifact"] == build_internal._path_env(artifact)


def test_mineru_preparse_does_not_accept_glob_pattern_near_match(
    monkeypatch, tmp_path
):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "Report [AB].pdf"
    source.write_text("pdf", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    output_root = storage_root / "output" / "ws"
    wrong_artifact = (
        output_root
        / source.stem
        / "hybrid_auto"
        / "Report A_content_list.json"
    )
    wrong_artifact.parent.mkdir(parents=True)
    wrong_artifact.write_text(
        json.dumps([{"type": "text", "text": "wrong"}]),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        build_internal,
        "_run_mineru_preparse_command",
        lambda input_dir, output_root: calls.append((input_dir, output_root)),
    )

    summary = build_internal.preparse_mineru_files(
        profile,
        [source],
        tmp_path / "reports",
    )

    assert calls
    assert summary["skipped_count"] == 0
    assert summary["missing_artifact_count"] == 1


def test_processor_reuses_legacy_safe_stem_mineru_output(tmp_path):
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "output" / "ws"
    raw_dir.mkdir()
    source = raw_dir / "A Test.pdf"
    source.write_text("pdf", encoding="utf-8")
    artifact = (
        output_root
        / "A_Test"
        / "A Test"
        / "auto"
        / "A Test_content_list.json"
    )
    artifact.parent.mkdir(parents=True)
    image_path = artifact.parent / "images" / "x.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"png")
    artifact.write_text(
        json.dumps([{"type": "image", "img_path": "images/x.png"}]),
        encoding="utf-8",
    )
    os.utime(artifact, (source.stat().st_mtime + 1, source.stat().st_mtime + 1))

    content = asyncio.run(
        _DummyProcessor()._try_load_existing_mineru_output(
            file_path=source,
            output_dir=str(output_root),
            parse_method="auto",
        )
    )

    assert content == [{"type": "image", "img_path": str(image_path.resolve())}]


def test_processor_reuses_safe_stem_method_dir_without_stem_parent(tmp_path):
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "output" / "ws"
    raw_dir.mkdir()
    source = raw_dir / "A Test.pdf"
    source.write_text("pdf", encoding="utf-8")
    artifact = output_root / "A_Test" / "hybrid_auto" / "A Test_content_list.json"
    artifact.parent.mkdir(parents=True)
    image_path = artifact.parent / "images" / "x.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"png")
    artifact.write_text(
        json.dumps([{"type": "image", "img_path": "images/x.png"}]),
        encoding="utf-8",
    )

    content = asyncio.run(
        _DummyProcessor()._try_load_existing_mineru_output(
            file_path=source,
            output_dir=str(output_root),
            parse_method="auto",
        )
    )

    assert content == [{"type": "image", "img_path": str(image_path.resolve())}]


def test_processor_reuses_nested_mineru_output_by_fallback_search(tmp_path):
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "output" / "ws"
    raw_dir.mkdir()
    source = raw_dir / "R2-2601385 Use cases.pdf"
    source.write_text("pdf", encoding="utf-8")
    artifact = (
        output_root
        / "legacy_mineru_output"
        / source.stem
        / "hybrid_auto"
        / f"{source.stem}_content_list.json"
    )
    artifact.parent.mkdir(parents=True)
    image_path = artifact.parent / "images" / "x.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"png")
    artifact.write_text(
        json.dumps([{"type": "image", "img_path": "images/x.png"}]),
        encoding="utf-8",
    )

    content = asyncio.run(
        _DummyProcessor()._try_load_existing_mineru_output(
            file_path=source,
            output_dir=str(output_root),
            parse_method="auto",
        )
    )

    assert content == [{"type": "image", "img_path": str(image_path.resolve())}]


def test_processor_reuses_bracketed_stem_mineru_output(tmp_path):
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "output" / "ws"
    raw_dir.mkdir()
    source = raw_dir / "R2-2601385 Report of [POST133][010][6G AI] Use cases.pdf"
    source.write_text("pdf", encoding="utf-8")
    artifact = (
        output_root
        / source.stem
        / "hybrid_auto"
        / f"{source.stem}_content_list.json"
    )
    artifact.parent.mkdir(parents=True)
    image_path = artifact.parent / "images" / "x.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"png")
    artifact.write_text(
        json.dumps([{"type": "image", "img_path": "images/x.png"}]),
        encoding="utf-8",
    )

    content = asyncio.run(
        _DummyProcessor()._try_load_existing_mineru_output(
            file_path=source,
            output_dir=str(output_root),
            parse_method="auto",
        )
    )

    assert content == [{"type": "image", "img_path": str(image_path.resolve())}]


def test_processor_does_not_reuse_glob_pattern_near_match(tmp_path):
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "output" / "ws"
    raw_dir.mkdir()
    source = raw_dir / "Report [AB].pdf"
    source.write_text("pdf", encoding="utf-8")
    wrong_artifact = (
        output_root
        / source.stem
        / "hybrid_auto"
        / "Report A_content_list.json"
    )
    wrong_artifact.parent.mkdir(parents=True)
    wrong_artifact.write_text(
        json.dumps([{"type": "text", "text": "wrong"}]),
        encoding="utf-8",
    )

    content = asyncio.run(
        _DummyProcessor()._try_load_existing_mineru_output(
            file_path=source,
            output_dir=str(output_root),
            parse_method="auto",
        )
    )

    assert content is None


def test_processor_reuses_older_mineru_output(tmp_path):
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "output" / "ws"
    raw_dir.mkdir()
    source = raw_dir / "A Test.pdf"
    source.write_text("pdf", encoding="utf-8")
    artifact = output_root / "A Test" / "auto" / "A Test_content_list.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps([{"type": "text", "text": "ok"}]),
        encoding="utf-8",
    )
    os.utime(artifact, (source.stat().st_mtime - 10, source.stat().st_mtime - 10))

    content = asyncio.run(
        _DummyProcessor()._try_load_existing_mineru_output(
            file_path=source,
            output_dir=str(output_root),
            parse_method="auto",
        )
    )

    assert content == [{"type": "text", "text": "ok"}]


def test_mineru_preparse_runs_for_missing_or_stale_artifact(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "a.pdf"
    source.write_text("pdf", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    calls = []

    def fake_run(input_dir, output_root):
        calls.append((input_dir, output_root))
        artifact = output_root / "a" / "auto" / "a_content_list.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps([{"type": "text", "text": "ok"}]),
            encoding="utf-8",
        )
        os.utime(artifact, (source.stat().st_mtime + 1, source.stat().st_mtime + 1))

    monkeypatch.setattr(build_internal, "_run_mineru_preparse_command", fake_run)

    summary = build_internal.preparse_mineru_files(
        profile,
        [source],
        tmp_path / "reports",
    )

    assert len(calls) == 1
    assert calls[0][1] == storage_root / "output" / "ws"
    assert summary["pending_count"] == 1
    assert summary["parsed_count"] == 1


def test_mineru_preparse_retries_missing_artifact_as_single_file(
    monkeypatch, tmp_path
):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "a.pdf"
    source.write_text("pdf", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    calls = []

    def fake_run(input_path, output_root):
        calls.append(Path(input_path))
        if len(calls) == 2:
            artifact = output_root / "a" / "auto" / "a_content_list.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                json.dumps([{"type": "text", "text": "ok"}]),
                encoding="utf-8",
            )

    monkeypatch.setattr(build_internal, "_run_mineru_preparse_command", fake_run)

    summary = build_internal.preparse_mineru_files(
        profile,
        [source],
        tmp_path / "reports",
    )

    assert len(calls) == 2
    assert calls[0].name == "inputs"
    assert calls[1].name == "a.pdf"
    assert summary["parsed_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["missing_after_parse"] == []


def test_mineru_preparse_writes_missing_artifact_reports(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "missing.pdf"
    source.write_text("pdf", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    monkeypatch.setattr(
        build_internal,
        "_run_mineru_preparse_command",
        lambda input_path, output_root: None,
    )

    report_dir = tmp_path / "reports"
    summary = build_internal.preparse_mineru_files(profile, [source], report_dir)

    assert summary["failed_count"] == 1
    missing = json.loads(
        (report_dir / "mineru_preparse_missing_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    all_failures = json.loads(
        (report_dir / "mineru_preparse_failures_all.json").read_text(
            encoding="utf-8"
        )
    )
    preparse_summary = json.loads(
        (report_dir / "mineru_preparse_summary.json").read_text(encoding="utf-8")
    )
    assert missing[0]["file"] == "missing.pdf"
    assert missing[0]["stem"] == "missing"
    assert missing[0]["output_root"]
    assert missing[0]["searched_roots"]
    assert "candidate_reject_reasons" in missing[0]
    assert all_failures[0]["stage"] == "mineru_preparse_missing_artifact"
    assert preparse_summary["failed_count"] == 1
    assert preparse_summary["missing_artifact_count"] == 1
    assert preparse_summary["conversion_failed_count"] == 0


def test_mineru_preparse_converts_office_pdf_inside_file_dir(monkeypatch, tmp_path):
    from raganything.parser import MineruParser

    output_root = tmp_path / "output" / "ws"
    source = tmp_path / "raw" / "A Test.docx"
    source.parent.mkdir()
    source.write_text("docx", encoding="utf-8")
    calls = []

    def fake_convert(doc_path, output_dir):
        calls.append((Path(doc_path), Path(output_dir)))
        pdf = Path(output_dir) / "A Test.pdf"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF")
        os.utime(pdf, (source.stat().st_mtime + 1, source.stat().st_mtime + 1))
        return pdf

    monkeypatch.setattr(MineruParser, "convert_office_to_pdf", fake_convert)

    pdf_path = build_internal._prepare_mineru_preparse_input(source, output_root)

    assert calls == [(source, output_root / "A_Test")]
    assert pdf_path == output_root / "A_Test" / "A Test.pdf"


def test_mineru_preparse_reuses_legacy_root_converted_pdf(tmp_path):
    output_root = tmp_path / "output" / "ws"
    source = tmp_path / "raw" / "A Test.docx"
    source.parent.mkdir()
    source.write_text("docx", encoding="utf-8")
    legacy_pdf = output_root / "A Test.pdf"
    legacy_pdf.parent.mkdir(parents=True)
    legacy_pdf.write_bytes(b"%PDF")
    os.utime(
        legacy_pdf,
        (source.stat().st_mtime + 1, source.stat().st_mtime + 1),
    )

    pdf_path = build_internal._prepare_mineru_preparse_input(source, output_root)

    assert pdf_path == legacy_pdf


def test_mineru_preparse_reuses_legacy_root_converted_pdf_even_if_older(
    monkeypatch, tmp_path
):
    from raganything.parser import MineruParser

    output_root = tmp_path / "output" / "ws"
    source = tmp_path / "raw" / "A Test.docx"
    source.parent.mkdir()
    source.write_text("docx", encoding="utf-8")
    legacy_pdf = output_root / "A Test.pdf"
    legacy_pdf.parent.mkdir(parents=True)
    legacy_pdf.write_bytes(b"%PDF")
    os.utime(
        legacy_pdf,
        (source.stat().st_mtime - 10, source.stat().st_mtime - 10),
    )
    called = []
    monkeypatch.setattr(
        MineruParser,
        "convert_office_to_pdf",
        lambda doc_path, output_dir: called.append((doc_path, output_dir)),
    )

    pdf_path = build_internal._prepare_mineru_preparse_input(source, output_root)

    assert pdf_path == legacy_pdf
    assert called == []


def test_mineru_preparse_records_conversion_failure_and_continues(
    monkeypatch, tmp_path
):
    from raganything.parser import MineruParser

    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    bad_doc = raw_dir / "Bad File.docx"
    good_pdf = raw_dir / "Good File.pdf"
    bad_doc.write_text("docx", encoding="utf-8")
    good_pdf.write_text("pdf", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )

    def fake_convert(doc_path, output_dir):
        raise RuntimeError("conversion failed")

    def fake_run(input_dir, output_root):
        staged_names = sorted(path.name for path in Path(input_dir).iterdir())
        assert staged_names == ["Good File.pdf"]
        artifact = output_root / "Good File" / "auto" / "Good File_content_list.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps([{"type": "text", "text": "ok"}]),
            encoding="utf-8",
        )

    monkeypatch.setattr(MineruParser, "convert_office_to_pdf", fake_convert)
    monkeypatch.setattr(build_internal, "_run_mineru_preparse_command", fake_run)

    report_dir = tmp_path / "reports"
    summary = build_internal.preparse_mineru_files(
        profile,
        [bad_doc, good_pdf],
        report_dir,
    )

    assert summary["failed_count"] == 1
    assert summary["conversion_failed"][0]["file"] == "Bad File.docx"
    assert summary["parsed_count"] == 1
    failures = json.loads(
        (report_dir / "mineru_preparse_failures.json").read_text(encoding="utf-8")
    )
    assert failures[0]["file"] == "Bad File.docx"
    commands = (report_dir / "manual_convert_commands.sh").read_text(
        encoding="utf-8"
    )
    assert "Bad File.docx" in commands
    assert "lo_profile_" in commands
    assert "manual_convert_logs" in commands
    assert "soffice" in commands
    assert 'TIMEOUT_SEC="${LIBREOFFICE_CONVERT_TIMEOUT_SECONDS:-1800}"' in commands
    assert "writer_pdf_Export" in commands
    assert "SAL_USE_VCLPLUGIN" in commands
    assert "pkill -9" in commands
    assert "manual_convert_failed=1" in commands
    assert "One or more manual conversions failed." in commands


def test_mineru_preparse_skips_historical_conversion_failure_without_manual_pdf(
    monkeypatch, tmp_path
):
    from raganything.parser import MineruParser

    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    bad_doc = raw_dir / "Bad File.docx"
    bad_doc.write_text("docx", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    old_report = storage_root / "reports" / "old"
    old_report.mkdir(parents=True)
    (old_report / "mineru_preparse_failures.json").write_text(
        json.dumps(
            [
                {
                    "file": bad_doc.name,
                    "source_path": str(bad_doc.resolve()),
                    "error": "previous conversion failed",
                }
            ]
        ),
        encoding="utf-8",
    )
    convert_calls = []
    run_calls = []
    monkeypatch.setattr(
        MineruParser,
        "convert_office_to_pdf",
        lambda doc_path, output_dir: convert_calls.append((doc_path, output_dir)),
    )
    monkeypatch.setattr(
        build_internal,
        "_run_mineru_preparse_command",
        lambda input_dir, output_root: run_calls.append((input_dir, output_root)),
    )

    report_dir = storage_root / "reports" / "new"
    summary = build_internal.preparse_mineru_files(profile, [bad_doc], report_dir)

    assert convert_calls == []
    assert run_calls == []
    assert summary["failed_count"] == 1
    assert summary["historical_failure_skipped_count"] == 1
    assert summary["conversion_failed"][0]["file"] == bad_doc.name
    assert "Previous MinerU preparse conversion failed" in summary["conversion_failed"][0]["error"]
    assert (report_dir / "mineru_preparse_failures.json").exists()
    assert (report_dir / "manual_convert_commands.sh").exists()


def test_mineru_preparse_recovers_historical_conversion_failure_with_manual_pdf(
    monkeypatch, tmp_path
):
    from raganything.parser import MineruParser

    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    bad_doc = raw_dir / "Bad File.docx"
    bad_doc.write_text("docx", encoding="utf-8")
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=raw_dir,
        storage_root=storage_root,
        workspace_id="ws",
    )
    old_report = storage_root / "reports" / "old"
    old_report.mkdir(parents=True)
    (old_report / "mineru_preparse_failures.json").write_text(
        json.dumps(
            [
                {
                    "file": bad_doc.name,
                    "source_path": str(bad_doc.resolve()),
                    "error": "previous conversion failed",
                }
            ]
        ),
        encoding="utf-8",
    )
    output_root = storage_root / "output" / "ws"
    manual_pdf = output_root / "Bad_File" / "Bad File.pdf"
    manual_pdf.parent.mkdir(parents=True)
    manual_pdf.write_bytes(b"%PDF")
    convert_calls = []
    monkeypatch.setattr(
        MineruParser,
        "convert_office_to_pdf",
        lambda doc_path, output_dir: convert_calls.append((doc_path, output_dir)),
    )

    def fake_run(input_dir, output_root):
        staged_names = sorted(path.name for path in Path(input_dir).iterdir())
        assert staged_names == ["Bad File.pdf"]
        artifact = output_root / "Bad File" / "auto" / "Bad File_content_list.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps([{"type": "text", "text": "ok"}]),
            encoding="utf-8",
        )

    monkeypatch.setattr(build_internal, "_run_mineru_preparse_command", fake_run)

    summary = build_internal.preparse_mineru_files(
        profile,
        [bad_doc],
        storage_root / "reports" / "new",
    )

    assert convert_calls == []
    assert summary["failed_count"] == 0
    assert summary["historical_failure_skipped_count"] == 0
    assert summary["parsed_count"] == 1


def test_build_returns_failure_for_preparse_failure_and_skips_ingest(
    monkeypatch, tmp_path
):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    bad_doc = raw_dir / "bad.docx"
    good_pdf = raw_dir / "good.pdf"
    bad_doc.write_text("docx", encoding="utf-8")
    good_pdf.write_text("pdf", encoding="utf-8")
    fake = _FakeService()

    monkeypatch.setattr(
        build_internal,
        "build_local_settings",
        lambda profile, **kwargs: object(),
    )
    monkeypatch.setattr(build_internal, "create_local_rag_service", lambda settings: fake)
    monkeypatch.setattr(
        build_internal,
        "preparse_mineru_files",
        lambda profile, files, report_dir: {
            "enabled": True,
            "output_dir": str(profile.output_dir / profile.workspace_id),
            "candidate_count": 2,
            "skipped_count": 0,
            "parsed_count": 1,
            "pending_count": 2,
            "failed_count": 1,
            "conversion_failed": [
                {
                    "file": bad_doc.name,
                    "source_path": str(bad_doc.resolve()),
                    "error": "conversion failed",
                }
            ],
        },
    )

    rc = build_internal.run_build(
        SimpleNamespace(
            profile="test",
            raw_dir=str(raw_dir),
            storage_root=str(storage_root),
            workspace_id="ws",
            max_async_ingest=4,
            file_batch_size=4,
            max_file_attempts=1,
            ingest_timeout_seconds=30,
            recycle_service_every=0,
            delete_doc_id=None,
            delete_first_file=False,
            delete_workspace=False,
            delete_llm_cache=False,
            doc_id=None,
            dry_run=False,
            allow_legacy_index_profile_adoption=False,
            skip_mineru_preparse=False,
            delete_check=False,
        )
    )

    assert rc == 1
    assert [Path(call["file_path"]).name for call in fake.ingest_calls] == ["good.pdf"]
    report_dir = sorted((storage_root / "reports").iterdir())[0]
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_count"] == 1
    assert summary["failed_files"][0]["file"] == "bad.docx"
    assert summary["succeeded_count"] == 1


def test_build_writes_partial_summary_after_preparse_if_later_stage_fails(
    monkeypatch, tmp_path
):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    source = raw_dir / "a.pdf"
    source.write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(
        build_internal,
        "build_local_settings",
        lambda profile, **kwargs: object(),
    )
    monkeypatch.setattr(
        build_internal,
        "preparse_mineru_files",
        lambda profile, files, report_dir: {
            "enabled": True,
            "output_dir": str(profile.output_dir / profile.workspace_id),
            "candidate_count": 1,
            "skipped_count": 1,
            "parsed_count": 0,
            "pending_count": 0,
            "failed_count": 0,
            "conversion_failed": [],
            "missing_after_parse": [],
        },
    )
    monkeypatch.setattr(
        build_internal,
        "create_local_rag_service",
        lambda settings: (_ for _ in ()).throw(RuntimeError("warmup failed")),
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        build_internal.run_build(
            SimpleNamespace(
                profile="test",
                raw_dir=str(raw_dir),
                storage_root=str(storage_root),
                workspace_id="ws",
                max_async_ingest=4,
                file_batch_size=4,
                max_file_attempts=1,
                ingest_timeout_seconds=30,
                recycle_service_every=0,
                delete_doc_id=None,
                delete_first_file=False,
                delete_workspace=False,
                delete_llm_cache=False,
                doc_id=None,
                dry_run=False,
                allow_legacy_index_profile_adoption=False,
                skip_mineru_preparse=False,
                delete_check=False,
            )
        )

    report_dir = sorted((storage_root / "reports").iterdir())[0]
    partial = json.loads(
        (report_dir / "summary.partial.json").read_text(encoding="utf-8")
    )
    assert partial["status"] == "preparse_complete"
    assert partial["mineru_preparse"]["failed_count"] == 0
    assert Path(partial["log_file"]).exists()


def test_libreoffice_timeout_defaults_to_60_and_env_override(monkeypatch, tmp_path):
    from raganything.parser import MineruParser
    import raganything.parser as parser_module

    source = tmp_path / "a.docx"
    source.write_text("docx", encoding="utf-8")
    output_dir = tmp_path / "out"
    timeouts = []

    def fake_run(cmd, **kwargs):
        timeouts.append(kwargs["timeout"])
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "a.pdf").write_bytes(b"%PDF" + b"x" * 200)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(parser_module.subprocess, "run", fake_run)
    monkeypatch.delenv("LIBREOFFICE_CONVERT_TIMEOUT_SECONDS", raising=False)

    MineruParser.convert_office_to_pdf(source, output_dir)
    monkeypatch.setenv("LIBREOFFICE_CONVERT_TIMEOUT_SECONDS", "600")
    MineruParser.convert_office_to_pdf(source, output_dir)

    assert timeouts == [60, 600]


def test_script_file_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/build_internal_workspace.py", "--help"],
        cwd=build_internal.REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--profile" in result.stdout
    assert "--max-async-ingest" in result.stdout


def test_dry_run_writes_local_summary_without_server_fields(tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    (raw_dir / "sample.pdf").write_text("pdf", encoding="utf-8")

    result = build_internal.main(
        [
            "--profile",
            "test",
            "--raw-dir",
            str(raw_dir),
            "--storage-root",
            str(storage_root),
            "--workspace-id",
            "internal_smoke",
            "--max-async-ingest",
            "1",
            "--dry-run",
        ]
    )

    assert result == 0
    summaries = sorted((storage_root / "reports").glob("*/summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["workspace_id"] == "internal_smoke"
    assert payload["execution_mode"] == "local_service"
    assert payload["file_count"] == 1
    assert "server_port" not in payload
    assert payload["concurrency"]["max_async_ingest"] == 1
    assert payload["settings"]["enable_entity_disambiguation"] is False
    assert payload["settings"]["enable_synonym_linking"] is True
    assert Path(payload["log_file"]).exists()
    assert payload["warning_count"] == 0
    assert payload["error_count"] == 0
    assert "Dry-run build start" in Path(payload["log_file"]).read_text(
        encoding="utf-8"
    )
    assert not (storage_root / "uploads").exists()


def test_build_uses_local_service_batches_and_exports_reports(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    for name in ("b.pdf", "a.pdf", "c.pdf"):
        (raw_dir / name).write_text(name, encoding="utf-8")
    fake = _patch_fake_service(monkeypatch)

    result = build_internal.main(
        [
            "--profile",
            "test",
            "--raw-dir",
            str(raw_dir),
            "--storage-root",
            str(storage_root),
            "--workspace-id",
            "ws",
            "--max-async-ingest",
            "2",
        ]
    )

    assert result == 0
    assert [Path(call["file_path"]).name for call in fake.ingest_calls] == [
        "a.pdf",
        "b.pdf",
        "c.pdf",
    ]
    assert all(call["workspace_id"] == "ws" for call in fake.ingest_calls)
    assert all(call["serialize_by_workspace_id"] is False for call in fake.ingest_calls)
    assert all(
        Path(call["output_dir"]) == storage_root / "output" / "ws"
        for call in fake.ingest_calls
    )
    assert fake.synonym_calls == [("ws", False, True)]
    report_dir = sorted((storage_root / "reports").iterdir())[0]
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert [batch["files"] for batch in summary["batch_results"]] == [
        ["a.pdf", "b.pdf"],
        ["c.pdf"],
    ]
    assert summary["failed_count"] == 0
    assert summary["mineru_preparse"]["enabled"] is True
    assert summary["mineru_preparse"]["skipped_count"] == 3
    assert Path(summary["log_file"]).exists()
    log_text = Path(summary["log_file"]).read_text(encoding="utf-8")
    assert "Internal build start" in log_text
    assert "Build batch complete" in log_text
    assert summary["error_count"] == 0
    assert json.loads((report_dir / "documents.json").read_text(encoding="utf-8"))["count"] == 2
    assert not (report_dir / "entities.json").exists()
    assert not (report_dir / "relations.json").exists()
    assert not (report_dir / "entities.csv").exists()
    assert not (report_dir / "relations.csv").exists()
    assert json.loads((report_dir / "graph_stats.json").read_text(encoding="utf-8"))[
        "entity_count"
    ] == 1


def test_build_failure_is_written_to_internal_log(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    storage_root = tmp_path / "internal"
    raw_dir.mkdir()
    (raw_dir / "bad.pdf").write_text("bad", encoding="utf-8")
    _patch_fake_service(monkeypatch, fail_names={"bad.pdf"})

    result = build_internal.main(
        [
            "--profile",
            "test",
            "--raw-dir",
            str(raw_dir),
            "--storage-root",
            str(storage_root),
            "--workspace-id",
            "ws",
            "--max-async-ingest",
            "1",
            "--max-file-attempts",
            "1",
        ]
    )

    assert result == 1
    report_dir = sorted((storage_root / "reports").iterdir())[0]
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    log_file = Path(summary["log_file"])
    log_text = log_file.read_text(encoding="utf-8")
    assert summary["failed_count"] == 1
    assert summary["error_count"] >= 1
    assert "ERROR" in log_text
    assert "Ingest failed file=bad.pdf" in log_text
    assert "forced ingest failure: bad.pdf" in log_text


def test_internal_log_captures_stdout_and_stderr(tmp_path):
    context = build_internal.setup_build_logging(tmp_path)
    try:
        print("stdout detail line")
        sys.stderr.write("stderr ERROR detail line\n")
        summary = {}
        build_internal._attach_log_summary(summary, context)
    finally:
        build_internal.close_build_logging(context)

    log_text = Path(summary["log_file"]).read_text(encoding="utf-8")
    assert "stdout detail line" in log_text
    assert "stderr ERROR detail line" in log_text
    assert summary["error_count"] >= 1


def test_internal_log_bridges_non_propagating_logger(tmp_path):
    context = build_internal.setup_build_logging(tmp_path)
    bridged_logger = logging.getLogger("test_internal_bridge_logger")
    old_propagate = bridged_logger.propagate
    old_level = bridged_logger.level
    try:
        bridged_logger.propagate = False
        build_internal._bridge_logger_to_internal_build_log(context, bridged_logger)
        bridged_logger.info("bridged lightrag-style detail")
        summary = {}
        build_internal._attach_log_summary(summary, context)
    finally:
        bridged_logger.propagate = old_propagate
        bridged_logger.setLevel(old_level)
        build_internal.close_build_logging(context)

    log_text = Path(summary["log_file"]).read_text(encoding="utf-8")
    assert "bridged lightrag-style detail" in log_text


def test_report_command_collects_storage_without_ingest(monkeypatch, tmp_path):
    storage_root = tmp_path / "internal"
    fake = _patch_fake_service(monkeypatch)

    result = build_internal.main(
        [
            "report",
            "--profile",
            "test",
            "--storage-root",
            str(storage_root),
            "--workspace-id",
            "ws",
        ]
    )

    assert result == 0
    assert fake.ingest_calls == []
    report_dir = sorted((storage_root / "reports").iterdir())[0]
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert Path(summary["log_file"]).exists()
    assert summary["error_count"] == 0
    assert (report_dir / "documents.csv").exists()
    assert not (report_dir / "entities.csv").exists()
    assert not (report_dir / "relations.csv").exists()
    assert not (report_dir / "entities.json").exists()
    assert not (report_dir / "relations.json").exists()


def test_delete_doc_deletes_storage_then_rebuilds_synonyms(monkeypatch, tmp_path):
    storage_root = tmp_path / "internal"
    fake = _patch_fake_service(monkeypatch)

    result = build_internal.main(
        [
            "delete-doc",
            "--profile",
            "test",
            "--storage-root",
            str(storage_root),
            "--workspace-id",
            "ws",
            "--doc-id",
            "doc-1",
        ]
    )

    assert result == 0
    assert fake.delete_calls == [("ws", "doc-1", False)]
    assert fake.synonym_calls == [("ws", True, True)]
    report_dir = sorted((storage_root / "reports").iterdir())[0]
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert Path(summary["log_file"]).exists()
    assert summary["error_count"] == 0
    delete_check = json.loads((report_dir / "delete_check.json").read_text(encoding="utf-8"))
    assert delete_check["target_doc_id"] == "doc-1"
    assert delete_check["delete_result"]["status"] == "success"
    assert delete_check["synonym_rebuild"]["created_edges"] == 2
