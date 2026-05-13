from pathlib import Path
import json
import subprocess
import sys

from scripts import build_internal_workspace as build_internal


def test_test_profile_resolves_internal_test_paths():
    profile = build_internal.resolve_profile("test")

    assert profile.raw_dir == Path("/data/y50056788/Yaliang/datasets_raw_test")
    assert profile.storage_root == Path("/data/y50056788/Yaliang/internal_test")
    assert profile.workspace_id == "internal_test"
    assert profile.uploads_dir == profile.storage_root / "uploads"
    assert profile.output_dir == profile.storage_root / "output"
    assert profile.working_dir_root == profile.storage_root / "rag_workspace"
    assert profile.log_dir == profile.storage_root / "logs"


def test_prod_profile_resolves_internal_paths():
    profile = build_internal.resolve_profile("prod")

    assert profile.raw_dir == Path("/data/y50056788/Yaliang/datasets_raw")
    assert profile.storage_root == Path("/data/y50056788/Yaliang/internal")
    assert profile.workspace_id == "internal"


def test_server_env_enables_internal_build_defaults():
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=Path("/tmp/raw"),
        storage_root=Path("/tmp/internal"),
        workspace_id="demo_ws",
    )

    env = build_internal.build_server_env(profile, base_env={"EXISTING": "1"})

    assert env["EXISTING"] == "1"
    assert env["RAGANYTHING_WORKDIR_ROOT"] == "/tmp/internal/rag_workspace"
    assert env["RAGANYTHING_OUTPUT_DIR"] == "/tmp/internal/output"
    assert env["RAGANYTHING_UPLOADS_DIR"] == "/tmp/internal/uploads"
    assert env["RAGANYTHING_LOG_DIR"] == "/tmp/internal/logs"
    assert env["RAGANYTHING_ENABLE_SYNONYM_LINKING"] == "true"
    assert env["RAGANYTHING_ENABLE_RESILIENCE"] == "true"
    assert env["RAGANYTHING_ENABLE_ENTITY_SURFACE_NORMALIZATION"] == "true"
    assert env["RAGANYTHING_ENABLE_KEYWORD_CASE_NORMALIZATION"] == "true"
    assert env["RAGANYTHING_STRICT_RELATION_ENDPOINT_ENTITY_MATCH"] == "true"
    assert env["ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE"] == "true"
    assert env["CONTEXT_ZERO_WINDOW_CONTENT_TYPES"]
    assert env["MAX_CONCURRENT_FILES"] == "4"


def test_server_env_allows_file_batch_size_override():
    profile = build_internal.resolve_profile(
        "test",
        raw_dir=Path("/tmp/raw"),
        storage_root=Path("/tmp/internal"),
        workspace_id="demo_ws",
    )

    env = build_internal.build_server_env(
        profile,
        base_env={},
        max_concurrent_files=1,
    )

    assert env["MAX_CONCURRENT_FILES"] == "1"


def test_supported_files_are_top_level_only(tmp_path):
    (tmp_path / "a.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt", encoding="utf-8")
    (tmp_path / "ignore.tmp").write_text("tmp", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.pdf").write_text("nested", encoding="utf-8")

    files = build_internal.collect_supported_files(tmp_path)

    assert [p.name for p in files] == ["a.pdf", "b.txt"]


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


def test_dry_run_writes_constants_backed_summary(tmp_path):
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
            "--file-batch-size",
            "1",
            "--dry-run",
        ]
    )

    assert result == 0
    summaries = sorted((storage_root / "reports").glob("*/summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["workspace_id"] == "internal_smoke"
    assert payload["file_count"] == 1
    assert payload["server_port"] > 0
    assert payload["concurrency"]["file_batch_size"] == 1
    assert payload["concurrency"]["MAX_CONCURRENT_FILES"] == "1"
    assert (
        payload["concurrency"]["lightrag_llm_model_max_async"]
        == build_internal.DEFAULT_LLM_MODEL_MAX_ASYNC
    )
