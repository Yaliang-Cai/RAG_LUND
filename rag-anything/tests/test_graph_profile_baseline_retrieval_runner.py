import json
import sys
import types
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")
    stub.CrossEncoder = object
    stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = stub

from evaluate_local import run_graph_profile_baseline_retrieval as runner


def _flag_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_dry_run_writes_default_v0_v1_config_without_subprocess(tmp_path):
    runs_root = tmp_path / "ablation_runs"
    output_root = tmp_path / "retrieval_runs"

    with patch.object(runner.subprocess, "run") as run_mock:
        code = runner.main(
            [
                "--dry-run",
                "--runs-root",
                str(runs_root),
                "--output-root",
                str(output_root),
            ]
        )

    assert code == 0
    run_mock.assert_not_called()

    config = json.loads(
        (output_root / "graphbm25_20260429" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["run_id"] == "graphbm25_20260429"
    assert [profile["key"] for profile in config["profiles"]] == ["v0", "v0_v1"]
    assert all(profile["key"] != "v0_v1_v2" for profile in config["profiles"])

    docbench_commands = [
        stage["command"]
        for stage in config["planned_stages"]
        if stage["task"] == "shared"
    ]
    surge_commands = [
        stage["command"] for stage in config["planned_stages"] if stage["task"] == "surge"
    ]
    assert len(docbench_commands) == 2
    assert len(surge_commands) == 4


def test_runner_builds_profile_specific_baseline_commands(tmp_path):
    calls = []

    def fake_run(command, cwd, env, stdout, stderr, text, check):
        calls.append({"command": list(command), "env": dict(env), "cwd": cwd})
        stdout.write("ok\n")
        return types.SimpleNamespace(returncode=0)

    with patch.object(runner.subprocess, "run", side_effect=fake_run):
        code = runner.main(
            [
                "--run-id",
                "graphbm25_20260429",
                "--runs-root",
                str(tmp_path / "ablation_runs"),
                "--output-root",
                str(tmp_path / "retrieval_runs"),
            ]
        )

    assert code == 0
    assert len(calls) == 6

    doc_v0 = calls[0]["command"]
    surge_v0 = calls[1]["command"]
    surge_survey_v0 = calls[2]["command"]
    doc_v1 = calls[3]["command"]
    surge_v1 = calls[4]["command"]
    surge_survey_v1 = calls[5]["command"]

    assert doc_v0[:3] == [sys.executable, "-m", "evaluate_local.DocBench.evaluate_shared"]
    assert "--mode" in doc_v0
    assert _flag_value(doc_v0, "--mode") == "generate"
    assert _flag_value(doc_v0, "--shared_workspace_id") == (
        "docbench_shared_graphbm25_20260429_v0"
    )
    assert _flag_value(doc_v0, "--enable-entity-disambiguation") == "false"
    assert _flag_value(doc_v0, "--enable-synonym-linking") == "false"
    assert _flag_value(doc_v0, "--enable-multi-hop") == "false"
    assert _flag_value(doc_v0, "--query_mode") == "hybrid"
    assert _flag_value(doc_v0, "--keyword_fanout_mode") == "joined"
    assert _flag_value(doc_v0, "--entity_retrieval_mode") == "dense"
    assert _flag_value(doc_v0, "--chunk_retrieval_mode") == "dense"
    assert _flag_value(doc_v0, "--enable_rerank") == "true"
    assert _flag_value(doc_v0, "--enable_kg_rerank") == "false"
    assert _flag_value(doc_v0, "--exclude_synonym_edges") == "true"
    assert _flag_value(doc_v0, "--kg_chunk_selection_source") == "truncated"
    assert _flag_value(doc_v0, "--answer_context_mode") == "kg_prompt"

    assert _flag_value(doc_v1, "--shared_workspace_id") == (
        "docbench_shared_graphbm25_20260429_v0_v1"
    )
    assert _flag_value(doc_v1, "--enable-entity-disambiguation") == "true"
    assert _flag_value(doc_v1, "--enable-synonym-linking") == "false"

    assert surge_v0[:3] == [
        sys.executable,
        "-m",
        "evaluate_local.SurGE.evaluate_surge_fast",
    ]
    assert _flag_value(surge_v0, "--mode") == "retrieval"
    assert _flag_value(surge_v0, "--workspace-id") == (
        "surge_fast_graphbm25_20260429_v0"
    )
    assert _flag_value(surge_v0, "--enable-entity-disambiguation") == "false"
    assert _flag_value(surge_v0, "--enable-synonym-linking") == "false"
    assert _flag_value(surge_v0, "--query-mode") == "hybrid"
    assert _flag_value(surge_v0, "--keyword_fanout_mode") == "joined"
    assert _flag_value(surge_v0, "--entity_retrieval_mode") == "dense"
    assert _flag_value(surge_v0, "--chunk_retrieval_mode") == "dense"
    assert _flag_value(surge_v0, "--enable-rerank") == "true"
    assert _flag_value(surge_v0, "--enable-kg-rerank") == "false"
    assert _flag_value(surge_v0, "--exclude_synonym_edges") == "true"
    assert _flag_value(surge_v0, "--kg-chunk-selection-source") == "untruncated"
    assert _flag_value(surge_v0, "--chunk-top-k") == "0"

    assert _flag_value(surge_survey_v0, "--mode") == "survey"
    assert _flag_value(surge_survey_v0, "--survey-stage") == "retrieval"
    assert _flag_value(surge_survey_v0, "--kg-chunk-selection-source") == "untruncated"

    assert _flag_value(surge_v1, "--workspace-id") == (
        "surge_fast_graphbm25_20260429_v0_v1"
    )
    assert _flag_value(surge_v1, "--enable-entity-disambiguation") == "true"
    assert _flag_value(surge_v1, "--enable-synonym-linking") == "false"
    assert _flag_value(surge_survey_v1, "--survey-stage") == "retrieval"

    assert calls[0]["env"]["QDRANT_ENABLE_SPARSE_BM25"] == "true"
    assert calls[0]["env"]["QDRANT_SPARSE_BM25_MODEL"] == "Qdrant/bm25"
    assert calls[0]["env"]["NEO4J_WORKSPACE"] == "docbench_shared_graphbm25_20260429_v0"
    assert calls[0]["env"]["QDRANT_WORKSPACE"] == "docbench_shared_graphbm25_20260429_v0"
    assert calls[1]["env"]["NEO4J_WORKSPACE"] == "surge_fast_graphbm25_20260429_v0"
    assert calls[1]["env"]["QDRANT_WORKSPACE"] == "surge_fast_graphbm25_20260429_v0"
