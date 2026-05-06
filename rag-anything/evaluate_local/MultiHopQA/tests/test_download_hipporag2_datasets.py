import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from evaluate_local.MultiHopQA import download_hipporag2_datasets as downloader  # noqa: E402
from evaluate_local.MultiHopQA.download_hipporag2_datasets import (  # noqa: E402
    _candidate_hf_paths,
)


def test_candidate_hf_paths_prefers_repo_root_and_keeps_legacy_subfolder():
    assert _candidate_hf_paths("hotpotqa.json") == [
        "hotpotqa.json",
        "reproduce/dataset/hotpotqa.json",
    ]


def test_auto_backend_falls_back_to_requests_after_hf_download_error(
    tmp_path, monkeypatch
):
    calls = []

    def fail_huggingface_hub(output_dir, force):
        calls.append(("hf", output_dir, force))
        raise RuntimeError("hf download failed")

    def succeed_requests(output_dir, force):
        calls.append(("requests", output_dir, force))

    def verify(output_dir):
        calls.append(("verify", output_dir))
        return True

    monkeypatch.setattr(
        downloader,
        "_download_with_huggingface_hub",
        fail_huggingface_hub,
    )
    monkeypatch.setattr(downloader, "_download_with_requests", succeed_requests)
    monkeypatch.setattr(downloader, "_verify", verify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_hipporag2_datasets.py",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "auto",
        ],
    )

    downloader.main()

    assert calls == [
        ("hf", tmp_path.resolve(), False),
        ("requests", tmp_path.resolve(), False),
        ("verify", tmp_path.resolve()),
    ]


def test_runner_defaults_to_requests_download_backend():
    script = (
        Path(__file__).resolve().parents[1] / "run_hipporag2_eval.sh"
    ).read_text(encoding="utf-8")

    assert (
        'HIPPO_RAG2_DOWNLOAD_BACKEND="${HIPPO_RAG2_DOWNLOAD_BACKEND:-requests}"'
        in script
    )
    assert '--backend "${HIPPO_RAG2_DOWNLOAD_BACKEND}"' in script
