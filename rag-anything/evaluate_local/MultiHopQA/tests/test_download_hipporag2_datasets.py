import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from evaluate_local.MultiHopQA.download_hipporag2_datasets import (  # noqa: E402
    _candidate_hf_paths,
)


def test_candidate_hf_paths_prefers_repo_root_and_keeps_legacy_subfolder():
    assert _candidate_hf_paths("hotpotqa.json") == [
        "hotpotqa.json",
        "reproduce/dataset/hotpotqa.json",
    ]
