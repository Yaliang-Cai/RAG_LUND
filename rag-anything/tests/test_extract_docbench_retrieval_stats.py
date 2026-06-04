import json
from pathlib import Path

from scripts import extract_docbench_retrieval_stats as stats


def _write_stats(path: Path, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "overall": {"accuracy": accuracy},
                "by_type": {
                    "text-only": {"accuracy": accuracy + 1.0},
                    "una-web": {"accuracy": accuracy + 2.0},
                },
            }
        ),
        encoding="utf-8",
    )


def test_iter_statistics_files_supports_docbench_single_evaluate_layout(tmp_path: Path):
    first = tmp_path / "docbench_single__v4_naive_dense" / "evaluate" / "statistics.json"
    second = tmp_path / "docbench_single__v7_ppr_no_synonym_edges" / "evaluate" / "statistics.json"
    _write_stats(first, 10.0)
    _write_stats(second, 20.0)

    assert stats._iter_statistics_files(tmp_path) == [first, second]


def test_row_for_single_doc_stats_uses_experiment_directory_name(tmp_path: Path):
    path = tmp_path / "docbench_single__v4_naive_dense" / "evaluate" / "statistics.json"
    _write_stats(path, 10.0)

    row = stats._row_for_stats(path, ("text-only", "un-web"))

    assert row == {
        "experiment": "docbench_single__v4_naive_dense",
        "overall": "10.000000",
        "text-only": "11.000000",
        "un-web": "12.000000",
    }
