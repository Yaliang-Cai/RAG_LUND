#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_QUERY_KS = (5, 10, 20, 30, 50)
DEFAULT_SURVEY_KS = (50, 100, 200, 500)


def _metric_value(metrics: dict[str, Any] | None, k: int) -> str:
    if not isinstance(metrics, dict):
        return ""
    value = metrics.get(str(k), metrics.get(k, ""))
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return "" if value is None else str(value)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else None


def _iter_experiment_dirs(base: Path) -> list[Path]:
    if base.is_file():
        if base.name in {"retrieval_summary.json", "survey_retrieval_summary.json"}:
            return [base.parents[2]]
        return []
    if (base / "evaluate_surge_fast").is_dir():
        return [base]
    if base.name == "evaluate_surge_fast":
        return [base.parent]
    direct = sorted(base.glob("surge__*/evaluate_surge_fast"))
    direct.extend(sorted(base.glob("surge_survey__*/evaluate_surge_fast")))
    if direct:
        return sorted({path.parent for path in direct})
    recursive = sorted(base.glob("**/surge__*/evaluate_surge_fast"))
    recursive.extend(sorted(base.glob("**/surge_survey__*/evaluate_surge_fast")))
    return sorted({path.parent for path in recursive})


def _row_for_experiment(
    experiment_dir: Path,
    query_ks: tuple[int, ...],
    survey_ks: tuple[int, ...],
) -> dict[str, str]:
    eval_dir = experiment_dir / "evaluate_surge_fast"
    query_summary = _load_json(
        eval_dir / "retrieval_results_fast" / "retrieval_summary.json"
    )
    survey_summary = _load_json(
        eval_dir / "survey_results_fast" / "survey_retrieval_summary.json"
    )

    row: dict[str, str] = {
        "experiment": experiment_dir.name,
        "has_query": "yes" if query_summary else "no",
        "has_survey": "yes" if survey_summary else "no",
    }

    query_recall = (
        query_summary.get("avg_recall_at_k", {}) if isinstance(query_summary, dict) else {}
    )
    for k in query_ks:
        row[f"query_avg_recall@{k}"] = _metric_value(query_recall, k)

    survey_by_scope = (
        survey_summary.get("macro_recall_at_k_by_scope", {})
        if isinstance(survey_summary, dict)
        else {}
    )
    survey_all = survey_by_scope.get("all", {}) if isinstance(survey_by_scope, dict) else {}
    for k in survey_ks:
        row[f"survey_all_macro_recall@{k}"] = _metric_value(survey_all, k)

    return row


def _parse_k_list(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if not str(raw).strip():
        return default
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return tuple(values) or default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SurGE retrieval-ablation query and survey recall summaries."
        )
    )
    parser.add_argument(
        "base",
        help=(
            "Run directory, experiment directory, evaluate_surge_fast directory, "
            "or a single retrieval/survey summary JSON."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Output CSV path. Use '-' for stdout.",
    )
    parser.add_argument(
        "--query-k-list",
        default=",".join(str(k) for k in DEFAULT_QUERY_KS),
        help="Comma-separated query-level recall@k columns.",
    )
    parser.add_argument(
        "--survey-k-list",
        default=",".join(str(k) for k in DEFAULT_SURVEY_KS),
        help="Comma-separated survey all-scope recall@k columns.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = Path(args.base).expanduser().resolve()
    query_ks = _parse_k_list(args.query_k_list, DEFAULT_QUERY_KS)
    survey_ks = _parse_k_list(args.survey_k_list, DEFAULT_SURVEY_KS)
    experiment_dirs = _iter_experiment_dirs(base)
    if not experiment_dirs:
        print(f"No SurGE experiment summaries found under {base}", file=sys.stderr)
        return 1

    fieldnames = ["experiment", "has_query", "has_survey"]
    fieldnames.extend(f"query_avg_recall@{k}" for k in query_ks)
    fieldnames.extend(f"survey_all_macro_recall@{k}" for k in survey_ks)

    rows = [_row_for_experiment(path, query_ks, survey_ks) for path in experiment_dirs]
    if args.output == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return 0

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
