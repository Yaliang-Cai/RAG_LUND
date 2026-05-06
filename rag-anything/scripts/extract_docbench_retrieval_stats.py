#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_TYPES = (
    "meta-data",
    "text-only",
    "multimodal-t",
    "multimodal-f",
    "unanswerable",
    "un-web",
)

TYPE_ALIASES = {
    "un-web": ("un-web", "una-web"),
    "una-web": ("una-web", "un-web"),
}


def _metric_value(metrics: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(metrics, dict):
        return ""
    value = metrics.get(key, "")
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _find_type_metrics(by_type: dict[str, Any], label: str) -> dict[str, Any] | None:
    candidates = TYPE_ALIASES.get(label, (label,))
    normalized = {str(key).strip().lower(): value for key, value in by_type.items()}
    for candidate in candidates:
        value = normalized.get(candidate.strip().lower())
        if isinstance(value, dict):
            return value
    return None


def _iter_statistics_files(base: Path) -> list[Path]:
    if base.is_file():
        return [base] if base.name == "statistics.json" else []
    if base.name == "evaluate_shared":
        candidate = base / "statistics.json"
        return [candidate] if candidate.exists() else []
    direct = sorted(base.glob("docbench__*/evaluate_shared/statistics.json"))
    if direct:
        return direct
    return sorted(base.glob("**/docbench__*/evaluate_shared/statistics.json"))


def _row_for_stats(path: Path, type_labels: tuple[str, ...]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    overall = payload.get("overall", {})
    by_type = payload.get("by_type", {})
    if not isinstance(by_type, dict):
        by_type = {}

    row: dict[str, Any] = {
        "experiment": path.parent.parent.name,
        "overall": _metric_value(overall, "accuracy"),
    }
    for label in type_labels:
        metrics = _find_type_metrics(by_type, label)
        row[label] = _metric_value(metrics, "accuracy")
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DocBench retrieval-ablation statistics from "
            "docbench__*/evaluate_shared/statistics.json files."
        )
    )
    parser.add_argument(
        "base",
        help=(
            "Run directory, evaluate_shared directory, or a single statistics.json. "
            "Example: evaluate_local/retrieval_ablation_runs/retrieval_v4_graphbm25_20260504_docbench"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Output CSV path. Use '-' for stdout.",
    )
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="Comma-separated by_type labels to extract.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = Path(args.base).expanduser().resolve()
    type_labels = tuple(
        token.strip() for token in str(args.types).split(",") if token.strip()
    )
    stats_files = _iter_statistics_files(base)
    if not stats_files:
        print(f"No statistics.json files found under {base}", file=sys.stderr)
        return 1

    fieldnames = [
        "experiment",
        "overall",
    ]
    fieldnames.extend(type_labels)

    rows = [_row_for_stats(path, type_labels) for path in stats_files]
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
