#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print a consolidated EM/F1/Recall@K table from HippoRAG2-aligned eval results.

Usage:
    python evaluate_local/MultiHopQA/print_results_table.py \
        --results-root results/multihopqa_hr2
    python evaluate_local/MultiHopQA/print_results_table.py \
        --results-root results/multihopqa_hr2 \
        --datasets hotpotqa musique 2wiki \
        --modes naive hybrid ppr auto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True, dest="results_root")
    p.add_argument(
        "--datasets", nargs="+",
        default=["hotpotqa", "musique", "2wiki"],
    )
    p.add_argument(
        "--modes", nargs="+",
        default=["naive", "hybrid", "ppr", "auto", "full"],
    )
    p.add_argument(
        "--recall-k", nargs="+", type=int,
        default=[2, 5], dest="recall_k",
    )
    args = p.parse_args()

    root = Path(args.results_root)
    ks = args.recall_k

    col_w = 8
    header = f"{'Dataset':<12} {'Mode':<12} {'EM':>{col_w}} {'F1':>{col_w}}"
    for k in ks:
        header += f" {'R@'+str(k):>{col_w}}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for ds in args.datasets:
        path = root / ds / f"{ds}_summary.json"
        if not path.exists():
            print(f"  {ds}: summary not found ({path})")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        source = data.get("corpus_source", "?")
        n = data.get("n_queries", "?")
        print(f"  # {ds}  corpus_source={source}  n_queries={n}")
        for mode in args.modes:
            m = data.get("results", {}).get(mode)
            if not m:
                continue
            row = f"{ds:<12} {mode:<12} {m.get('em', 0):{col_w}.4f} {m.get('f1', 0):{col_w}.4f}"
            for k in ks:
                val = m.get(f"recall@{k}")
                row += f" {val:{col_w}.4f}" if val is not None else f" {'N/A':>{col_w}}"
            print(row)
        print()


if __name__ == "__main__":
    main()
