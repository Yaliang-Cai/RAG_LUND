#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Hop QA Evaluation Orchestrator
=====================================

Sequentially runs evaluate_multihop.py for each requested dataset and mode set.

Usage:
    python run_multihop_evals.py \
        --workspace my_wiki_workspace \
        --working-dir /data/y50056788/.../rag_workspaces/my_wiki_workspace \
        --output-dir /data/y50056788/.../multihop_results \
        --datasets hotpotqa musique 2wiki simpleqa \
        --modes ppr hybrid mix
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parent / "evaluate_local" / "MultiHopQA" / "evaluate_multihop.py"

VALID_DATASETS = ("hotpotqa", "musique", "2wiki", "simpleqa")
VALID_MODES = ("ppr", "ppr_local", "global", "local", "hybrid", "mix", "naive", "rrf", "bypass")


def _run_one(dataset: str, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, str(_SCRIPT),
        "--dataset",     dataset,
        "--workspace",   args.workspace,
        "--working-dir", args.working_dir,
        "--output-dir",  args.output_dir,
        "--modes",       *args.modes,
        "--n-samples",   str(args.n_samples),
        "--seed",        str(args.seed),
        "--recall-k",    *[str(k) for k in args.recall_k],
    ]
    if args.resume:
        cmd.append("--resume")

    print(f"\n{'='*60}")
    print(f"[orchestrator] Dataset: {dataset}  modes: {args.modes}")
    print(f"[orchestrator] cmd: {' '.join(cmd)}")
    print("="*60)

    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-hop QA eval orchestrator")
    p.add_argument("--workspace",   required=True)
    p.add_argument("--working-dir", required=True, dest="working_dir")
    p.add_argument("--output-dir",  required=True, dest="output_dir")
    p.add_argument("--datasets",    nargs="+", default=list(VALID_DATASETS),
                   choices=VALID_DATASETS, metavar="DATASET")
    p.add_argument("--modes",       nargs="+", default=["ppr", "hybrid", "mix"],
                   choices=VALID_MODES, metavar="MODE")
    p.add_argument("--n-samples",   type=int, default=500, dest="n_samples")
    p.add_argument("--recall-k",    type=int, nargs="+", default=[5, 10, 20], dest="recall_k")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    failed = []
    for dataset in args.datasets:
        rc = _run_one(dataset, args)
        if rc != 0:
            print(f"[orchestrator] WARN: {dataset} exited with code {rc}")
            failed.append(dataset)

    print(f"\n[orchestrator] Done. Results in: {args.output_dir}")
    if failed:
        print(f"[orchestrator] Failed datasets: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
