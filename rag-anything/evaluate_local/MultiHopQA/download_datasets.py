#!/usr/bin/env python
"""One-time script to download all benchmark datasets to HuggingFace local cache.

Usage:
    python evaluate_local/MultiHopQA/download_datasets.py
    python evaluate_local/MultiHopQA/download_datasets.py --data-dir /data/hf_cache
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="HuggingFace cache dir (sets HF_DATASETS_CACHE)")
    args = p.parse_args()

    if args.data_dir:
        os.environ["HF_DATASETS_CACHE"] = args.data_dir
        print(f"[download] HF cache dir: {args.data_dir}")

    from datasets import load_dataset

    configs = [
        ("hotpot_qa",                    "distractor",  "validation", True),
        ("dgslibisey/MuSiQue",           None,          "validation", True),
        ("framolfese/2WikiMultihopQA",    None,          "validation", True),
        ("basicv8vc/SimpleQA",           None,          "test",       True),
    ]

    for name, config, split, trust in configs:
        label = f"{name}[{config or 'default'}]/{split}"
        print(f"[download] Downloading {label} ...")
        try:
            kwargs = {}
            if trust:
                kwargs["trust_remote_code"] = True
            if config:
                load_dataset(name, config, split=split, **kwargs)
            else:
                load_dataset(name, split=split, **kwargs)
            print(f"[download] OK: {label}")
        except Exception as e:
            print(f"[download] FAILED: {label} — {e}")

    print("[download] Done.")


if __name__ == "__main__":
    main()
