#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download the exact HippoRAG2 evaluation datasets from HuggingFace.

Source: https://huggingface.co/datasets/osunlp/HippoRAG_v2

Downloads 6 files into --output-dir:
  hotpotqa.json              (1 000 queries)
  hotpotqa_corpus.json       (9 221 paragraphs)
  musique.json               (1 000 queries)
  musique_corpus.json        (6 119 paragraphs)
  2wikimultihopqa.json       (1 000 queries)
  2wikimultihopqa_corpus.json (11 656 paragraphs)

Usage:
    python evaluate_local/MultiHopQA/download_hipporag2_datasets.py \
        --output-dir evaluate_local/MultiHopQA/hipporag2_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HF_REPO = "osunlp/HippoRAG_v2"
_HF_SUBFOLDER = "reproduce/dataset"

_FILES = [
    "hotpotqa.json",
    "hotpotqa_corpus.json",
    "musique.json",
    "musique_corpus.json",
    "2wikimultihopqa.json",
    "2wikimultihopqa_corpus.json",
]

# Expected sizes for a basic sanity check after download
_EXPECTED_SIZES = {
    "hotpotqa.json":               1_000,
    "hotpotqa_corpus.json":        9_221,
    "musique.json":                1_000,
    "musique_corpus.json":         6_119,
    "2wikimultihopqa.json":        1_000,
    "2wikimultihopqa_corpus.json": 11_656,
}


def _download_with_huggingface_hub(output_dir: Path, force: bool) -> None:
    from huggingface_hub import hf_hub_download

    for filename in _FILES:
        dest = output_dir / filename
        if dest.exists() and not force:
            print(f"  [skip] {filename} already exists")
            continue
        print(f"  [download] {filename} ...", end="", flush=True)
        local_path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=f"{_HF_SUBFOLDER}/{filename}",
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place the file in a subfolder; move if needed
        downloaded = Path(local_path)
        if downloaded != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded.rename(dest)
        print(f" done ({dest.stat().st_size // 1024} KB)")


def _download_with_requests(output_dir: Path, force: bool) -> None:
    import urllib.request

    base_url = (
        f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/{_HF_SUBFOLDER}"
    )
    for filename in _FILES:
        dest = output_dir / filename
        if dest.exists() and not force:
            print(f"  [skip] {filename} already exists")
            continue
        url = f"{base_url}/{filename}"
        print(f"  [download] {filename} from {url} ...", end="", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f" done ({dest.stat().st_size // 1024} KB)")


def _verify(output_dir: Path) -> bool:
    ok = True
    for filename, expected_len in _EXPECTED_SIZES.items():
        path = output_dir / filename
        if not path.exists():
            print(f"  [MISSING] {filename}")
            ok = False
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  [INVALID JSON] {filename}: {exc}")
            ok = False
            continue
        actual = len(data)
        status = "OK" if actual == expected_len else "WARN"
        print(f"  [{status}] {filename}: {actual} records (expected {expected_len})")
        if actual != expected_len:
            ok = False
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Download HippoRAG2 evaluation datasets")
    p.add_argument(
        "--output-dir",
        default="evaluate_local/MultiHopQA/hipporag2_data",
        dest="output_dir",
        help="Directory to save the 6 JSON files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    p.add_argument(
        "--backend",
        choices=["huggingface_hub", "requests", "auto"],
        default="auto",
        help="Download backend (auto tries huggingface_hub first, falls back to requests)",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    if args.backend == "huggingface_hub":
        _download_with_huggingface_hub(output_dir, args.force)
    elif args.backend == "requests":
        _download_with_requests(output_dir, args.force)
    else:
        try:
            import huggingface_hub  # noqa: F401
            _download_with_huggingface_hub(output_dir, args.force)
        except ImportError:
            print("huggingface_hub not installed, falling back to urllib")
            _download_with_requests(output_dir, args.force)

    print("\nVerifying downloads:")
    ok = _verify(output_dir)
    if ok:
        print("\nAll files verified. Run evaluation with:")
        print("  python evaluate_local/MultiHopQA/build_index.py \\")
        print("    --dataset hotpotqa \\")
        print("    --workspace hotpotqa_hr2 \\")
        print("    --working-dir /path/to/workspace \\")
        print(f"    --hipporag2-data-dir {output_dir}")
    else:
        print("\nSome files missing or incorrect. Re-run with --force to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
