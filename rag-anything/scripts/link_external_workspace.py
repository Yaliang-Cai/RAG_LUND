"""Materialise UPLOADS_DIR / OUTPUT_DIR contents for a pre-built workspace.

The web UI lists documents from ``UPLOADS_DIR/<ws>/`` and previews markdown
from ``OUTPUT_DIR/<ws>/**/hybrid_auto/<stem>.md``. When a workspace's LightRAG
index has been copied in from elsewhere (e.g. a DocBench evaluation cache),
the originating PDFs and MinerU parse outputs still live outside the app
tree, so the UI shows an empty workspace. This script wires them in:

* PDFs  → ``UPLOADS_DIR/<ws>/<basename>.pdf`` as **symlinks**.
* MinerU outputs (``docbench_N/<DOCID>/hybrid_auto/…``) → ``OUTPUT_DIR/<ws>/``
  as **hard-link mirrors** (``cp -lr``). Hard links are used because the
  server resolves paths under OUTPUT_DIR for the MD-load and image-serve
  endpoints; symlinks pointing outside OUTPUT_DIR would fail that check.
  Hard links require the source and destination to be on the same
  filesystem.

Example (DocBench, on the Ubuntu host)::

    python scripts/link_external_workspace.py \\
        --workspace-id docbench_shared_graphbm25_20260421_v0_v1_v2 \\
        --docbench-root /data/y50056788/Yaliang/datasets_for_eval/data_for_DocBench \\
        --mineru-root   /data/y50056788/Yaliang/projects/rag-anything/evaluate_local/DocBench/docbench_shared_results/mineru_outputs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from raganything.constants import DEFAULT_OUTPUT_DIR, DEFAULT_UPLOADS_DIR


def _symlink_pdf(link: Path, target: Path, *, dry_run: bool) -> str:
    target = target.resolve()
    if link.is_symlink():
        try:
            current_target = Path(os.readlink(link))
        except OSError:
            current_target = None
        if current_target is not None and (
            current_target == target or current_target.resolve() == target
        ):
            return "ok"
        if dry_run:
            return "would-replace"
        link.unlink()
    elif link.exists():
        return "blocked"

    if dry_run:
        return "would-create"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return "created"


def _hardlink_tree(src: Path, dst: Path, *, dry_run: bool) -> str:
    """Mirror ``src`` into ``dst`` using ``cp -lr`` (recursive hard-link copy)."""
    if dst.exists():
        if dst.is_dir() and any(dst.iterdir()):
            return "ok"
        return "blocked"
    if dry_run:
        return "would-create"
    dst.parent.mkdir(parents=True, exist_ok=True)
    # cp -l: hard link files, -r: recurse. Same filesystem required.
    result = subprocess.run(
        ["cp", "-lr", str(src), str(dst)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"cp -lr failed for {src} -> {dst}: {msg}")
    return "created"


def link_pdfs(workspace_id: str, docbench_root: Path, uploads_dir: Path, dry_run: bool) -> dict:
    dest_dir = uploads_dir / workspace_id
    stats: dict = {"created": 0, "ok": 0, "would-create": 0, "would-replace": 0, "blocked": 0, "collisions": []}
    seen: dict[str, Path] = {}
    for sub in sorted(p for p in docbench_root.iterdir() if p.is_dir()):
        for pdf in sorted(sub.glob("*.pdf")):
            if pdf.name in seen and seen[pdf.name] != pdf:
                stats["collisions"].append((pdf.name, str(seen[pdf.name]), str(pdf)))
                continue
            seen[pdf.name] = pdf
            status = _symlink_pdf(dest_dir / pdf.name, pdf, dry_run=dry_run)
            stats[status] = stats.get(status, 0) + 1
    return stats


def link_mineru(workspace_id: str, mineru_root: Path, output_dir: Path, dry_run: bool) -> dict:
    dest_dir = output_dir / workspace_id
    stats: dict = {"created": 0, "ok": 0, "would-create": 0, "blocked": 0}
    for sub in sorted(p for p in mineru_root.iterdir() if p.is_dir()):
        status = _hardlink_tree(sub, dest_dir / sub.name, dry_run=dry_run)
        stats[status] = stats.get(status, 0) + 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workspace-id", required=True)
    ap.add_argument("--docbench-root", required=True, type=Path,
                    help="Folder containing per-doc subdirs with the original PDF")
    ap.add_argument("--mineru-root", required=True, type=Path,
                    help="Folder containing docbench_<N>/<DOCID>/hybrid_auto/ trees")
    ap.add_argument("--uploads-dir", type=Path,
                    default=Path(os.getenv("RAGANYTHING_UPLOADS_DIR", DEFAULT_UPLOADS_DIR)))
    ap.add_argument("--output-dir", type=Path,
                    default=Path(os.getenv("RAGANYTHING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for label, p in [("docbench-root", args.docbench_root), ("mineru-root", args.mineru_root)]:
        if not p.is_dir():
            print(f"error: --{label} is not a directory: {p}", file=sys.stderr)
            return 2

    print(f"workspace_id : {args.workspace_id}")
    print(f"uploads_dir  : {args.uploads_dir}")
    print(f"output_dir   : {args.output_dir}")
    print(f"docbench     : {args.docbench_root}")
    print(f"mineru       : {args.mineru_root}")
    print(f"mode         : {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print()

    pdf_stats = link_pdfs(args.workspace_id, args.docbench_root, args.uploads_dir, args.dry_run)
    print("PDF symlinks  →", {k: v for k, v in pdf_stats.items() if k != "collisions" and v})
    if pdf_stats["collisions"]:
        print(f"  WARN: {len(pdf_stats['collisions'])} filename collision(s); first few:")
        for name, a, b in pdf_stats["collisions"][:5]:
            print(f"    {name}: kept {a}, skipped {b}")

    try:
        mineru_stats = link_mineru(args.workspace_id, args.mineru_root, args.output_dir, args.dry_run)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Hint: cp -lr requires src and dst to be on the SAME filesystem. "
              "If they aren't, point --output-dir at a path on the source filesystem "
              "(and set RAGANYTHING_OUTPUT_DIR accordingly for the server).", file=sys.stderr)
        return 3
    print("MinerU mirror →", {k: v for k, v in mineru_stats.items() if v})

    blocked = pdf_stats.get("blocked", 0) + mineru_stats.get("blocked", 0)
    if blocked:
        print(f"\nNOTE: {blocked} link(s) blocked by a real file/dir at the target — "
              "remove or rename it and rerun.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
