#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precisely delete one evaluation workspace from all three storage layers:
  1. Qdrant  — delete all vectors tagged workspace_id == WORKSPACE
  2. Neo4j   — detach-delete all nodes with label :WORKSPACE
  3. Local   — rm -rf WORKING_DIR

Usage:
    python evaluate_local/MultiHopQA/drop_workspace.py \
        --workspace hotpotqa_500_seed42 \
        --working-dir /data/rag_workspaces/hotpotqa_500_seed42

    # Dry run — show what would be deleted without touching anything:
    python evaluate_local/MultiHopQA/drop_workspace.py \
        --workspace hotpotqa_500_seed42 \
        --working-dir /data/rag_workspaces/hotpotqa_500_seed42 \
        --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
_lightrag_root = _project_root.parent / "lightrag"
for p in (_project_root, _lightrag_root):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def _qdrant_drop(workspace: str, dry_run: bool) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
    except ImportError:
        print("[drop] SKIP Qdrant: qdrant-client not installed")
        return

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    client = QdrantClient(url=url, api_key=api_key)

    workspace_filter = Filter(must=[FieldCondition(key="workspace_id", match=MatchValue(value=workspace))])

    collections = [c.name for c in client.get_collections().collections]
    if not collections:
        print("[drop] Qdrant: no collections found")
        return

    for col in collections:
        try:
            count = client.count(collection_name=col, count_filter=workspace_filter, exact=True).count
        except Exception:
            continue
        if count == 0:
            continue
        print(f"[drop] Qdrant collection '{col}': {count} points with workspace_id='{workspace}'")
        if not dry_run:
            client.delete(collection_name=col, points_selector=workspace_filter)
            print(f"[drop] Qdrant: deleted {count} points from '{col}'")
        else:
            print(f"[drop] Qdrant: (dry-run) would delete {count} points from '{col}'")


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

async def _neo4j_drop(workspace: str, dry_run: bool) -> None:
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        print("[drop] SKIP Neo4j: neo4j driver not installed")
        return

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    # Escape backticks in the label to prevent injection
    safe_label = workspace.replace("`", "``")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session(database=database) as session:
            count_result = await session.run(
                f"MATCH (n:`{safe_label}`) RETURN count(n) AS n"
            )
            record = await count_result.single()
            count = record["n"] if record else 0
            await count_result.consume()

            if count == 0:
                print(f"[drop] Neo4j: no nodes with label :{workspace}")
            else:
                print(f"[drop] Neo4j: {count} nodes with label :{workspace}")
                if not dry_run:
                    await session.run(f"MATCH (n:`{safe_label}`) DETACH DELETE n")
                    print(f"[drop] Neo4j: deleted {count} nodes")
                else:
                    print(f"[drop] Neo4j: (dry-run) would DETACH DELETE {count} nodes")
    finally:
        await driver.close()


# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------

def _local_drop(working_dir: str, dry_run: bool) -> None:
    p = Path(working_dir)
    if not p.exists():
        print(f"[drop] Local: directory not found: {working_dir}")
        return
    size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    print(f"[drop] Local: {working_dir}  ({size_mb:.1f} MB)")
    if not dry_run:
        shutil.rmtree(working_dir)
        print(f"[drop] Local: removed {working_dir}")
    else:
        print(f"[drop] Local: (dry-run) would rm -rf {working_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    tag = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{'='*60}")
    print(f"[drop] {tag}Dropping workspace: {args.workspace!r}")
    print(f"[drop] Working dir:  {args.working_dir}")
    print(f"{'='*60}\n")

    if not args.dry_run:
        confirm = input(
            f"This will permanently delete workspace '{args.workspace}' from Qdrant, Neo4j, and disk.\n"
            f"Type the workspace name to confirm: "
        ).strip()
        if confirm != args.workspace:
            print("[drop] Aborted — workspace name did not match.")
            sys.exit(1)

    print("\n[drop] Step 1/3: Qdrant")
    _qdrant_drop(args.workspace, args.dry_run)

    print("\n[drop] Step 2/3: Neo4j")
    await _neo4j_drop(args.workspace, args.dry_run)

    print("\n[drop] Step 3/3: Local files")
    _local_drop(args.working_dir, args.dry_run)

    print(f"\n[drop] {'(dry-run) ' if args.dry_run else ''}Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delete an evaluation workspace from all storage layers.")
    p.add_argument("--workspace",   required=True,
                   help="Workspace ID to delete (must match what was passed to build_index.py)")
    p.add_argument("--working-dir", required=True, dest="working_dir",
                   help="Local directory for this workspace (will be rm -rf'd)")
    p.add_argument("--dry-run",     action="store_true", dest="dry_run",
                   help="Print what would be deleted without actually deleting")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
