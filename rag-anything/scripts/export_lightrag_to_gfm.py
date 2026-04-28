#!/usr/bin/env python3
"""
Export LightRAG graph and chunks to GFM-RAG CSV format (Path B: pre-built index).

Usage:
    python scripts/export_lightrag_to_gfm.py \\
        --working-dir ./rag_storage/My_Graph \\
        --data-dir ./data \\
        --graph-name My_Graph \\
        --workspace My_Graph          # Neo4j workspace label (defaults to graph-name)

Output layout:
    <data-dir>/<graph-name>/processed/stage1/nodes.csv
    <data-dir>/<graph-name>/processed/stage1/edges.csv
    <data-dir>/<graph-name>/processed/stage1/relations.csv
    <data-dir>/<graph-name>/raw/documents.json

After running, set in .env:
    GFM_DATA_DIR=<data-dir>
    GFM_DATA_NAME=<graph-name>
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_chunks(working_dir: str) -> dict[str, str]:
    """Read all chunks from LightRAG JSON KV store.

    Returns {chunk_id: content_string}.
    """
    kv_path = Path(working_dir) / "kv_store_text_chunks.json"
    if not kv_path.exists():
        raise FileNotFoundError(
            f"LightRAG chunk KV store not found: {kv_path}\n"
            "Make sure --working-dir points to a LightRAG workspace directory."
        )
    with open(kv_path, encoding="utf-8") as f:
        raw = json.load(f)
    chunks: dict[str, str] = {}
    for chunk_id, chunk_data in raw.items():
        if isinstance(chunk_data, dict):
            content = chunk_data.get("content", "")
        else:
            content = str(chunk_data)
        chunks[chunk_id] = content
    logger.info("Loaded %d chunks from KV store", len(chunks))
    return chunks


def load_neo4j_graph(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    workspace: str,
) -> tuple[list[dict], list[dict]]:
    """Read entities and relations from Neo4j for the given workspace."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError(
            "neo4j Python driver not installed. Run: pip install neo4j"
        ) from exc

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    entities: list[dict] = []
    relations: list[dict] = []

    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {workspace: $ws}) "
                "RETURN e.entity_id AS entity_id, e.entity_name AS name, "
                "       e.description AS description, e.source_id AS source_ids",
                ws=workspace,
            )
            for record in result:
                entities.append(
                    {
                        "entity_id": record["entity_id"] or "",
                        "name": record["name"] or "",
                        "description": record["description"] or "",
                        "source_ids": record["source_ids"] or "",
                    }
                )
            logger.info("Loaded %d entities from Neo4j (workspace=%s)", len(entities), workspace)

            result = session.run(
                "MATCH (src:Entity {workspace: $ws})-[r:RELATES_TO]->(tgt:Entity {workspace: $ws}) "
                "RETURN src.entity_id AS src_id, r.relation AS relation, tgt.entity_id AS tgt_id",
                ws=workspace,
            )
            for record in result:
                relations.append(
                    {
                        "src": record["src_id"] or "",
                        "relation": record["relation"] or "",
                        "tgt": record["tgt_id"] or "",
                    }
                )
            logger.info("Loaded %d relations from Neo4j", len(relations))
    finally:
        driver.close()

    return entities, relations


def write_nodes_csv(out_dir: Path, chunks: dict[str, str], entities: list[dict]) -> None:
    """Write nodes.csv — header: name,type,attributes"""
    out_path = out_dir / "nodes.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "type", "attributes"])
        for chunk_id, content in chunks.items():
            attrs = json.dumps({"content": content}, ensure_ascii=False)
            writer.writerow([chunk_id, "document", attrs])
        for entity in entities:
            attrs = json.dumps({"description": entity["description"]}, ensure_ascii=False)
            writer.writerow([f"entity_{entity['name']}", "entity", attrs])
    logger.info(
        "Written %d document nodes + %d entity nodes → %s",
        len(chunks), len(entities), out_path,
    )


def write_edges_csv(
    out_dir: Path,
    entities: list[dict],
    relations: list[dict],
    chunks: dict[str, str],
) -> None:
    """Write edges.csv — header: source,relation,target,attributes"""
    out_path = out_dir / "edges.csv"
    entity_by_id: dict[str, str] = {e["entity_id"]: e["name"] for e in entities}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "relation", "target", "attributes"])

        for entity in entities:
            source_ids_str = entity.get("source_ids", "") or ""
            for chunk_id in source_ids_str.split("<SEP>"):
                chunk_id = chunk_id.strip()
                if chunk_id and chunk_id in chunks:
                    writer.writerow(
                        [f"entity_{entity['name']}", "mentioned_in", chunk_id, "{}"]
                    )

        for rel in relations:
            src_name = entity_by_id.get(rel["src"], rel["src"])
            tgt_name = entity_by_id.get(rel["tgt"], rel["tgt"])
            if src_name and tgt_name and rel["relation"]:
                writer.writerow(
                    [f"entity_{src_name}", rel["relation"], f"entity_{tgt_name}", "{}"]
                )

    logger.info("Written edges → %s", out_path)


def write_relations_csv(out_dir: Path, relations: list[dict]) -> None:
    """Write relations.csv — header: name,attributes"""
    out_path = out_dir / "relations.csv"
    unique_relations = sorted({rel["relation"] for rel in relations if rel["relation"]} | {"mentioned_in"})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "attributes"])
        for rel_name in unique_relations:
            writer.writerow([rel_name, "{}"])
    logger.info("Written %d relation types → %s", len(unique_relations), out_path)


def write_documents_json(raw_dir: Path, chunks: dict[str, str]) -> None:
    """Write raw/documents.json — {chunk_id: chunk_content}"""
    out_path = raw_dir / "documents.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Written %d documents → %s", len(chunks), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LightRAG KV store + Neo4j graph → GFM-RAG CSV index (Path B)"
    )
    parser.add_argument(
        "--working-dir", required=True,
        help="LightRAG workspace directory (contains kv_store_text_chunks.json)",
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="GFM-RAG root data directory (default: ./data)",
    )
    parser.add_argument(
        "--graph-name", required=True,
        help="Graph name — becomes GFM_DATA_NAME and the subdirectory name",
    )
    parser.add_argument(
        "--workspace", default=None,
        help="Neo4j workspace label (default: same as --graph-name)",
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    args = parser.parse_args()

    workspace = args.workspace or args.graph_name

    stage1_dir = Path(args.data_dir) / args.graph_name / "processed" / "stage1"
    raw_dir = Path(args.data_dir) / args.graph_name / "raw"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.working_dir)
    entities, relations = load_neo4j_graph(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password, workspace
    )

    write_nodes_csv(stage1_dir, chunks, entities)
    write_edges_csv(stage1_dir, entities, relations, chunks)
    write_relations_csv(stage1_dir, relations)
    write_documents_json(raw_dir, chunks)

    logger.info("Export complete.")
    logger.info("Set in .env:  GFM_DATA_DIR=%s  GFM_DATA_NAME=%s", args.data_dir, args.graph_name)


if __name__ == "__main__":
    main()
