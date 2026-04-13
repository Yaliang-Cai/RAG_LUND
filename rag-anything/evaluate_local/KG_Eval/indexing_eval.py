import argparse
import asyncio
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import igraph as ig
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Optional in-file run config
# Set USE_INLINE_CONFIG=True to run without CLI args and use paths below.
# -----------------------------------------------------------------------------
USE_INLINE_CONFIG = True
INLINE_FRAMEWORK = "lightrag"  # choices: microsoft_graphrag, lightrag, fast_graphrag, hipporag2, graphml, neo4j
INLINE_BASE_PATH = ""          # e.g. /data/.../rag_workspaces
INLINE_FOLDER_NAME = None      # only needed for hipporag2
INLINE_OUTPUT = None           # e.g. /data/.../indexing_metrics.txt


SCRIPT_PATH = Path(__file__).resolve()
RAG_ANYTHING_ROOT = SCRIPT_PATH.parents[2]
LIGHTRAG_PROJECT_ROOT = RAG_ANYTHING_ROOT.parent / "lightrag"
for _path in (RAG_ANYTHING_ROOT, LIGHTRAG_PROJECT_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


WORKSPACE_MARKERS = (
    "kv_store_doc_status.json",
    "kv_store_full_docs.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
    "graph_chunk_entity_relation.graphml",
)


def analyze_graph(g: ig.Graph) -> Dict[str, float]:
    """
    Analyze a single graph object and calculate related metrics.

    :param g: igraph.Graph object
    :return: A dictionary containing the graph's related metrics
    """
    # Calculate various metrics
    num_nodes = g.vcount()
    num_edges = g.ecount()
    average_degree = sum(g.degree()) / num_nodes if num_nodes > 0 else 0
    density = g.density()
    components = g.components()
    num_components = len(components)
    largest_component_size = components.giant().vcount()
    average_clustering_coefficient = g.transitivity_avglocal_undirected()
    diameter = g.diameter() if g.is_connected() else float('inf')  # If graph is not connected, diameter is infinity

    # Calculate average connected component size (excluding isolated single nodes)
    component_sizes = [len(component) for component in components if len(component) > 1]
    if component_sizes:  # If there are non-isolated connected components
        average_component_size = sum(component_sizes) / len(component_sizes)
        median_component_size = np.median(component_sizes)  # Median of connected components
        num_components_excluding_isolated = len(component_sizes)  # Number of connected components excluding isolated entities
        num_components_above_average = sum(1 for size in component_sizes if size > average_component_size)  # Number of components above average
        num_nodes_excluding_isolated = sum(component_sizes)  # Number of entities excluding isolated ones

        # Calculate trimmed mean (excluding one highest and one lowest value)
        component_sizes_sorted = sorted(component_sizes)
        trimmed_mean_component_size = sum(component_sizes_sorted[1:-1]) / (len(component_sizes_sorted) - 2) if len(component_sizes_sorted) > 2 else average_component_size

        # Calculate geometric mean
        geometric_mean_component_size = np.exp(np.mean(np.log(component_sizes))) if len(component_sizes) > 0 else 0

        # Calculate harmonic mean
        harmonic_mean_component_size = len(component_sizes) / sum(1.0 / size for size in component_sizes) if len(component_sizes) > 0 else 0

    else:  # If all connected components are isolated single nodes
        average_component_size = 0
        median_component_size = 0
        num_components_excluding_isolated = 0
        num_components_above_average = 0
        num_nodes_excluding_isolated = 0
        trimmed_mean_component_size = 0
        geometric_mean_component_size = 0
        harmonic_mean_component_size = 0

    degrees = g.degree(mode="all")  # Use appropriate mode for directed graphs ("in", "out" or "all")

    num_isolated_nodes = sum(1 for d in degrees if d == 0)
    num_nodes_excluding_isolated = sum(1 for d in degrees if d > 0)

    num_nodes_degree_above_1 = sum(1 for d in degrees if d > 1)
    num_nodes_degree_above_2 = sum(1 for d in degrees if d > 2)
    num_nodes_degree_above_3 = sum(1 for d in degrees if d > 3)

    # Return results
    return {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "average_degree": float(average_degree),
        "density": float(density),
        "num_components": float(num_components),
        "largest_component_size": float(largest_component_size),
        "average_clustering_coefficient": float(average_clustering_coefficient),
        "diameter": float(diameter),
        "average_component_size": float(average_component_size),
        "median_component_size": float(median_component_size),
        "trimmed_mean_component_size": float(trimmed_mean_component_size),
        "geometric_mean_component_size": float(geometric_mean_component_size),
        "harmonic_mean_component_size": float(harmonic_mean_component_size),
        "num_components_excluding_isolated": float(num_components_excluding_isolated),
        "num_components_above_average": float(num_components_above_average),
        "num_nodes_excluding_isolated": float(num_nodes_excluding_isolated),
        "num_isolated_nodes": float(num_isolated_nodes),
        "num_nodes_degree_above_1": float(num_nodes_degree_above_1),
        "num_nodes_degree_above_2": float(num_nodes_degree_above_2),
        "num_nodes_degree_above_3": float(num_nodes_degree_above_3)
    }


def load_graph_from_parquet(entities_path: str, relationships_path: str) -> ig.Graph:
    """
    Load graph data from entities.parquet and relationships.parquet files and convert to igraph.Graph object.

    :param entities_path: Path to entities.parquet file
    :param relationships_path: Path to relationships.parquet file
    :return: igraph.Graph object
    """
    # Read entities.parquet file
    entities_df = pd.read_parquet(entities_path)
    
    # Read relationships.parquet file
    relationships_df = pd.read_parquet(relationships_path)
    
    # Create igraph graph object
    g = ig.Graph()

    # Add nodes
    for _, row in entities_df.iterrows():
        entity_id = row['id']  # Use 'id' column as unique identifier for nodes
        g.add_vertex(name=entity_id)  # Use entity's unique identifier as node name

    # Ensure all edge sources and targets are in the graph
    for _, row in relationships_df.iterrows():
        source_id = row['source']  # Use 'source' column as edge source
        target_id = row['target']  # Use 'target' column as edge target

        # Check if source and target are in the graph, add if not
        if source_id not in g.vs['name']:
            g.add_vertex(name=source_id)
        if target_id not in g.vs['name']:
            g.add_vertex(name=target_id)

        # Get edge weight, default to 1 if not present
        weight = row.get('weight', 1)
        g.add_edge(source_id, target_id, weight=weight)  # Add edge with weight as edge attribute

    return g


def load_graph_from_pickle(pickle_path: str) -> ig.Graph:
    """
    Load graph data from pickle file.

    :param pickle_path: Path to pickle file
    :return: igraph.Graph object
    """
    with open(pickle_path, 'rb') as f:
        g = pickle.load(f)
    return g


def load_graph_from_picklez(picklez_path: str) -> ig.Graph:
    """
    Load graph data from picklez file.

    :param picklez_path: Path to picklez file
    :return: igraph.Graph object
    """
    g = ig.Graph.Read_Picklez(picklez_path)
    return g


def load_graph_from_graphml(graphml_path: str) -> ig.Graph:
    """
    Load graph data from GraphML file.

    :param graphml_path: Path to GraphML file
    :return: igraph.Graph object
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = ig.Graph.Read_GraphML(graphml_path)
    return g


def _is_workspace_dir(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in WORKSPACE_MARKERS)


def _resolve_workspace_dirs(base_path: str, workspace_name: Optional[str] = None) -> List[Path]:
    base = Path(base_path).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base path does not exist: {base}")

    if workspace_name:
        candidate = (base / workspace_name).resolve()
        if _is_workspace_dir(candidate):
            return [candidate]
        raise FileNotFoundError(
            f"Workspace '{workspace_name}' not found under {base} or missing workspace marker files."
        )

    if _is_workspace_dir(base):
        return [base]

    workspaces = [path for path in sorted(base.iterdir()) if _is_workspace_dir(path)]
    return workspaces


def _build_igraph_from_neo4j_records(
    nodes: List[Dict[str, object]], edges: List[Dict[str, object]]
) -> ig.Graph:
    g = ig.Graph(directed=False)
    node_names: List[str] = []
    seen_nodes: set[str] = set()

    for node in nodes:
        entity_id = node.get("entity_id") or node.get("id")
        if entity_id is None:
            continue
        entity_name = str(entity_id)
        if entity_name not in seen_nodes:
            node_names.append(entity_name)
            seen_nodes.add(entity_name)

    if node_names:
        g.add_vertices(node_names)

    edge_pairs: List[tuple[str, str]] = []
    edge_weights: List[float] = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None:
            continue

        source_name = str(source)
        target_name = str(target)
        if source_name not in seen_nodes:
            g.add_vertex(name=source_name)
            seen_nodes.add(source_name)
        if target_name not in seen_nodes:
            g.add_vertex(name=target_name)
            seen_nodes.add(target_name)

        properties = edge.get("properties") or {}
        weight = 1.0
        if isinstance(properties, dict):
            try:
                weight = float(properties.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0

        edge_pairs.append((source_name, target_name))
        edge_weights.append(weight)

    if edge_pairs:
        g.add_edges(edge_pairs)
        g.es["weight"] = edge_weights

    return g


async def _load_graph_from_neo4j_workspace(workspace_dir: Path) -> ig.Graph:
    from lightrag.kg.neo4j_impl import Neo4JStorage
    from lightrag.namespace import NameSpace

    storage = Neo4JStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace=workspace_dir.name,
        global_config={"working_dir": str(workspace_dir)},
        embedding_func=None,
    )
    await storage.initialize()
    try:
        nodes = await storage.get_all_nodes()
        edges = await storage.get_all_edges()
    finally:
        await storage.finalize()

    return _build_igraph_from_neo4j_records(nodes, edges)


async def _process_graphs_neo4j_async(
    base_path: str, workspace_name: Optional[str] = None
) -> List[Dict]:
    results: List[Dict] = []
    for workspace_dir in _resolve_workspace_dirs(base_path, workspace_name):
        try:
            g = await _load_graph_from_neo4j_workspace(workspace_dir)
            result = analyze_graph(g)
            results.append(result)
        except Exception as e:
            print(f"Error processing Neo4j workspace {workspace_dir.name}: {e}")
    return results


def process_graphs_neo4j(base_path: str, workspace_name: Optional[str] = None) -> List[Dict]:
    """
    Process graph data stored in Neo4j, one workspace label per workspace directory.

    :param base_path: Workspaces root directory, or a single workspace directory
    :param workspace_name: Optional workspace directory name to evaluate
    :return: A list containing metric dictionaries for each graph
    """
    return asyncio.run(_process_graphs_neo4j_async(base_path, workspace_name))


def process_graphs_microsoft_graphrag(base_path: str, folder_name: str) -> List[Dict]:
    """
    Process graph data generated by Microsoft GraphRAG.

    :param base_path: Root path containing multiple subdirectories
    :param folder_name: Name of subdirectory containing graph data
    :return: A list containing metric dictionaries for each graph
    """
    results = []

    # Traverse each subdirectory under base_path
    for subdir, dirs, files in os.walk(base_path):
        entities_path = os.path.join(subdir, 'entities.parquet')
        relationships_path = os.path.join(subdir, 'relationships.parquet')
        if os.path.exists(entities_path) and os.path.exists(relationships_path):
            try:
                g = load_graph_from_parquet(entities_path, relationships_path)
                result = analyze_graph(g)
                results.append(result)
            except Exception as e:
                print(f"Error processing {subdir}: {e}")

    return results


def process_graphs_lightrag_fastgraphrag(base_path: str, folder_name: str) -> List[Dict]:
    """
    Process graph data generated by LightRAG and Fast-GraphRAG.

    :param base_path: Root path containing multiple subdirectories
    :param folder_name: Name of subdirectory containing graph data
    :return: A list containing metric dictionaries for each graph
    """
    results = []

    # Traverse each subdirectory under base_path
    for subdir, dirs, files in os.walk(base_path):
        # For LightRAG: look for graph_chunk_entity_relation.graphml files
        # For Fast-GraphRAG: look for graph_igraph_data.pklz files
        lightrag_path = os.path.join(subdir, 'graph_chunk_entity_relation.graphml')
        fastgraphrag_path = os.path.join(subdir, 'graph_igraph_data.pklz')
        
        if os.path.exists(lightrag_path):
            try:
                # Load graph from GraphML file (LightRAG)
                g = ig.Graph.Read_GraphML(lightrag_path)
                result = analyze_graph(g)
                results.append(result)
            except Exception as e:
                print(f"Error loading LightRAG graph from {lightrag_path}: {e}")
        elif os.path.exists(fastgraphrag_path):
            try:
                # Load graph from pickle file (Fast-GraphRAG)
                g = load_graph_from_picklez(fastgraphrag_path)
                result = analyze_graph(g)
                results.append(result)
            except Exception as e:
                print(f"Error loading Fast-GraphRAG graph from {fastgraphrag_path}: {e}")

    return results


def process_graphs_hipporag2(base_path: str, folder_name: str) -> List[Dict]:
    """
    Process graph data generated by HippoRAG2.

    :param base_path: Root path containing multiple subdirectories
    :param folder_name: Name of subdirectory containing graph data
    :return: A list containing metric dictionaries for each graph
    """
    results = []

    # Traverse each subdirectory under base_path
    for subdir, dirs, files in os.walk(base_path):
        target_folder = os.path.join(subdir, folder_name)
        if os.path.exists(target_folder):
            graph_path = os.path.join(target_folder, 'graph.pickle')
            if os.path.exists(graph_path):
                try:
                    g = load_graph_from_pickle(graph_path)
                    result = analyze_graph(g)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {subdir}: {e}")

    return results


def process_graphs_graphml(base_path: str, pattern: str = "*.graphml") -> List[Dict]:
    """
    Process graph data in GraphML format.

    :param base_path: Root path containing graph files
    :param pattern: File matching pattern
    :return: A list containing metric dictionaries for each graph
    """
    results = []

    # Traverse each subdirectory under base_path
    for subdir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.graphml'):
                graph_path = os.path.join(subdir, file)
                try:
                    g = load_graph_from_graphml(graph_path)
                    result = analyze_graph(g)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {graph_path}: {e}")

    return results


def calculate_average(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate average metrics for all graphs.

    :param results: A list containing metric dictionaries for each graph
    :return: A dictionary containing average values for all metrics
    """
    if not results:
        return {}

    # Initialize dictionary to store averages
    avg_results = {key: 0.0 for key in results[0].keys()}

    # Accumulate metrics from all graphs
    for result in results:
        for key, value in result.items():
            avg_results[key] += value

    # Calculate averages
    num_graphs = len(results)
    for key in avg_results:
        avg_results[key] /= num_graphs

    return avg_results


def _collect_indexing_results(framework: str, base_path: str, folder_name: Optional[str] = None) -> List[Dict]:
    if framework == 'microsoft_graphrag':
        return process_graphs_microsoft_graphrag(base_path, folder_name or "")
    elif framework in ['lightrag', 'fast_graphrag']:
        return process_graphs_lightrag_fastgraphrag(base_path, folder_name or "")
    elif framework == 'hipporag2':
        if not folder_name:
            raise ValueError("HippoRAG2 requires folder_name parameter")
        return process_graphs_hipporag2(base_path, folder_name)
    elif framework == 'graphml':
        return process_graphs_graphml(base_path)
    elif framework == 'neo4j':
        return process_graphs_neo4j(base_path, folder_name)
    raise ValueError(f"Unsupported framework: {framework}")


def calculate_indexing_metrics(framework: str, base_path: str, folder_name: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate indexing graph metrics for specified framework.

    :param framework: Framework name ('microsoft_graphrag', 'lightrag', 'fast_graphrag', 'hipporag2', 'graphml', 'neo4j')
    :param base_path: Root path containing graph data
    :param folder_name: Subdirectory name (required for some frameworks)
    :return: Average metrics dictionary
    """
    results = _collect_indexing_results(framework, base_path, folder_name)

    if not results:
        print(f"Warning: No graph data found for {framework} in {base_path}")
        return {}

    return calculate_average(results)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate indexing graph metrics for different GraphRAG frameworks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    required_cli = not USE_INLINE_CONFIG

    parser.add_argument(
        '--framework', 
        type=str, 
        required=required_cli,
        default=None,
        choices=['microsoft_graphrag', 'lightrag', 'fast_graphrag', 'hipporag2', 'graphml', 'neo4j'],
        help='Framework to analyze'
    )
    
    parser.add_argument(
        '--base_path', 
        type=str, 
        required=required_cli,
        default=None,
        help='Root path containing graph data'
    )
    
    parser.add_argument(
        '--folder_name', 
        type=str, 
        default=None,
        help='Subdirectory name (required for hipporag2; for neo4j it filters a single workspace name)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output file path (optional, prints to stdout if not specified)'
    )
    
    args = parser.parse_args()

    if USE_INLINE_CONFIG:
        args.framework = INLINE_FRAMEWORK
        args.base_path = INLINE_BASE_PATH
        args.folder_name = INLINE_FOLDER_NAME
        args.output = INLINE_OUTPUT

    if not args.framework or not args.base_path:
        raise ValueError(
            "framework/base_path is empty. Set CLI args, or set USE_INLINE_CONFIG=True "
            "and fill INLINE_FRAMEWORK/INLINE_BASE_PATH in indexing_eval.py."
        )

    return args


def main():
    """
    Main function for command line usage.
    """
    args = parse_args()
    
    try:
        print(f"Calculating indexing graph metrics for {args.framework}...")
        print(f"Base path: {args.base_path}")
        if args.folder_name:
            print(f"Folder name: {args.folder_name}")
        print()

        all_results = _collect_indexing_results(
            framework=args.framework,
            base_path=args.base_path,
            folder_name=args.folder_name,
        )
        metrics = calculate_average(all_results) if all_results else {}

        if metrics:
            output_lines = [
                f"Average metrics for {args.framework}:",
                f"  graph_count: {len(all_results)}",
            ]
            for key, value in metrics.items():
                output_lines.append(f"  {key}: {value:.4f}")
            
            output_text = "\n".join(output_lines)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_text)
                print(f"Results saved to {args.output}")
            else:
                print(output_text)
        else:
            print(f"No graph data found for {args.framework} in {args.base_path}")
            
    except Exception as e:
        print(f"Error calculating metrics for {args.framework}: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
