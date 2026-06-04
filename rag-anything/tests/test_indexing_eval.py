import sys
import types
import asyncio
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if "igraph" not in sys.modules:
    igraph_stub = types.ModuleType("igraph")

    class FakeGraph:
        def __init__(self, directed=False):
            self.directed = directed
            self.vertices = []
            self.edges = []
            self.es = {}

        def add_vertices(self, names):
            self.vertices.extend(names)

        def add_vertex(self, name):
            self.vertices.append(name)

        def add_edges(self, pairs):
            self.edges.extend(pairs)

    igraph_stub.Graph = FakeGraph
    sys.modules["igraph"] = igraph_stub

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.median = lambda values: values[0] if values else 0
    numpy_stub.exp = lambda value: value
    numpy_stub.mean = lambda values: sum(values) / len(values) if values else 0
    numpy_stub.log = lambda values: values
    sys.modules["numpy"] = numpy_stub

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.read_parquet = lambda *_args, **_kwargs: None
    sys.modules["pandas"] = pandas_stub


from evaluate_local.KG_Eval import indexing_eval


def test_cli_args_are_used_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "indexing_eval.py",
            "--framework",
            "neo4j",
            "--base_path",
            str(tmp_path),
            "--workspace-id",
            "docbench_shared_graphbm25_20260421_v0_v1_v2",
        ],
    )

    args = indexing_eval.parse_args()

    assert args.framework == "neo4j"
    assert args.base_path == str(tmp_path)
    assert args.folder_name == "docbench_shared_graphbm25_20260421_v0_v1_v2"


def test_cli_args_override_inline_config(monkeypatch, tmp_path):
    inline_path = tmp_path / "inline"
    cli_path = tmp_path / "cli"
    monkeypatch.setattr(indexing_eval, "USE_INLINE_CONFIG", True)
    monkeypatch.setattr(indexing_eval, "INLINE_FRAMEWORK", "lightrag")
    monkeypatch.setattr(indexing_eval, "INLINE_BASE_PATH", str(inline_path))
    monkeypatch.setattr(indexing_eval, "INLINE_FOLDER_NAME", "inline_ws")
    monkeypatch.setattr(indexing_eval, "INLINE_OUTPUT", str(inline_path / "out.txt"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "indexing_eval.py",
            "--framework",
            "neo4j",
            "--base_path",
            str(cli_path),
            "--workspace-id",
            "cli_ws",
            "--output",
            str(cli_path / "out.txt"),
        ],
    )

    args = indexing_eval.parse_args()

    assert args.framework == "neo4j"
    assert args.base_path == str(cli_path)
    assert args.folder_name == "cli_ws"
    assert args.output == str(cli_path / "out.txt")


def test_neo4j_workspace_resolution_allows_cache_only_workspace(tmp_path):
    workspace_id = "docbench_shared_graphbm25_20260421_v0_v1_v2"
    workspace_root = tmp_path / "rag_workspaces"
    cache_only_workspace = workspace_root / workspace_id
    (cache_only_workspace / workspace_id / "llm_cache").mkdir(parents=True)

    specs = indexing_eval._resolve_neo4j_workspace_specs(
        str(workspace_root),
        workspace_id,
    )

    assert len(specs) == 1
    assert specs[0].working_dir == cache_only_workspace.resolve()
    assert specs[0].workspace_id == workspace_id


def test_neo4j_workspace_resolution_collapses_repeated_workspace_dir(tmp_path):
    workspace_id = "docbench_shared_graphbm25_20260421_v0_v1_v2"
    outer_workspace = tmp_path / "rag_workspaces" / workspace_id
    repeated_workspace = outer_workspace / workspace_id
    (repeated_workspace / "llm_cache").mkdir(parents=True)

    specs = indexing_eval._resolve_neo4j_workspace_specs(str(repeated_workspace))

    assert len(specs) == 1
    assert specs[0].working_dir == outer_workspace.resolve()
    assert specs[0].workspace_id == workspace_id


def test_build_igraph_accepts_neo4j_node_and_edge_shapes():
    graph = indexing_eval._build_igraph_from_neo4j_records(
        nodes=[{"entity_id": "A"}, {"entity_id": "B"}],
        edges=[{"src": "A", "tgt": "B", "weight": 2.5}],
    )

    assert graph.vertices == ["A", "B"]
    assert graph.edges == [("A", "B")]
    assert graph.es["weight"] == [2.5]


def test_neo4j_loader_initializes_shared_storage_before_storage(monkeypatch, tmp_path):
    forbidden_modules = {
        "lightrag.kg.neo4j_impl",
        "lightrag.kg.shared_storage",
    }

    class ForbiddenModules(dict):
        def __contains__(self, key):
            if key in forbidden_modules:
                return False
            return super().__contains__(key)

        def __getitem__(self, key):
            if key in forbidden_modules:
                raise AssertionError(f"{key} must not be imported")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in forbidden_modules:
                raise AssertionError(f"{key} must not be imported")
            return super().get(key, default)

    monkeypatch.setattr(sys, "modules", ForbiddenModules(sys.modules))
    monkeypatch.setenv("NEO4J_URI", "bolt://example.invalid:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")

    namespace_stub = types.ModuleType("lightrag.namespace")
    namespace_stub.NameSpace = types.SimpleNamespace(
        GRAPH_STORE_CHUNK_ENTITY_RELATION="chunk_entity_relation"
    )
    monkeypatch.setitem(sys.modules, "lightrag.namespace", namespace_stub)

    events = []
    neo4j_stub = types.ModuleType("neo4j")

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        async def data(self):
            return self._rows

    class FakeSession:
        def __init__(self, database=None, default_access_mode=None):
            self.database = database
            self.default_access_mode = default_access_mode

        async def __aenter__(self):
            events.append(("session", self.database, self.default_access_mode))
            return self

        async def __aexit__(self, *_args):
            return False

        async def run(self, query):
            query_upper = query.upper()
            assert "CREATE " not in query_upper
            assert "MERGE " not in query_upper
            assert "DELETE " not in query_upper
            assert "`ws``x`" in query
            if "MATCH (N:" in query_upper:
                return FakeResult([{"entity_id": "A"}, {"entity_id": "B"}])
            return FakeResult([{"src": "A", "tgt": "B", "weight": 1.0}])

    class FakeDriver:
        def session(self, database=None, default_access_mode=None):
            return FakeSession(database=database, default_access_mode=default_access_mode)

        async def close(self):
            events.append("close")

    class FakeAsyncGraphDatabase:
        @staticmethod
        def driver(uri, auth=None, **_kwargs):
            events.append(("driver", uri, auth))
            return FakeDriver()

    neo4j_stub.AsyncGraphDatabase = FakeAsyncGraphDatabase
    monkeypatch.setitem(sys.modules, "neo4j", neo4j_stub)

    graph = asyncio.run(indexing_eval._load_graph_from_neo4j_workspace(tmp_path, "ws`x"))

    assert events == [
        ("driver", "bolt://example.invalid:7687", ("neo4j", "password")),
        ("session", "neo4j", "READ"),
        ("session", "neo4j", "READ"),
        "close",
    ]
    assert graph.edges == [("A", "B")]
