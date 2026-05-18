import asyncio
import configparser
import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, List, final

import numpy as np
import pipmaster as pm

from ..base import BaseVectorStorage
from ..exceptions import DataMigrationError
from ..kg.shared_storage import get_data_init_lock
from ..utils import compute_mdhash_id, compute_entity_vdb_id, logger

if not pm.is_installed("qdrant-client"):
    pm.install("qdrant-client")

from qdrant_client import QdrantClient, models  # type: ignore

DEFAULT_WORKSPACE = "_"
WORKSPACE_ID_FIELD = "workspace_id"
ENTITY_PREFIX = "ent-"
CREATED_AT_FIELD = "created_at"
ID_FIELD = "id"

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

DEFAULT_ENABLE_SPARSE_BM25 = True
DEFAULT_SPARSE_BM25_MODEL = "Qdrant/bm25"
DEFAULT_QDRANT_UPSERT_MAX_PAYLOAD_BYTES = 24 * 1024 * 1024


def compute_mdhash_id_for_qdrant(
    content: str, prefix: str = "", style: str = "simple"
) -> str:
    """
    Generate a UUID based on the content and support multiple formats.

    :param content: The content used to generate the UUID.
    :param style: The format of the UUID, optional values are "simple", "hyphenated", "urn".
    :return: A UUID that meets the requirements of Qdrant.
    """
    if not content:
        raise ValueError("Content must not be empty.")

    # Use the hash value of the content to create a UUID.
    hashed_content = hashlib.sha256((prefix + content).encode("utf-8")).digest()
    generated_uuid = uuid.UUID(bytes=hashed_content[:16], version=4)

    # Return the UUID according to the specified format.
    if style == "simple":
        return generated_uuid.hex
    elif style == "hyphenated":
        return str(generated_uuid)
    elif style == "urn":
        return f"urn:uuid:{generated_uuid}"
    else:
        raise ValueError("Invalid style. Choose from 'simple', 'hyphenated', or 'urn'.")


def workspace_filter_condition(workspace: str) -> models.FieldCondition:
    """
    Create a workspace filter condition for Qdrant queries.
    """
    return models.FieldCondition(
        key=WORKSPACE_ID_FIELD, match=models.MatchValue(value=workspace)
    )


def _normalize_positive_int_env(
    env_name: str, default: int, *, minimum: int = 1
) -> int:
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r; falling back to %d", env_name, raw_value, default
        )
        return default
    if value < minimum:
        logger.warning(
            "Invalid %s=%r below minimum %d; falling back to %d",
            env_name,
            raw_value,
            minimum,
            default,
        )
        return default
    return value


def _qdrant_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _qdrant_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_qdrant_jsonable(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _qdrant_jsonable(model_dump(mode="json"))
        except TypeError:
            return _qdrant_jsonable(model_dump())
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _qdrant_jsonable(as_dict())
    return value


def _estimate_qdrant_point_json_bytes(point: models.PointStruct) -> int:
    """Conservatively estimate one point's JSON size in Qdrant's upsert body."""
    point_payload = {
        "id": _qdrant_jsonable(point.id),
        "vector": _qdrant_jsonable(point.vector),
        "payload": _qdrant_jsonable(point.payload),
    }
    try:
        return len(
            json.dumps(
                point_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
    except Exception:
        # Fall back to a conservative estimate when qdrant-client changes
        # internal model shapes in a way our converter does not recognize.
        dense_vector = point.vector
        if isinstance(dense_vector, dict):
            dense_vector = dense_vector.get("", [])
        vector_len = len(dense_vector) if hasattr(dense_vector, "__len__") else 0
        payload_size = len(str(point.payload).encode("utf-8", errors="ignore"))
        return payload_size + vector_len * 24 + 1024


def _iter_qdrant_point_batches_by_payload(
    points: list[models.PointStruct], max_payload_bytes: int
) -> list[list[models.PointStruct]]:
    if not points:
        return []

    batches: list[list[models.PointStruct]] = []
    current_batch: list[models.PointStruct] = []
    # Account for the request JSON wrapper and commas around points.
    current_size = 1024

    for point in points:
        point_size = _estimate_qdrant_point_json_bytes(point) + 4
        if current_batch and current_size + point_size > max_payload_bytes:
            batches.append(current_batch)
            current_batch = []
            current_size = 1024

        current_batch.append(point)
        current_size += point_size

        if len(current_batch) == 1 and point_size > max_payload_bytes:
            logger.warning(
                "Single Qdrant point payload is estimated at %.2f MiB, above configured %.2f MiB limit",
                point_size / (1024 * 1024),
                max_payload_bytes / (1024 * 1024),
            )
            batches.append(current_batch)
            current_batch = []
            current_size = 1024

    if current_batch:
        batches.append(current_batch)
    return batches


def _coerce_qdrant_dense_vector(vector_data: Any) -> list[float] | None:
    """Return the dense vector from Qdrant's vector payload.

    Qdrant returns a plain vector for dense-only collections, but returns a
    mapping for hybrid dense+sparse collections.  LightRAG's vector storage
    interface expects get_vectors_by_ids() to expose only the dense vector.
    """
    if isinstance(vector_data, dict):
        dense_vector = vector_data.get("")
        if dense_vector is None:
            for candidate in vector_data.values():
                if isinstance(candidate, np.ndarray):
                    dense_vector = candidate
                    break
                if isinstance(candidate, (list, tuple)):
                    dense_vector = candidate
                    break
        vector_data = dense_vector

    if isinstance(vector_data, np.ndarray):
        vector_data = vector_data.tolist()
    elif isinstance(vector_data, tuple):
        vector_data = list(vector_data)

    if not isinstance(vector_data, list):
        return None

    try:
        return [float(value) for value in vector_data]
    except (TypeError, ValueError):
        return None


def _normalize_qdrant_retrieval_mode(mode: Any) -> str:
    """Normalize Qdrant retrieval mode to a supported value."""
    normalized = str(mode or os.environ.get("QDRANT_RETRIEVAL_MODE", "dense")).lower()
    if normalized not in {"dense", "bm25", "hybrid"}:
        logger.warning(
            "Invalid Qdrant retrieval mode %r; falling back to dense", mode
        )
        return "dense"
    return normalized


def _coerce_dense_query_vector(vector_data: Any) -> list[float]:
    """Convert dense query vectors to plain float lists for Qdrant models."""
    if isinstance(vector_data, np.ndarray):
        vector_data = vector_data.tolist()
    elif isinstance(vector_data, tuple):
        vector_data = list(vector_data)

    return [float(value) for value in vector_data]


def _coerce_qdrant_sparse_vector(vector_data: Any) -> models.SparseVector:
    """Convert fastembed sparse output to Qdrant's SparseVector model."""
    if isinstance(vector_data, models.SparseVector):
        return vector_data

    indices = getattr(vector_data, "indices", None)
    values = getattr(vector_data, "values", None)
    if indices is None or values is None:
        raise ValueError(f"Unsupported sparse vector type: {type(vector_data).__name__}")

    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if hasattr(values, "tolist"):
        values = values.tolist()

    return models.SparseVector(
        indices=[int(index) for index in indices],
        values=[float(value) for value in values],
    )


def _normalize_timeout_seconds(raw_value: Any, env_name: str) -> int:
    """Parse timeout input and normalize to positive integer seconds."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{env_name} must be a numeric timeout in seconds, got {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(f"{env_name} must be > 0, got {value}")

    return max(1, int(math.ceil(value)))


def _find_legacy_collection(
    client: QdrantClient,
    namespace: str,
    workspace: str = None,
    model_suffix: str = None,
) -> str | None:
    """
    Find legacy collection with backward compatibility support.

    This function tries multiple naming patterns to locate legacy collections
    created by older versions of LightRAG:

    1. lightrag_vdb_{namespace} - if model_suffix is provided (HIGHEST PRIORITY)
    2. {workspace}_{namespace} or {namespace} - no matter if model_suffix is provided or not
    3. lightrag_vdb_{namespace} - fall back value no matter if model_suffix is provided or not (LOWEST PRIORITY)

    Args:
        client: QdrantClient instance
        namespace: Base namespace (e.g., "chunks", "entities")
        workspace: Optional workspace identifier
        model_suffix: Optional model suffix for new collection

    Returns:
        Collection name if found, None otherwise
    """
    # Try multiple naming patterns for backward compatibility
    # More specific names (with workspace) have higher priority
    candidates = [
        f"lightrag_vdb_{namespace}" if model_suffix else None,
        f"{workspace}_{namespace}" if workspace else None,
        f"lightrag_vdb_{namespace}",
        namespace,
    ]

    for candidate in candidates:
        # Skip candidates containing path separators — a workspace that is a
        # filesystem path (e.g. "/data/foo/bar") would produce a name with "/"
        # which makes the Qdrant REST URL malformed and returns 404.
        if candidate and "/" not in candidate and "\\" not in candidate and client.collection_exists(candidate):
            logger.info(
                f"Qdrant: Found legacy collection '{candidate}' "
                f"(namespace={namespace}, workspace={workspace or 'none'})"
            )
            return candidate

    return None


@final
@dataclass
class QdrantVectorDBStorage(BaseVectorStorage):
    def __init__(
        self, namespace, global_config, embedding_func, workspace=None, meta_fields=None
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self.__post_init__()

    @staticmethod
    def setup_collection(
        client: QdrantClient,
        collection_name: str,
        namespace: str,
        workspace: str,
        vectors_config: models.VectorParams,
        hnsw_config: models.HnswConfigDiff,
        model_suffix: str,
        quantization_config=None,
        index_relation_fields: bool = False,
        sparse_vectors_config: dict[str, models.SparseVectorParams] | None = None,
    ):
        """
        Setup Qdrant collection with migration support from legacy collections.

        Ensure final collection has workspace isolation index.
        Check vector dimension compatibility before new collection creation.
        Drop legacy collection if it exists and is empty.
        Only migrate data from legacy collection to new collection when new collection first created and legacy collection is not empty.

        Args:
            client: QdrantClient instance
            collection_name: Name of the final collection
            namespace: Base namespace (e.g., "chunks", "entities")
            workspace: Workspace identifier for data isolation
            vectors_config: Vector configuration parameters for the collection
            hnsw_config: HNSW index configuration diff for the collection
        """
        if not namespace or not workspace:
            raise ValueError("namespace and workspace must be provided")

        workspace_count_filter = models.Filter(
            must=[workspace_filter_condition(workspace)]
        )

        new_collection_exists = client.collection_exists(collection_name)
        legacy_collection = _find_legacy_collection(
            client, namespace, workspace, model_suffix
        )

        # Case 1: Only new collection exists or  new collection is the same as legacy collection
        #         No data migration needed,  and ensuring index is created then return
        if (new_collection_exists and not legacy_collection) or (
            collection_name == legacy_collection
        ):
            if sparse_vectors_config:
                try:
                    client.update_collection(
                        collection_name=collection_name,
                        sparse_vectors_config=sparse_vectors_config,
                    )
                except Exception as e:
                    logger.warning(
                        f"Qdrant: Failed to update sparse_vectors_config for existing collection '{collection_name}': {e}"
                    )
            # create_payload_index return without error if index already exists
            client.create_payload_index(
                collection_name=collection_name,
                field_name=WORKSPACE_ID_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
            if index_relation_fields:
                for field in ("src_id", "tgt_id"):
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=models.KeywordIndexParams(
                            type=models.KeywordIndexType.KEYWORD,
                        ),
                    )
            new_workspace_count = client.count(
                collection_name=collection_name,
                count_filter=workspace_count_filter,
                exact=True,
            ).count

            # Skip data migration if new collection already has workspace data
            if new_workspace_count == 0 and not (collection_name == legacy_collection):
                logger.warning(
                    f"Qdrant: workspace data in collection '{collection_name}' is empty. "
                    f"Ensure it is caused by new workspace setup and not an unexpected embedding model change."
                )

            return

        legacy_count = None
        if not new_collection_exists:
            # Check vector dimension compatibility before creating new collection
            if legacy_collection:
                legacy_count = client.count(
                    collection_name=legacy_collection, exact=True
                ).count
                if legacy_count > 0:
                    legacy_info = client.get_collection(legacy_collection)
                    legacy_dim = legacy_info.config.params.vectors.size

                    if vectors_config.size and legacy_dim != vectors_config.size:
                        logger.error(
                            f"Qdrant: Dimension mismatch detected! "
                            f"Legacy collection '{legacy_collection}' has {legacy_dim}d vectors, "
                            f"but new embedding model expects {vectors_config.size}d."
                        )

                        raise DataMigrationError(
                            f"Dimension mismatch between legacy collection '{legacy_collection}' "
                            f"and new collection. Expected {vectors_config.size}d but got {legacy_dim}d."
                        )

            client.create_collection(
                collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
            )
            logger.info(f"Qdrant: Collection '{collection_name}' created successfully")
            if not legacy_collection:
                logger.warning(
                    "Qdrant: Ensure this new collection creation is caused by new workspace setup and not an unexpected embedding model change."
                )

        # create_payload_index return without error if index already exists
        client.create_payload_index(
            collection_name=collection_name,
            field_name=WORKSPACE_ID_FIELD,
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )
        if index_relation_fields:
            for field in ("src_id", "tgt_id"):
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                    ),
                )

        # Case 2: Legacy collection exist
        if legacy_collection:
            # Only drop legacy collection if it's empty
            if legacy_count is None:
                legacy_count = client.count(
                    collection_name=legacy_collection, exact=True
                ).count
            if legacy_count == 0:
                client.delete_collection(collection_name=legacy_collection)
                logger.info(
                    f"Qdrant: Empty legacy collection '{legacy_collection}' deleted successfully"
                )
                return

            new_workspace_count = client.count(
                collection_name=collection_name,
                count_filter=workspace_count_filter,
                exact=True,
            ).count

            # Skip data migration if new collection already has workspace data
            if new_workspace_count > 0:
                logger.warning(
                    f"Qdrant: Both new and legacy collection have data. "
                    f"{legacy_count} records in {legacy_collection} require manual deletion after migration verification."
                )
                return

            # Case 3: Only legacy exists - migrate data from legacy collection to new collection
            # Check if legacy collection has workspace_id to determine migration strategy
            # Note: payload_schema only reflects INDEXED fields, so we also sample
            # actual payloads to detect unindexed workspace_id fields
            legacy_info = client.get_collection(legacy_collection)
            has_workspace_index = WORKSPACE_ID_FIELD in (
                legacy_info.payload_schema or {}
            )

            # Detect workspace_id field presence by sampling payloads if not indexed
            # This prevents cross-workspace data leakage when workspace_id exists but isn't indexed
            has_workspace_field = has_workspace_index
            if not has_workspace_index:
                # Sample a small batch of points to check for workspace_id in payloads
                # All points must have workspace_id if any point has it
                sample_result = client.scroll(
                    collection_name=legacy_collection,
                    limit=10,  # Small sample is sufficient for detection
                    with_payload=True,
                    with_vectors=False,
                )
                sample_points, _ = sample_result
                for point in sample_points:
                    if point.payload and WORKSPACE_ID_FIELD in point.payload:
                        has_workspace_field = True
                        logger.info(
                            f"Qdrant: Detected unindexed {WORKSPACE_ID_FIELD} field "
                            f"in legacy collection '{legacy_collection}' via payload sampling"
                        )
                        break

            # Build workspace filter if legacy collection has workspace support
            # This prevents cross-workspace data leakage during migration
            legacy_scroll_filter = None
            if has_workspace_field:
                legacy_scroll_filter = models.Filter(
                    must=[workspace_filter_condition(workspace)]
                )
                # Recount with workspace filter for accurate migration tracking
                legacy_count = client.count(
                    collection_name=legacy_collection,
                    count_filter=legacy_scroll_filter,
                    exact=True,
                ).count
                logger.info(
                    f"Qdrant: Legacy collection has workspace support, "
                    f"filtering to {legacy_count} records for workspace '{workspace}'"
                )

            logger.info(
                f"Qdrant: Found legacy collection '{legacy_collection}' with {legacy_count} records to migrate."
            )
            logger.info(
                f"Qdrant: Migrating data from legacy collection '{legacy_collection}' to new collection '{collection_name}'"
            )

            try:
                # Batch migration (500 records per batch)
                migrated_count = 0
                offset = None
                batch_size = 500

                while True:
                    # Scroll through legacy data with optional workspace filter
                    result = client.scroll(
                        collection_name=legacy_collection,
                        scroll_filter=legacy_scroll_filter,
                        limit=batch_size,
                        offset=offset,
                        with_vectors=True,
                        with_payload=True,
                    )
                    points, next_offset = result

                    if not points:
                        break

                    # Transform points for new collection
                    new_points = []
                    for point in points:
                        # Set workspace_id in payload
                        new_payload = dict(point.payload or {})
                        new_payload[WORKSPACE_ID_FIELD] = workspace

                        # Create new point with workspace-prefixed ID
                        original_id = new_payload.get(ID_FIELD)
                        if original_id:
                            new_point_id = compute_mdhash_id_for_qdrant(
                                original_id, prefix=workspace
                            )
                        else:
                            # Fallback: use original point ID
                            new_point_id = str(point.id)

                        new_points.append(
                            models.PointStruct(
                                id=new_point_id,
                                vector=point.vector,
                                payload=new_payload,
                            )
                        )

                    # Upsert to new collection
                    client.upsert(
                        collection_name=collection_name, points=new_points, wait=True
                    )

                    migrated_count += len(points)
                    logger.info(
                        f"Qdrant: {migrated_count}/{legacy_count} records migrated"
                    )

                    # Check if we've reached the end
                    if next_offset is None:
                        break
                    offset = next_offset

                new_count_after = client.count(
                    collection_name=collection_name,
                    count_filter=workspace_count_filter,
                    exact=True,
                ).count
                inserted_count = new_count_after - new_workspace_count
                if inserted_count != legacy_count:
                    error_msg = (
                        "Qdrant: Migration verification failed, expected "
                        f"{legacy_count} inserted records, got {inserted_count}."
                    )
                    logger.error(error_msg)
                    raise DataMigrationError(error_msg)

            except DataMigrationError:
                # Re-raise DataMigrationError as-is to preserve specific error messages
                raise
            except Exception as e:
                logger.error(
                    f"Qdrant: Failed to migrate data from legacy collection '{legacy_collection}' to new collection '{collection_name}': {e}"
                )
                raise DataMigrationError(
                    f"Failed to migrate data from legacy collection '{legacy_collection}' to new collection '{collection_name}'"
                ) from e

            logger.info(
                f"Qdrant: Migration from '{legacy_collection}' to '{collection_name}' completed successfully"
            )
            logger.warning(
                "Qdrant: Manual deletion is required after data migration verification."
            )

    def __post_init__(self):
        self._validate_embedding_func()
        # Check for QDRANT_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all Qdrant storage instances
        qdrant_workspace = os.environ.get("QDRANT_WORKSPACE")
        if qdrant_workspace and qdrant_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = qdrant_workspace.strip()
            logger.info(
                f"Using QDRANT_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        self.effective_workspace = effective_workspace or DEFAULT_WORKSPACE

        # Generate model suffix
        self.model_suffix = self._generate_collection_suffix()

        # Read BM25 flag here so the collection name can include it.
        # Collections built with and without BM25 are physically separate:
        # BM25-enabled collections carry a _bm25 suffix, preventing schema
        # conflicts when the same Qdrant server is used for both modes.
        self._enable_sparse_bm25 = (
            os.environ.get(
                "QDRANT_ENABLE_SPARSE_BM25", str(DEFAULT_ENABLE_SPARSE_BM25)
            ).lower()
            in {"1", "true", "yes", "y", "on"}
        )
        _bm25_suffix = "_bm25" if self._enable_sparse_bm25 else ""

        # New naming scheme with model isolation
        # BM25 off:  "lightrag_vdb_chunks_text_embedding_ada_002_1536d"
        # BM25 on:   "lightrag_vdb_chunks_text_embedding_ada_002_1536d_bm25"
        if self.model_suffix:
            self.final_namespace = f"lightrag_vdb_{self.namespace}_{self.model_suffix}{_bm25_suffix}"
            logger.info(f"Qdrant collection: {self.final_namespace}")
        else:
            # Fallback: use legacy namespace if model_suffix is unavailable
            self.final_namespace = f"lightrag_vdb_{self.namespace}{_bm25_suffix}"
            logger.warning(
                f"Qdrant collection: {self.final_namespace} missing suffix. Pls add model_name to embedding_func for proper workspace data isolation."
            )

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Initialize client as None - will be created in initialize() method
        self._client = None
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._max_upsert_payload_bytes = _normalize_positive_int_env(
            "QDRANT_UPSERT_MAX_PAYLOAD_BYTES",
            DEFAULT_QDRANT_UPSERT_MAX_PAYLOAD_BYTES,
            minimum=1024 * 1024,
        )
        self._initialized = False
        self._client_timeout = _normalize_timeout_seconds(
            os.environ.get(
                "QDRANT_CLIENT_TIMEOUT",
                config.get("qdrant", "timeout", fallback=1200),
            ),
            "QDRANT_CLIENT_TIMEOUT",
        )
        self._operation_timeout = _normalize_timeout_seconds(
            os.environ.get("QDRANT_OPERATION_TIMEOUT", self._client_timeout),
            "QDRANT_OPERATION_TIMEOUT",
        )

        # --- Tuning options from environment variables ---
        # Distance metric
        _dist_map = {
            "COSINE": models.Distance.COSINE,
            "DOT": models.Distance.DOT,
            "EUCLID": models.Distance.EUCLID,
            "MANHATTAN": models.Distance.MANHATTAN,
        }
        self._distance = _dist_map.get(
            os.environ.get("QDRANT_DISTANCE", "COSINE").upper(),
            models.Distance.COSINE,
        )

        # HNSW build parameters
        _m_env = os.environ.get("QDRANT_HNSW_M")
        self._hnsw_m = int(_m_env) if _m_env else None
        _ef_env = os.environ.get("QDRANT_HNSW_EF_CONSTRUCT")
        self._hnsw_ef_construct = int(_ef_env) if _ef_env else None
        self._hnsw_on_disk = os.environ.get("QDRANT_HNSW_ON_DISK", "false").lower() == "true"

        # Query-time HNSW ef (accuracy vs speed)
        _sef_env = os.environ.get("QDRANT_SEARCH_EF")
        self._search_ef = int(_sef_env) if _sef_env else None

        # Quantization
        _quant = os.environ.get("QDRANT_QUANTIZATION", "none").lower()
        if _quant == "scalar":
            self._quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8)
            )
        elif _quant == "binary":
            self._quantization_config = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig()
            )
        else:
            self._quantization_config = None

        # gRPC transport
        self._prefer_grpc = os.environ.get("QDRANT_PREFER_GRPC", "false").lower() == "true"

        # Build keyword indexes on src_id / tgt_id for faster relation deletion
        self._index_relation_fields = (
            os.environ.get("QDRANT_INDEX_RELATION_FIELDS", "false").lower() == "true"
        )
        self._sparse_bm25_model = os.environ.get(
            "QDRANT_SPARSE_BM25_MODEL", DEFAULT_SPARSE_BM25_MODEL
        )
        self._sparse_vector_name: str | None = None

    async def initialize(self):
        """Initialize Qdrant collection"""
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                # Create QdrantClient if not already created
                if self._client is None:
                    self._client = QdrantClient(
                        url=os.environ.get(
                            "QDRANT_URL", config.get("qdrant", "uri", fallback=None)
                        ),
                        api_key=os.environ.get(
                            "QDRANT_API_KEY",
                            config.get("qdrant", "apikey", fallback=None),
                        ),
                        prefer_grpc=self._prefer_grpc,
                        timeout=self._client_timeout,
                    )
                    logger.debug(
                        f"[{self.workspace}] QdrantClient created successfully"
                    )

                sparse_vectors_config: dict[str, models.SparseVectorParams] | None = None
                if self._enable_sparse_bm25:
                    try:
                        if not pm.is_installed("fastembed"):
                            pm.install("fastembed")
                        self._client.set_sparse_model(self._sparse_bm25_model)
                        self._sparse_vector_name = (
                            self._client.get_sparse_vector_field_name()
                        )
                        sparse_vectors_config = (
                            self._client.get_fastembed_sparse_vector_params(
                                modifier=models.Modifier.IDF
                            )
                        )
                        logger.info(
                            "[%s] Enabled sparse BM25 indexing in Qdrant (model=%s, field=%s)",
                            self.workspace,
                            self._sparse_bm25_model,
                            self._sparse_vector_name,
                        )
                    except Exception as e:
                        self._enable_sparse_bm25 = False
                        self._sparse_vector_name = None
                        logger.warning(
                            "[%s] Failed to enable sparse BM25 indexing; fallback to dense-only. reason=%s",
                            self.workspace,
                            e,
                        )

                # Setup collection (create if not exists and configure indexes)
                # Pass namespace and workspace for backward-compatible migration support
                QdrantVectorDBStorage.setup_collection(
                    self._client,
                    self.final_namespace,
                    namespace=self.namespace,
                    workspace=self.effective_workspace,
                    vectors_config=models.VectorParams(
                        size=self.embedding_func.embedding_dim,
                        distance=self._distance,
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=self._hnsw_m,
                        ef_construct=self._hnsw_ef_construct,
                        on_disk=self._hnsw_on_disk or None,
                    ),
                    model_suffix=self.model_suffix,
                    quantization_config=self._quantization_config,
                    index_relation_fields=self._index_relation_fields,
                    sparse_vectors_config=sparse_vectors_config,
                )

                self._initialized = True
                logger.info(
                    f"[{self.workspace}] Qdrant collection '{self.namespace}' initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to initialize Qdrant collection '{self.namespace}': {e}"
                )
                raise

    async def _run_client_call_with_timeout(self, method, *args, **kwargs):
        """
        Run blocking qdrant-client calls in a worker thread.

        Prefer per-call timeout, but gracefully fall back for older qdrant-client
        versions that don't accept the `timeout` keyword.
        """
        kwargs_with_timeout = dict(kwargs)
        kwargs_with_timeout.setdefault("timeout", self._operation_timeout)
        try:
            return await asyncio.to_thread(method, *args, **kwargs_with_timeout)
        except TypeError as exc:
            error_text = str(exc)
            if "unexpected keyword argument 'timeout'" not in error_text:
                raise
            return await asyncio.to_thread(method, *args, **kwargs)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        import time

        current_time = int(time.time())

        list_data = [
            {
                ID_FIELD: k,
                WORKSPACE_ID_FIELD: self.effective_workspace,
                CREATED_AT_FIELD: current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)

        sparse_vectors: list[models.SparseVector] | None = None
        if self._enable_sparse_bm25 and self._sparse_vector_name:
            try:
                sparse_vectors = list(
                    self._client._sparse_embed_documents(
                        contents,
                        embedding_model_name=self._sparse_bm25_model,
                        batch_size=self._max_batch_size,
                    )
                )
            except Exception as e:
                logger.warning(
                    "[%s] Sparse BM25 embedding failed for this batch; fallback dense-only. reason=%s",
                    self.workspace,
                    e,
                )
                sparse_vectors = None

        list_points = []
        for i, d in enumerate(list_data):
            point_vector: list[float] | dict[str, Any]
            if sparse_vectors is not None and self._sparse_vector_name:
                point_vector = {
                    "": embeddings[i],
                    self._sparse_vector_name: sparse_vectors[i],
                }
            else:
                point_vector = embeddings[i]
            list_points.append(
                models.PointStruct(
                    id=compute_mdhash_id_for_qdrant(
                        d[ID_FIELD], prefix=self.effective_workspace
                    ),
                    vector=point_vector,
                    payload=d,
                )
            )

        point_batches = _iter_qdrant_point_batches_by_payload(
            list_points, self._max_upsert_payload_bytes
        )
        if len(point_batches) > 1:
            logger.info(
                "[%s] Splitting Qdrant upsert for %s into %d requests below %.2f MiB",
                self.workspace,
                self.namespace,
                len(point_batches),
                self._max_upsert_payload_bytes / (1024 * 1024),
            )

        results = None
        for point_batch in point_batches:
            results = await self._run_client_call_with_timeout(
                self._client.upsert,
                collection_name=self.final_namespace,
                points=point_batch,
                wait=True,
            )
        return results

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] = None,
        qdrant_retrieval_mode: str | None = None,
        candidate_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding_result = await self.embedding_func(
                [query], _priority=5
            )  # higher priority for query
            embedding = embedding_result[0]

        embedding = _coerce_dense_query_vector(embedding)
        retrieval_mode = _normalize_qdrant_retrieval_mode(
            qdrant_retrieval_mode
            or self.global_config.get("qdrant_retrieval_mode")
        )
        filter_conditions: list[Any] = [
            workspace_filter_condition(self.effective_workspace)
        ]
        if candidate_ids is not None:
            normalized_candidate_ids = [
                str(candidate_id).strip()
                for candidate_id in candidate_ids
                if str(candidate_id).strip()
            ]
            if not normalized_candidate_ids:
                return []
            qdrant_candidate_ids = [
                compute_mdhash_id_for_qdrant(
                    candidate_id,
                    prefix=self.effective_workspace,
                )
                for candidate_id in dict.fromkeys(normalized_candidate_ids)
            ]
            filter_conditions.append(
                models.HasIdCondition(has_id=qdrant_candidate_ids)
            )
        query_filter = models.Filter(must=filter_conditions)
        search_params = (
            models.SearchParams(hnsw_ef=self._search_ef)
            if self._search_ef
            else None
        )

        if retrieval_mode == "bm25":
            query_response = await self._query_sparse(
                query=query,
                top_k=top_k,
                query_filter=query_filter,
            )
        elif retrieval_mode == "hybrid":
            query_response = await self._query_hybrid(
                query=query,
                embedding=embedding,
                top_k=top_k,
                query_filter=query_filter,
                search_params=search_params,
            )
        else:
            query_response = await self._query_dense(
                embedding=embedding,
                top_k=top_k,
                query_filter=query_filter,
                search_params=search_params,
            )

        results = query_response.points

        return [
            {
                **dp.payload,
                "distance": dp.score,
                CREATED_AT_FIELD: dp.payload.get(CREATED_AT_FIELD),
            }
            for dp in results
        ]

    async def _query_dense(
        self,
        embedding: list[float],
        top_k: int,
        query_filter: models.Filter,
        search_params: models.SearchParams | None,
    ):
        return await self._run_client_call_with_timeout(
            self._client.query_points,
            collection_name=self.final_namespace,
            query=embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=self.cosine_better_than_threshold,
            query_filter=query_filter,
            search_params=search_params,
        )

    async def _query_sparse(
        self,
        query: str,
        top_k: int,
        query_filter: models.Filter,
    ):
        if not self._enable_sparse_bm25 or not self._sparse_vector_name:
            logger.warning(
                "[%s] Qdrant BM25 retrieval requested but sparse BM25 is not enabled; falling back to dense retrieval is required by caller",
                self.workspace,
            )
            raise ValueError("Qdrant BM25 retrieval requires sparse BM25 indexing")

        sparse_query = self._embed_sparse_query(query)
        return await self._run_client_call_with_timeout(
            self._client.query_points,
            collection_name=self.final_namespace,
            query=sparse_query,
            using=self._sparse_vector_name,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

    async def _query_hybrid(
        self,
        query: str,
        embedding: list[float],
        top_k: int,
        query_filter: models.Filter,
        search_params: models.SearchParams | None,
    ):
        if not self._enable_sparse_bm25 or not self._sparse_vector_name:
            logger.warning(
                "[%s] Qdrant hybrid retrieval requested but sparse BM25 is not enabled; falling back to dense retrieval",
                self.workspace,
            )
            return await self._query_dense(
                embedding=embedding,
                top_k=top_k,
                query_filter=query_filter,
                search_params=search_params,
            )

        sparse_query = self._embed_sparse_query(query)
        prefetch = [
            models.Prefetch(
                query=embedding,
                filter=query_filter,
                params=search_params,
                limit=top_k,
            ),
            models.Prefetch(
                query=sparse_query,
                using=self._sparse_vector_name,
                filter=query_filter,
                limit=top_k,
            ),
        ]
        return await self._run_client_call_with_timeout(
            self._client.query_points,
            collection_name=self.final_namespace,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

    def _embed_sparse_query(self, query: str) -> models.SparseVector:
        try:
            sparse_model = self._client._get_or_init_sparse_model(
                model_name=self._sparse_bm25_model
            )
            sparse_vector = list(sparse_model.query_embed(query=query))[0]
        except Exception:
            sparse_vector = list(
                self._client._sparse_embed_documents(
                    [query],
                    embedding_model_name=self._sparse_bm25_model,
                    batch_size=1,
                )
            )[0]

        return _coerce_qdrant_sparse_vector(sparse_vector)

    async def index_done_callback(self) -> None:
        # Qdrant handles persistence automatically
        pass

    async def delete(self, ids: List[str]) -> None:
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            if not ids:
                return

            # Convert regular ids to Qdrant compatible ids
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=self.effective_workspace)
                for id in ids
            ]
            # Delete points from the collection with workspace filtering
            await self._run_client_call_with_timeout(
                self._client.delete,
                collection_name=self.final_namespace,
                points_selector=models.PointIdsList(points=qdrant_ids),
                wait=True,
            )
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )

    async def delete_entity(self, entity_name: str, entity_type: str = "") -> None:
        """Delete an entity by name

        Args:
            entity_name: Plain name of the entity to delete.
            entity_type: Entity type.  Required when entity disambiguation is enabled.
        """
        try:
            _disambig = self.global_config.get("enable_entity_disambiguation", True)
            entity_id = compute_entity_vdb_id(entity_name, entity_type, _disambig)
            logger.debug(
                f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            # Scroll to find the entity by its ID field in payload with workspace filtering
            # This is safer than reconstructing the Qdrant point ID
            results = await self._run_client_call_with_timeout(
                self._client.scroll,
                collection_name=self.final_namespace,
                scroll_filter=models.Filter(
                    must=[
                        workspace_filter_condition(self.effective_workspace),
                        models.FieldCondition(
                            key=ID_FIELD, match=models.MatchValue(value=entity_id)
                        ),
                    ]
                ),
                with_payload=False,
                limit=1,
            )

            # Extract point IDs to delete
            points = results[0]
            if points:
                ids_to_delete = [point.id for point in points]
                await self._run_client_call_with_timeout(
                    self._client.delete,
                    collection_name=self.final_namespace,
                    points_selector=models.PointIdsList(points=ids_to_delete),
                    wait=True,
                )
                logger.debug(
                    f"[{self.workspace}] Successfully deleted entity {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] Entity {entity_name} not found in storage"
                )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity

        Args:
            entity_name: Name of the entity whose relations should be deleted
        """
        try:
            # Build the filter to find relations where entity is either source or target
            # must + should = workspace_id matches AND (src_id matches OR tgt_id matches)
            relation_filter = models.Filter(
                must=[workspace_filter_condition(self.effective_workspace)],
                should=[
                    models.FieldCondition(
                        key="src_id", match=models.MatchValue(value=entity_name)
                    ),
                    models.FieldCondition(
                        key="tgt_id", match=models.MatchValue(value=entity_name)
                    ),
                ],
            )

            # Paginate through all matching relations to handle large datasets
            total_deleted = 0
            offset = None
            batch_size = 1000

            while True:
                # Scroll to find relations, using with_payload=False for efficiency
                # since we only need point IDs for deletion
                results = await self._run_client_call_with_timeout(
                    self._client.scroll,
                    collection_name=self.final_namespace,
                    scroll_filter=relation_filter,
                    with_payload=False,
                    with_vectors=False,
                    limit=batch_size,
                    offset=offset,
                )

                points, next_offset = results
                if not points:
                    break

                # Extract point IDs to delete
                ids_to_delete = [point.id for point in points]

                # Delete the batch of relations
                await self._run_client_call_with_timeout(
                    self._client.delete,
                    collection_name=self.final_namespace,
                    points_selector=models.PointIdsList(points=ids_to_delete),
                    wait=True,
                )
                total_deleted += len(ids_to_delete)

                # Check if we've reached the end
                if next_offset is None:
                    break
                offset = next_offset

            if total_deleted > 0:
                logger.debug(
                    f"[{self.workspace}] Deleted {total_deleted} relations for {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            # Convert to Qdrant compatible ID
            qdrant_id = compute_mdhash_id_for_qdrant(
                id, prefix=self.effective_workspace
            )

            # Retrieve the point by ID with workspace filtering
            result = await self._run_client_call_with_timeout(
                self._client.retrieve,
                collection_name=self.final_namespace,
                ids=[qdrant_id],
                with_payload=True,
            )

            if not result:
                return None

            payload = result[0].payload
            if CREATED_AT_FIELD not in payload:
                payload[CREATED_AT_FIELD] = None

            return payload
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        try:
            # Convert to Qdrant compatible IDs
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=self.effective_workspace)
                for id in ids
            ]

            # Retrieve the points by IDs
            results = await self._run_client_call_with_timeout(
                self._client.retrieve,
                collection_name=self.final_namespace,
                ids=qdrant_ids,
                with_payload=True,
            )

            # Ensure each result contains created_at field and preserve caller ordering
            payload_by_original_id: dict[str, dict[str, Any]] = {}
            payload_by_qdrant_id: dict[str, dict[str, Any]] = {}

            for point in results:
                payload = dict(point.payload or {})
                if CREATED_AT_FIELD not in payload:
                    payload[CREATED_AT_FIELD] = None

                qdrant_point_id = str(point.id) if point.id is not None else ""
                if qdrant_point_id:
                    payload_by_qdrant_id[qdrant_point_id] = payload

                original_id = payload.get(ID_FIELD)
                if original_id is not None:
                    payload_by_original_id[str(original_id)] = payload

            ordered_payloads: list[dict[str, Any] | None] = []
            for requested_id, qdrant_id in zip(ids, qdrant_ids):
                payload = payload_by_original_id.get(str(requested_id))
                if payload is None:
                    payload = payload_by_qdrant_id.get(str(qdrant_id))
                ordered_payloads.append(payload)

            return ordered_payloads
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        try:
            # Convert to Qdrant compatible IDs
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=self.effective_workspace)
                for id in ids
            ]

            # Retrieve the points by IDs with vectors
            results = await self._run_client_call_with_timeout(
                self._client.retrieve,
                collection_name=self.final_namespace,
                ids=qdrant_ids,
                with_vectors=True,  # Important: request vectors
                with_payload=True,
            )

            vectors_dict = {}
            for point in results:
                if point and point.vector is not None and point.payload:
                    # Get original ID from payload
                    original_id = point.payload.get(ID_FIELD)
                    if original_id:
                        vector_data = _coerce_qdrant_dense_vector(point.vector)
                        if vector_data is None:
                            logger.warning(
                                "[%s] Skipping vector for ID %s from %s: no dense vector found in Qdrant payload type=%s",
                                self.workspace,
                                original_id,
                                self.namespace,
                                type(point.vector).__name__,
                            )
                            continue
                        vectors_dict[original_id] = vector_data

            return vectors_dict
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will delete all data for the current workspace from the Qdrant collection.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        # No need to lock: data integrity is ensured by allowing only one process to hold pipeline at a time
        try:
            # Delete all points for the current workspace
            await self._run_client_call_with_timeout(
                self._client.delete,
                collection_name=self.final_namespace,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[workspace_filter_condition(self.effective_workspace)]
                    )
                ),
                wait=True,
            )

            logger.info(
                f"[{self.workspace}] Process {os.getpid()} dropped workspace data from Qdrant collection {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping workspace data from Qdrant collection {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}
