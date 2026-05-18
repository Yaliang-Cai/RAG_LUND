import numpy as np

from lightrag.kg.qdrant_impl import (
    _estimate_qdrant_point_json_bytes,
    _iter_qdrant_point_batches_by_payload,
)
from qdrant_client import models


def _point(point_id: str, content_size: int) -> models.PointStruct:
    return models.PointStruct(
        id=point_id,
        vector=np.zeros(8, dtype=np.float32),
        payload={"id": point_id, "content": "x" * content_size},
    )


def test_qdrant_upsert_batches_by_estimated_payload_size():
    points = [_point("a", 400), _point("b", 400), _point("c", 400)]
    max_payload = (
        1024
        + _estimate_qdrant_point_json_bytes(points[0])
        + 4
        + _estimate_qdrant_point_json_bytes(points[1])
        + 4
    )

    batches = _iter_qdrant_point_batches_by_payload(points, max_payload)

    assert [len(batch) for batch in batches] == [2, 1]


def test_qdrant_upsert_keeps_single_oversized_point_retryable():
    point = _point("oversized", 2048)

    batches = _iter_qdrant_point_batches_by_payload([point], 1024)

    assert batches == [[point]]
