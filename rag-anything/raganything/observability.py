# raganything/observability.py
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
_phoenix_initialized: bool = False


def setup_phoenix(project_name: str = "rag-agentic", port: int = 6006) -> None:
    """Initialize Arize Phoenix local OTEL tracing.

    Safe to call multiple times; initializes only once.
    If arize-phoenix is not installed, logs a warning and returns silently.
    """
    global _phoenix_initialized
    if _phoenix_initialized:
        return
    try:
        from phoenix.otel import register  # type: ignore[import]
        register(project_name=project_name, auto_instrument=True)
        logger.info("Arize Phoenix initialized at http://localhost:%d", port)
    except (ImportError, TypeError):
        logger.warning(
            "arize-phoenix not installed; observability disabled. "
            "Install with: pip install 'arize-phoenix[otel]>=4.0'"
        )
    finally:
        _phoenix_initialized = True
