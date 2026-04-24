from .profiles import RetrievalProfile, PROFILE_REGISTRY, KNOWN_PATHS

try:
    from .classifier import QueryClassifier
    from .paths import run_path
    from .router import RetrievalRouter, RetrievalError
except (ImportError, ModuleNotFoundError):
    pass

__all__ = [
    "RetrievalProfile",
    "PROFILE_REGISTRY",
    "KNOWN_PATHS",
    "QueryClassifier",
    "run_path",
    "RetrievalRouter",
    "RetrievalError",
]
