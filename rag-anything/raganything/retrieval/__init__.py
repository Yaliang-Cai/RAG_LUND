from .profiles import RetrievalProfile, PROFILE_REGISTRY, KNOWN_PATHS

# TODO: remove guard once classifier, paths, and router modules are all implemented
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
