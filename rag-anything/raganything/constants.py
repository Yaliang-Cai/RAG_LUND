"""
Centralized configuration constants for RAG-Anything.

This module defines default values for all configuration constants used across
the RAG-Anything system. Centralizing these values ensures consistency and
makes maintenance easier.

Usage:
    from raganything.constants import DEFAULT_WORKING_DIR_ROOT, DEFAULT_PARSER
"""

# =============================================================================
# Directory defaults
# =============================================================================
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_WORKING_DIR_ROOT = "./rag_workspace"
DEFAULT_UPLOAD_DIR = "./uploads"
DEFAULT_LOG_DIR = "./logs"

# =============================================================================
# Parser configuration
# =============================================================================
DEFAULT_PARSER = "mineru"  # "mineru" or "docling"
DEFAULT_PARSE_METHOD = "auto"  # "auto", "ocr", or "txt"
DEFAULT_CONTENT_FORMAT = "minerU"
DEFAULT_DISPLAY_CONTENT_STATS = True

# =============================================================================
# Multimodal processing
# =============================================================================
DEFAULT_ENABLE_IMAGE_PROCESSING = True
DEFAULT_ENABLE_TABLE_PROCESSING = True
DEFAULT_ENABLE_EQUATION_PROCESSING = True

# Maximum number of multimodal chunks kept after reranking in VLM-enhanced query
DEFAULT_MULTIMODAL_TOP_K = 3

# =============================================================================
# Batch processing
# =============================================================================
DEFAULT_MAX_CONCURRENT_FILES = 1
DEFAULT_SUPPORTED_FILE_EXTENSIONS = (
    ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,"
    ".doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md"
)
DEFAULT_RECURSIVE_FOLDER_PROCESSING = True

# =============================================================================
# Context extraction
# =============================================================================
DEFAULT_CONTEXT_WINDOW = 1
DEFAULT_CONTEXT_MODE = "page"  # "page" or "chunk"
DEFAULT_MAX_CONTEXT_TOKENS = 2000
DEFAULT_INCLUDE_HEADERS = True
DEFAULT_INCLUDE_CAPTIONS = True
DEFAULT_CONTEXT_FILTER_CONTENT_TYPES = "text"

# =============================================================================
# Path handling
# =============================================================================
DEFAULT_USE_FULL_PATH = False

# =============================================================================
# Image validation
# =============================================================================
DEFAULT_MAX_IMAGE_SIZE_MB = 50
SUPPORTED_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
]

# =============================================================================
# Query defaults (used by server and QueryRequest)
# =============================================================================
DEFAULT_TOP_K = 20        # default and max allowed value for top_k
DEFAULT_CHUNK_TOP_K = 10  # default and max allowed value for chunk_top_k

# =============================================================================
# Local deployment — model paths
# =============================================================================
DEFAULT_TIKTOKEN_CACHE_DIR = "/data/h50056787/workspaces/lightrag/tiktoken_cache"
DEFAULT_EMBEDDING_MODEL_PATH = "/data/h50056787/models/bge-m3"
DEFAULT_RERANK_MODEL_PATH = "/data/h50056787/models/bge-reranker-v2-m3"

# =============================================================================
# Local deployment — LLM / VLM service
# =============================================================================
DEFAULT_VLLM_API_BASE = "http://localhost:8001/v1"
DEFAULT_VLLM_API_KEY = "EMPTY"
DEFAULT_LLM_MODEL_NAME = "OpenGVLab/InternVL2-26B-AWQ"
DEFAULT_DEVICE = "cuda:0"

# =============================================================================
# Local deployment — generation parameters
# =============================================================================
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_MAX_TOKEN_SIZE = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_QUERY_MAX_TOKENS = 2048
DEFAULT_INGEST_MAX_TOKENS = 8192

# =============================================================================
# Local deployment — VLM parameters
# =============================================================================
DEFAULT_VLM_ENABLE_JSON_SCHEMA = True

# =============================================================================
# Logging
# =============================================================================
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_LOG_BACKUP_COUNT = 5
