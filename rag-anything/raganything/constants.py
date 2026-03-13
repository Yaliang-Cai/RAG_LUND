"""
Centralized configuration constants for RAG-Anything.

This module defines default values for all configuration constants used across
the RAG-Anything system. Centralizing these values ensures consistency and
makes maintenance easier.

Usage:
    from raganything.constants import DEFAULT_WORKING_DIR_ROOT, DEFAULT_PARSER
"""

from pathlib import Path as _Path

# Data root: constants.py lives at raganything/constants.py
# .parent.parent → RAG_LUND/rag-anything/
# / "rag-anything" → RAG_LUND/rag-anything/rag-anything/  (matches web-server CWD behaviour)
_PKG_ROOT = _Path(__file__).resolve().parent.parent / "rag-anything"

# =============================================================================
# Directory defaults
# =============================================================================
DEFAULT_OUTPUT_DIR = str(_PKG_ROOT / "output")
DEFAULT_WORKING_DIR_ROOT = str(_PKG_ROOT / "rag_workspace")
DEFAULT_LOG_DIR = str(_PKG_ROOT / "logs")

# =============================================================================
# Parser configuration
# =============================================================================
DEFAULT_PARSER = "mineru"  # "mineru" or "docling"
DEFAULT_PARSE_METHOD = "auto"  # "auto", "ocr", or "txt"
DEFAULT_CONTENT_FORMAT = "minerU"
DEFAULT_DISPLAY_CONTENT_STATS = True
DEFAULT_MINERU_VLLM_GPU_MEMORY_UTILIZATION = 0.1

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
DEFAULT_QUERY_MODE = "hybrid"   # "naive" | "local" | "global" | "hybrid"
DEFAULT_ENABLE_RERANK = True
DEFAULT_VLM_ENHANCED = True

# Reranker score 过滤阈值：rerank 完成后，得分低于此值的 chunk 会被丢弃。
# LightRAG 原默认值为 0.0（即不过滤）；BGE-reranker-v2-m3（CrossEncoder）
# 对相关 chunk 的典型得分 > 0.5，不相关 chunk 通常 < 0.3。
# 设为 0.3 可过滤掉明显不相关的 chunk，同时保留模糊相关内容。
# 调高此值（如 0.5）可进一步提升精确度，但可能降低召回率。
DEFAULT_MIN_RERANK_SCORE = 0.3

# =============================================================================
# Knowledge graph visualization defaults
# =============================================================================
DEFAULT_GRAPH_MAX_DEPTH = 2
DEFAULT_GRAPH_MAX_NODES = 50
DEFAULT_GRAPH_OVERVIEW_MAX_NODES = 30   # for overview endpoint (no query filter)
DEFAULT_GRAPH_HTML_MAX_NODES = 60       # for pyvis HTML rendering
DEFAULT_GRAPH_SEARCH_SEED_LIMIT = 10    # max seed nodes when filtering by query
DEFAULT_GRAPH_SEARCH_MAX_RESULTS = 20   # default limit for entity search endpoint
DEFAULT_GRAPH_SEARCH_MAX_SAFE = 100     # hard cap for entity search results

# =============================================================================
# Local deployment - model paths
# =============================================================================
DEFAULT_TIKTOKEN_CACHE_DIR = "/data/y50056788/Yaliang/projects/lightrag/tiktoken_cache"
DEFAULT_EMBEDDING_MODEL_PATH = "/data/h50056787/models/bge-m3"
DEFAULT_RERANK_MODEL_PATH = "/data/h50056787/models/bge-reranker-v2-m3"
DEFAULT_VISION_MODEL_PATH = "/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8"
DEFAULT_TOKENIZER_MODEL_PATH = DEFAULT_VISION_MODEL_PATH

# =============================================================================
# Local deployment - LLM / VLM service
# =============================================================================
DEFAULT_VLLM_API_BASE = "http://localhost:8001/v1"
DEFAULT_VLLM_API_KEY = "EMPTY"
DEFAULT_LLM_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
DEFAULT_DEVICE = "cuda:0"

# =============================================================================
# Indexing concurrency & quality
#
# 这些参数控制 LightRAG indexing 阶段的并发度和提取质量，通过
# local_rag.py._build_rag() 的 lightrag_kwargs 传入 LightRAG 实例。
# =============================================================================

# 每个 chunk 的 entity extraction LLM 调用最大并发数。
# LightRAG 默认值为 4；A6000 48GB + FP8 30B 模型理论最大约 59，
# 设为 16 可大幅提速同时保留安全余量。
DEFAULT_LLM_MODEL_MAX_ASYNC = 16

# Entity extraction 的 gleaning（补充提取）轮数。
# gleaning=1 表示每个 chunk 做 2 次串行 LLM 调用（初始 + 1 次补充），
# 可提高覆盖率但 indexing 时间翻倍。设为 0 可禁用 gleaning 换取速度。
DEFAULT_ENTITY_EXTRACT_MAX_GLEANING = 1

# 文档级最大并发插入数（pipeline 层面，非 LLM 层面）。
# LightRAG 默认值为 2，适当增大可在多文档批量 indexing 时提升吞吐。
DEFAULT_MAX_PARALLEL_INSERT = 4

# Embedding 模型单次批处理的最大文本数。
# LightRAG 默认值为 10；BGE-M3 支持更大 batch，设为 32 可减少
# embedding 调用次数，提升 GPU 利用率。
DEFAULT_EMBEDDING_BATCH_NUM = 32

# Embedding 调用最大并发数（与 LLM 并发独立计数）。
# LightRAG 默认值为 8，通常无需修改。
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8

# =============================================================================
# Chunking strategy
# =============================================================================
DEFAULT_CHUNKING_STRATEGY = "token"   # "token" | "recursive" | "sentence" | "paragraph" | "semantic"
DEFAULT_CHUNK_TOKEN_SIZE = 1200       # max tokens per chunk
DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE = 100  # overlap tokens between consecutive chunks

# =============================================================================
# Local deployment - generation parameters
# =============================================================================
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_MAX_TOKEN_SIZE = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_QUERY_MAX_TOKENS = 2048
DEFAULT_INGEST_MAX_TOKENS = 8192

# =============================================================================
# Local deployment - VLM parameters
# =============================================================================
DEFAULT_VLM_ENABLE_JSON_SCHEMA = True
DEFAULT_IMAGE_TOKEN_ESTIMATE_METHOD = "qwen_vl"
DEFAULT_IMAGE_WRAPPER_TOKENS_PER_IMAGE = 2

# =============================================================================
# Logging
# =============================================================================
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_LOG_BACKUP_COUNT = 5
