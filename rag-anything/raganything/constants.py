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
# / "rag_data" → RAG_LUND/rag-anything/rag_data/
_PKG_ROOT = _Path(__file__).resolve().parent.parent / "rag_data"

# =============================================================================
# Directory defaults
# =============================================================================
DEFAULT_OUTPUT_DIR = str(_PKG_ROOT / "output")
DEFAULT_WORKING_DIR_ROOT = str(_PKG_ROOT / "rag_workspace")
DEFAULT_UPLOADS_DIR = str(_PKG_ROOT / "uploads")
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

# Multimodal ingest timeout/guardrail defaults
# Single multimodal item request timeout (seconds)
DEFAULT_MULTIMODAL_ITEM_TIMEOUT_SECONDS = 1200
# Whole multimodal batch watchdog timeout (seconds)
DEFAULT_MULTIMODAL_BATCH_WATCHDOG_SECONDS = 7200
# Grace period for cancelled multimodal tasks to exit (seconds)
DEFAULT_MULTIMODAL_CANCEL_GRACE_SECONDS = 10
# Enable two-stage strict fallback for multimodal ingest:
# strict=true first, then fallback strict=false path on failure.
DEFAULT_MULTIMODAL_ENABLE_STRICT_FALLBACK = True

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
DEFAULT_ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE = True
DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES = (
    "page_number,page_footnote,footer,header,ref_text"
)

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
DEFAULT_CHUNK_TOP_K = 10  # final window size after reranking (chunk_top_k)
DEFAULT_NAIVE_TOP_K = 20  # naive VDB retrieval count (mix/naive modes); independent of chunk_top_k
DEFAULT_QUERY_MODE = "hybrid"   # "naive" | "local" | "global" | "hybrid" | "mix" | "rrf" | "ppr_local" | "ppr" | "gfm"
DEFAULT_ENABLE_RERANK = True
DEFAULT_ENABLE_KG_RERANK = True  # rerank entity/relation KG results (hybrid/mix); independent of chunk rerank
DEFAULT_VLM_ENHANCED = True
DEFAULT_KG_CHUNK_SELECTION_SOURCE = "truncated"  # "truncated" | "untruncated"

# Reranker score 过滤阈值：rerank 完成后，得分低于此值的 chunk 会被丢弃。
# 0.3：过滤 rerank 分数断崖后的低质量 chunk，避免噪声 chunk 干扰 LLM。
DEFAULT_MIN_RERANK_SCORE = 0.3
DEFAULT_RERANK_BATCH_SIZE = 8           # was 32 — locks batch, eliminates OOM backoff
DEFAULT_RERANK_ENABLE_OOM_BACKOFF = False  # was True
DEFAULT_RERANK_MIN_BATCH_SIZE = 4

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
DEFAULT_TEXT_REQUEST_TIMEOUT_SECONDS = 1800.0
DEFAULT_VISION_REQUEST_TIMEOUT_SECONDS = 1800.0
DEFAULT_DEVICE = "cuda:0"

# =============================================================================
# Indexing concurrency & quality
#
# 这些参数控制 LightRAG indexing 阶段的并发度和提取质量，通过
# local_rag.py._build_rag() 的 lightrag_kwargs 传入 LightRAG 实例。
# =============================================================================

# 每个 chunk 的 entity extraction LLM 调用最大并发数。
# 单卡 48GB FP8 MoE 模型建议不超过 6，配合 vLLM --max-num-seqs 6 使用。
# 过高会导致请求在 vLLM scheduler 排队，产生超时。
DEFAULT_LLM_MODEL_MAX_ASYNC = 16

# Entity extraction 的 gleaning（补充提取）轮数。
# gleaning=1 表示每个 chunk 做 2 次串行 LLM 调用（初始 + 1 次补充），
# 可提高覆盖率但 indexing 时间翻倍。设为 0 可禁用 gleaning 换取速度。
DEFAULT_ENTITY_EXTRACT_MAX_GLEANING = 1

# 文档级最大并发插入数（pipeline 层面，非 LLM 层面）。
# LightRAG 默认值为 2，适当增大可在多文档批量 indexing 时提升吞吐。
DEFAULT_MAX_PARALLEL_INSERT = 4

# Per-workspace ingest lock default:
# True keeps same-workspace ingest serialized (safer); False allows concurrent
# ingest into one shared workspace when caller manages race-safety.
DEFAULT_SERIALIZE_INGEST_BY_WORKSPACE_ID = True

# CLI 文件夹模式下的文档级并发入库数（对应 evaluate_shared.py 的 max_async_ingest）。
# 与 DEFAULT_LLM_MODEL_MAX_ASYNC 分属两层：
#   - 此值控制同时进入 pipeline（解析 + LLM 抽取）的文件数
#   - DEFAULT_LLM_MODEL_MAX_ASYNC 控制所有文件共享的 LLM HTTP 请求并发上限
DEFAULT_MAX_ASYNC_INGEST = 4

# Embedding 模型单次批处理的最大文本数。
# LightRAG 默认值为 10；BGE-M3 支持更大 batch，设为 32 可减少
# embedding 调用次数，提升 GPU 利用率。
DEFAULT_EMBEDDING_BATCH_NUM = 32

# Embedding 调用最大并发数（与 LLM 并发独立计数）。
# LightRAG 默认值为 8，通常无需修改。
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8

# =============================================================================
# Qdrant vector storage defaults
# =============================================================================
# Controls whether RAG-Anything uses Qdrant hybrid dense+BM25 sparse collections
# by default. When enabled, Qdrant collection names receive a "_bm25" suffix.
# Set to False when querying or extending older dense-only Qdrant collections.
DEFAULT_QDRANT_ENABLE_SPARSE_BM25 = True
DEFAULT_QDRANT_SPARSE_BM25_MODEL = "Qdrant/bm25"
# Query-time retrieval mode for Qdrant collections that contain dense+BM25 vectors.
# Supported: "dense", "bm25", "hybrid". Dense preserves the previous behavior.
DEFAULT_QDRANT_RETRIEVAL_MODE = "dense"

# =============================================================================
# V1: Entity disambiguation
# =============================================================================
DEFAULT_ENABLE_ENTITY_DISAMBIGUATION = True

# =============================================================================
# Entity surface normalization (ingest-time, optional)
# =============================================================================
DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION = True
DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION = True
DEFAULT_ENTITY_UPPERCASE_ALLOWLIST = [
    "AI",
    "API",
    "ASR",
    "BERT",
    "CNN",
    "CPU",
    "GPU",
    "GPT",
    "HTTP",
    "HTTPS",
    "JSON",
    "LLM",
    "LSTM",
    "ML",
    "NLP",
    "OCR",
    "RAG",
    "RNN",
    "SDK",
    "SQL",
    "TTS",
    "XML",
    "YAML",
    "3G",
    "4G",
    "5G",
    "6G",
]
DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH = True

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
# Citation behavior
# =============================================================================

# When True, LLM is instructed to embed [DC1]/[DC2] inline citations after every
# factual statement. When False, only the default doc-level ### References section
# at the end of the answer is produced.
DEFAULT_ENABLE_INLINE_CITATIONS = False

# =============================================================================
# V2: Synonym Linking (neo4j-milvus branch)
# =============================================================================
DEFAULT_ENABLE_SYNONYM_LINKING = False          # Enable/disable V2
DEFAULT_SYNONYMY_THRESHOLD = 0.8                # cosine similarity threshold for synonyms
DEFAULT_SYNONYMY_TOPK = 2048                    # KNN top-K for synonym detection（Qdrant 无上限，对齐 HippoRAG2 ~2047）
DEFAULT_SYNONYMY_MIN_ENTITY_LEN = 2             # Min entity name length (filter short entities)

# =============================================================================
# V3: PPR Multi-hop Reasoning (neo4j-milvus branch)
# =============================================================================
DEFAULT_ENABLE_MULTI_HOP = False                # [Deprecated] Legacy flag for ppr_local. Use mode="ppr" instead.
DEFAULT_MULTI_HOP_DEPTH = 2                     # BFS depth for ppr_local subgraph (legacy; unused in global PPR)
DEFAULT_PPR_DAMPING = 0.5                       # PPR damping factor (alpha)
DEFAULT_PPR_TOP_K = 50                          # Number of chunks returned by PPR
DEFAULT_PASSAGE_NODE_WEIGHT = 0.05              # HippoRAG2 param: DPR chunk score scaling in PPR seed
DEFAULT_PPR_SYNONYM_WEIGHT_MODE = "raw"         # "raw" | "plus_one" (retrieval-time mapping only)
# None = auto (PPR modes default False, non-PPR modes default True).
DEFAULT_EXCLUDE_SYNONYM_EDGES = None            # Hard filter synonym edges at query time (local/global/hybrid/ppr)
DEFAULT_RECOGNITION_TOP_K = 20                  # Recognition-memory relation top-k (global PPR)
DEFAULT_LINKING_TOP_K = 5                       # Max entity seeds from recognition memory (HippoRAG2 link_top_k)
DEFAULT_PPR_QA_TOP_K = 5                        # Chunks fed to LLM after PPR retrieval (HippoRAG2 qa_top_k)
DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS = 65536   # LLM recognition prompt hard cap for global PPR
DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS = 8192  # LLM recognition output token cap
DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS = 200  # Safety reserve for wrappers/system overhead
DEFAULT_RECOGNITION_DIFFLIB_CUTOFF = 0.5        # difflib fuzzy-match cutoff: LLM output → entity_id (0.0–1.0)
# Resilience & callback (service-level optional controls)
# =============================================================================
# 是否在 LocalRagService 层启用重试与熔断机制。
# 默认开启，统一由服务层承担重试/熔断能力。
DEFAULT_ENABLE_RESILIENCE = True

# 重试总次数（包含首次调用）。
DEFAULT_RESILIENCE_MAX_ATTEMPTS = 3

# ingest/query 的初始退避时间（秒）。
DEFAULT_INGEST_RETRY_BASE_DELAY = 2.0
DEFAULT_QUERY_RETRY_BASE_DELAY = 1.0

# 单次退避等待上限（秒）。
DEFAULT_RESILIENCE_MAX_DELAY = 20.0

# 熔断阈值：在 reset_timeout 窗口内累计失败达到阈值后打开熔断。
DEFAULT_INGEST_BREAKER_FAILURE_THRESHOLD = 8
DEFAULT_QUERY_BREAKER_FAILURE_THRESHOLD = 12

# 熔断打开后进入 half-open 试探调用前的等待时间（秒）。
DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS = 120.0

# callback 相关开关（默认关闭，按需启用）。
DEFAULT_ENABLE_METRICS_CALLBACK = False
DEFAULT_ENABLE_CALLBACK_EVENT_LOG = False

# =============================================================================
# Evaluation defaults
# =============================================================================
# When True, evaluate_shared generate mode ingests only failed docs listed in
# shared_ingest_failures.jsonl (within start_id/end_id range).
DEFAULT_EVAL_RETRY_FAILED_ONLY = False

# =============================================================================
# Logging
# =============================================================================
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_LOG_BACKUP_COUNT = 5

# =============================================================================
# GFM-RAG retrieval
# =============================================================================
GFM_DATA_DIR = "./data"          # root data dir for GFM-RAG index
GFM_DATA_NAME = ""               # graph name; empty string disables GFM path
GFM_MODEL_PATH = "rmanluo/G-reasoner-34M"  # HuggingFace model id or local path

# =============================================================================
# Agentic RAG (V4)
# =============================================================================
DEFAULT_AGENTIC_MAX_RETRIEVE_CYCLES = 3
DEFAULT_AGENTIC_MAX_CHECK_CYCLES = 2
DEFAULT_AGENTIC_ROUTER_CACHE_SIZE = 2048
DEFAULT_AGENTIC_ROUTER_FALLBACK_PROFILE = "semantic"
DEFAULT_AGENTIC_DECOMPOSE_MAX_SUBQUESTIONS = 4
DEFAULT_AGENTIC_PARALLEL_RETRIEVE_CONCURRENCY = 3
DEFAULT_AGENTIC_GRADER_FALLBACK_SUFFICIENT = False
