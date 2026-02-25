"""
Configuration classes for RAGAnything

Contains configuration dataclasses with environment variable support
"""

from dataclasses import dataclass, field
from typing import List
from lightrag.utils import get_env_value

from raganything.constants import (
    DEFAULT_WORKING_DIR,
    DEFAULT_PARSE_METHOD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARSER,
    DEFAULT_DISPLAY_CONTENT_STATS,
    DEFAULT_ENABLE_IMAGE_PROCESSING,
    DEFAULT_ENABLE_TABLE_PROCESSING,
    DEFAULT_ENABLE_EQUATION_PROCESSING,
    DEFAULT_MAX_CONCURRENT_FILES,
    DEFAULT_SUPPORTED_FILE_EXTENSIONS,
    DEFAULT_RECURSIVE_FOLDER_PROCESSING,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_INCLUDE_HEADERS,
    DEFAULT_INCLUDE_CAPTIONS,
    DEFAULT_CONTEXT_FILTER_CONTENT_TYPES,
    DEFAULT_CONTENT_FORMAT,
    DEFAULT_USE_FULL_PATH,
)


@dataclass
class RAGAnythingConfig:
    """Configuration class for RAGAnything with environment variable support"""

    # Directory Configuration
    # ---
    working_dir: str = field(default=get_env_value("WORKING_DIR", DEFAULT_WORKING_DIR, str))
    """Directory where RAG storage and cache files are stored."""

    # Parser Configuration
    # ---
    parse_method: str = field(default=get_env_value("PARSE_METHOD", DEFAULT_PARSE_METHOD, str))
    """Default parsing method for document parsing: 'auto', 'ocr', or 'txt'."""

    parser_output_dir: str = field(default=get_env_value("OUTPUT_DIR", DEFAULT_OUTPUT_DIR, str))
    """Default output directory for parsed content."""

    parser: str = field(default=get_env_value("PARSER", DEFAULT_PARSER, str))
    """Parser selection: 'mineru' or 'docling'."""

    display_content_stats: bool = field(
        default=get_env_value("DISPLAY_CONTENT_STATS", DEFAULT_DISPLAY_CONTENT_STATS, bool)
    )
    """Whether to display content statistics during parsing."""

    # Multimodal Processing Configuration
    # ---
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", DEFAULT_ENABLE_IMAGE_PROCESSING, bool)
    )
    """Enable image content processing."""

    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", DEFAULT_ENABLE_TABLE_PROCESSING, bool)
    )
    """Enable table content processing."""

    enable_equation_processing: bool = field(
        default=get_env_value("ENABLE_EQUATION_PROCESSING", DEFAULT_ENABLE_EQUATION_PROCESSING, bool)
    )
    """Enable equation content processing."""

    # Batch Processing Configuration
    # ---
    max_concurrent_files: int = field(
        default=get_env_value("MAX_CONCURRENT_FILES", DEFAULT_MAX_CONCURRENT_FILES, int)
    )
    """Maximum number of files to process concurrently."""

    supported_file_extensions: List[str] = field(
        default_factory=lambda: get_env_value(
            "SUPPORTED_FILE_EXTENSIONS",
            DEFAULT_SUPPORTED_FILE_EXTENSIONS,
            str,
        ).split(",")
    )
    """List of supported file extensions for batch processing."""

    recursive_folder_processing: bool = field(
        default=get_env_value("RECURSIVE_FOLDER_PROCESSING", DEFAULT_RECURSIVE_FOLDER_PROCESSING, bool)
    )
    """Whether to recursively process subfolders in batch mode."""

    # Context Extraction Configuration
    # ---
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW, int))
    """Number of pages/chunks to include before and after current item for context."""

    context_mode: str = field(default=get_env_value("CONTEXT_MODE", DEFAULT_CONTEXT_MODE, str))
    """Context extraction mode: 'page' for page-based, 'chunk' for chunk-based."""

    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS, int)
    )
    """Maximum number of tokens in extracted context."""

    include_headers: bool = field(default=get_env_value("INCLUDE_HEADERS", DEFAULT_INCLUDE_HEADERS, bool))
    """Whether to include document headers and titles in context."""

    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", DEFAULT_INCLUDE_CAPTIONS, bool)
    )
    """Whether to include image/table captions in context."""

    context_filter_content_types: List[str] = field(
        default_factory=lambda: get_env_value(
            "CONTEXT_FILTER_CONTENT_TYPES", DEFAULT_CONTEXT_FILTER_CONTENT_TYPES, str
        ).split(",")
    )
    """Content types to include in context extraction (e.g., 'text', 'image', 'table')."""

    content_format: str = field(default=get_env_value("CONTENT_FORMAT", DEFAULT_CONTENT_FORMAT, str))
    """Default content format for context extraction when processing documents."""

    # Path Handling Configuration
    # ---
    use_full_path: bool = field(default=get_env_value("USE_FULL_PATH", DEFAULT_USE_FULL_PATH, bool))
    """Whether to use full file path (True) or just basename (False) for file references in LightRAG."""

    def __post_init__(self):
        """Post-initialization setup for backward compatibility"""
        # Support legacy environment variable names for backward compatibility
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parse_method = legacy_parse_method
            import warnings

            warnings.warn(
                "MINERU_PARSE_METHOD is deprecated. Use PARSE_METHOD instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def mineru_parse_method(self) -> str:
        """
        Backward compatibility property for old code.

        .. deprecated::
           Use `parse_method` instead. This property will be removed in a future version.
        """
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        """Setter for backward compatibility"""
        import warnings

        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parse_method = value
