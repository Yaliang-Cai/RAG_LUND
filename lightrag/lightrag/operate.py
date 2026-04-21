from __future__ import annotations
from functools import partial
from pathlib import Path
import unicodedata
import hashlib
import os
import math

import asyncio
import json
import json_repair
from typing import Any, AsyncIterator, overload, Literal, Callable
from collections import Counter, defaultdict

from lightrag.exceptions import (
    PipelineCancelledException,
    ChunkTokenLimitExceededError,
)
from lightrag.utils import (
    logger,
    compute_mdhash_id,
    compute_entity_id,
    compute_entity_vdb_id,
    Tokenizer,
    is_float_regex,
    sanitize_and_normalize_extracted_text,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    use_llm_func_with_cache,
    update_chunk_cache_list,
    remove_think_tags,
    pick_by_weighted_polling,
    pick_by_vector_similarity,
    process_chunks_unified,
    safe_vdb_operation_with_exception,
    create_prefixed_exception,
    fix_tuple_delimiter_corruption,
    convert_to_user_format,
    generate_reference_list_from_chunks,
    apply_source_ids_limit,
    merge_source_ids,
    make_relation_chunk_key,
    apply_rerank_if_enabled,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    QueryResult,
    QueryContextResult,
)
from lightrag.prompt import PROMPTS
from lightrag.constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_ENABLE_IMAGE_TOKEN_BUDGET,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
    DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION,
    DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION,
    DEFAULT_ENTITY_UPPERCASE_ALLOWLIST,
    DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH,
    DEFAULT_RECOGNITION_TOP_K,
    DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS,
    DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS,
    DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS,
)
from lightrag.kg.shared_storage import get_storage_keyed_lock
import time
from dotenv import load_dotenv

FACTUAL_EDGE_TYPE = "FACTUAL"
FACTUAL_EDGE_PROVENANCE = "relation_extraction"
SYNONYM_EDGE_TYPE = "SYNONYM"
SYNONYM_EDGE_PROVENANCE = "synonym_detection"

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - optional dependency
    PILImage = None


def _is_synonym_edge(edge_data: dict[str, Any] | None) -> bool:
    """Best-effort synonym edge detection for compatibility with legacy records."""
    if not isinstance(edge_data, dict):
        return False

    edge_type = str(edge_data.get("edge_type", "")).upper()
    if edge_type == SYNONYM_EDGE_TYPE:
        return True

    provenance = str(edge_data.get("provenance", "")).strip().lower()
    if provenance == SYNONYM_EDGE_PROVENANCE:
        return True

    # Legacy fallback: synonym edges often have no real source_id and synonym keywords.
    source_id = str(edge_data.get("source_id", "") or "").strip()
    keywords = str(edge_data.get("keywords", "") or "").lower()
    if not source_id and ("synonym" in keywords or "alias" in keywords):
        return True

    return False


def _should_exclude_synonym_edges(query_param: QueryParam) -> bool:
    """Resolve query-time synonym filtering with mode-aware defaults.

    Auto default (exclude_synonym_edges is None):
    - PPR mode (ppr/ppr_local or legacy enable_multi_hop): False
    - Non-PPR modes: True
    """
    if query_param.exclude_synonym_edges is not None:
        return bool(query_param.exclude_synonym_edges)

    ppr_mode = query_param.mode in ("ppr", "ppr_local") or query_param.enable_multi_hop
    return not ppr_mode


def _is_factual_or_legacy_edge(edge_data: dict[str, Any] | None) -> bool:
    """Treat untyped historical extraction edges as factual unless recognized as synonym."""
    if not isinstance(edge_data, dict):
        return False

    edge_type = str(edge_data.get("edge_type", "")).upper()
    if edge_type == FACTUAL_EDGE_TYPE:
        return True
    if edge_type == SYNONYM_EDGE_TYPE:
        return False

    if _is_synonym_edge(edge_data):
        return False

    # Legacy edges without explicit typing are considered factual.
    return True


def _to_non_negative_float(value: Any, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    if parsed < 0.0:
        return 0.0
    return parsed


def _to_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 0:
        return 0
    return parsed


def _extract_existing_factual_weight_raw(edge_data: dict[str, Any] | None) -> float:
    """Read factual raw weight from edge; fallback to historical `weight` if needed."""
    if not _is_factual_or_legacy_edge(edge_data):
        return 0.0
    if not isinstance(edge_data, dict):
        return 0.0
    if edge_data.get("weight_raw") is not None:
        return _to_non_negative_float(edge_data.get("weight_raw"), default=0.0)
    return _to_non_negative_float(edge_data.get("weight"), default=1.0)


def _factual_weight_from_raw(weight_raw: float) -> float:
    return math.log1p(max(0.0, _to_non_negative_float(weight_raw, default=0.0)))


async def _remove_relation_edge_and_vector(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    src_id: str,
    tgt_id: str,
) -> None:
    try:
        await knowledge_graph_inst.remove_edges([(src_id, tgt_id)])
    except Exception as exc:
        logger.debug(
            "Failed to remove graph edge `%s`~`%s` during strict endpoint cleanup: %s",
            src_id,
            tgt_id,
            exc,
        )

    if relationships_vdb is None:
        return

    rel_src, rel_tgt = (src_id, tgt_id)
    if rel_src > rel_tgt:
        rel_src, rel_tgt = rel_tgt, rel_src

    rel_vdb_id = compute_mdhash_id(rel_src + rel_tgt, prefix="rel-")
    rel_vdb_id_reverse = compute_mdhash_id(rel_tgt + rel_src, prefix="rel-")
    try:
        await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
    except Exception as exc:
        logger.debug(
            "Failed to remove relationship vectors `%s`/`%s` during strict endpoint cleanup: %s",
            rel_vdb_id,
            rel_vdb_id_reverse,
            exc,
        )

try:
    from transformers import AutoImageProcessor
except Exception:  # pragma: no cover - optional dependency
    AutoImageProcessor = None

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

import re as _re

# Matches any valid URL scheme (http, https, ftp, s3, git, doi, arxiv, etc.)
# Used to preserve legitimate hyperlinks from source documents.
_URL_SCHEME_RE = _re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", _re.IGNORECASE)

# Bare filename with a known file/code extension but no directory separators.
# Catches artifacts like "figure1.png", "config.yaml", "run.sh".
_BARE_FILENAME_RE = _re.compile(
    r"^[^\s\\\/]*\.(?:"
    r"jpg|jpeg|png|gif|bmp|webp|tiff|tif|svg|ico"
    r"|pdf|doc|docx|txt|xlsx|xls|pptx|ppt|html|htm|md|csv"
    r"|py|js|ts|java|cpp|c|h|go|rs|rb|sh|bat"
    r"|yaml|yml|json|xml|toml|cfg|ini|conf|log|env|lock"
    r")$",
    _re.IGNORECASE,
)


# Common inline math / LaTeX signals to avoid false path filtering.
_LATEX_CMD_RE = _re.compile(
    r"\\(?:mathbf|mathcal|frac|sum|prod|sqrt|alpha|beta|gamma|delta|epsilon|varepsilon|mathbb|mathrm|text)\b",
    _re.IGNORECASE,
)
_LATEX_DELIM_RE = _re.compile(r"(\\\(|\\\)|\\\[|\\\]|\$[^$]+\$)")

# Common slash phrases that are not filesystem paths.
_SLASH_PHRASE_WHITELIST = {
    "and/or",
    "input/output",
    "output/input",
    "read/write",
    "yes/no",
    "on/off",
}

# Layout/structure metadata that should not enter the knowledge graph as entities.
_LAYOUT_METADATA_ENTITIES = {
    "page index",
    "page number",
    "footer content",
    "header content",
    "page footnote",
    "page footnote content",
    "bounding box",
    "bounding box coordinates",
    "reference type",
    "token type",
    "token types",
    "image path",
}

# Generic low-information residues.
_GENERIC_NOISE_ENTITIES = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "of",
    "in",
    "on",
    "to",
    "for",
    "by",
    "and",
    "or",
    "et",
    "new",
    "related",
    "none",
    "null",
    "n/a",
    "na",
    "tr",
    "rt",
    "tt",
    "xt",
    "etc",
    "content",
    "analysis",
    "method",
    "results",
    "source",
    "total",
    "team",
    "list",
    "url",
}

_AMBIGUOUS_REFERENCE_ENTITIES = {
    "paper",
    "title",
    "other",
    "others",
    "they",
    "them",
    "their",
    "ours",
    "all",
    "labels",
    "this paper",
    "our paper",
    "our work",
    "this study",
}

_LOCATOR_WITH_TITLE_RE = _re.compile(
    r"^(?:table|figure|fig|eq|equation|formula|section|sec|chapter|ch|appendix|app|page|p|ref|reference|footnote|fn)\.?\s*"
    r"(?:[A-Za-z]?\d+(?:\.\d+)*|[A-Za-z])\s*[:\-]\s*.+$",
    _re.IGNORECASE,
)
_LOCATOR_BARE_LABEL_RE = _re.compile(
    r"^(?:table|figure|fig|eq|equation|formula|section|sec|chapter|ch|appendix|app|page|p|ref|reference|footnote|fn)\.?\s*"
    r"(?:[A-Za-z]?\d+(?:\.\d+)*|[A-Za-z])$",
    _re.IGNORECASE,
)
_LOCATOR_REF_BRACKET_RE = _re.compile(
    r"^(?:ref|reference)\s*\[\s*\d+\s*\]$", _re.IGNORECASE
)
_TABLE_WITH_TITLE_RE = _re.compile(
    r"^table\.?\s*(?:[A-Za-z]?\d+(?:\.\d+)*|[A-Za-z])\s*[:\-]\s*.+$",
    _re.IGNORECASE,
)
_FIGURE_WITH_TITLE_RE = _re.compile(
    r"^(?:figure|fig)\.?\s*(?:[A-Za-z]?\d+(?:\.\d+)*|[A-Za-z])\s*[:\-]\s*.+$",
    _re.IGNORECASE,
)
_EQUATION_WITH_TITLE_RE = _re.compile(
    r"^(?:eq|equation|formula)\.?\s*(?:[A-Za-z]?\d+(?:\.\d+)*|[A-Za-z])\s*[:\-]\s*.+$",
    _re.IGNORECASE,
)
_MODAL_ENTITY_SUFFIX_RE = _re.compile(r"\s*\((?:table|image|equation)\)$", _re.IGNORECASE)
_IMAGE_STEM_NO_EXT_RE = _re.compile(r"^image(?:_|-)[a-z0-9]{4,}$", _re.IGNORECASE)
_PAGE_NUMBER_ENTITY_RE = _re.compile(r"^page\s+\d+$", _re.IGNORECASE)
_COORDINATE_ENTITY_RE = _re.compile(r"^coordinates?\s*\[[^\]]+\]$", _re.IGNORECASE)
_COORDINATE_BRACKET_RE = _re.compile(
    r"\[\s*-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*){1,}\]"
)

_PATH_SEGMENT_SPLIT_RE = _re.compile(r"[\\/]+")
_PATH_SEGMENT_TOKEN_RE = _re.compile(r"[_\-\s]+")
_CONTENT_WINDOWS_PATH_RE = _re.compile(r"(?<!\w)[A-Za-z]:[\\/][^\s\"'`<>|]+")
_CONTENT_POSIX_PATH_RE = _re.compile(r"(?<!\w)/(?:[^\s\"'`<>|]+)")
_CONTENT_REL_PATH_RE = _re.compile(r"(?<!\w)(?:\.{1,2}|~)[\\/][^\s\"'`<>|]+")
_IMAGE_PATH_LINE_RE = _re.compile(r"(?im)^\s*image path\s*:\s*(.+?)\s*$")
# Match full URL spans inside free text so path regexes won't capture URL internals
# such as "//github.com/microsoft/unilm".
_URL_IN_TEXT_RE = _re.compile(r"\b[a-zA-Z][a-zA-Z0-9+\-.]*://[^\s\"'`<>|]+")
_QUERY_IMAGE_PATH_RE = _re.compile(
    r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))",
    _re.IGNORECASE,
)


def _looks_like_math_expression(text: str) -> bool:
    """Return True for common math/LaTeX formatted strings."""
    if _LATEX_CMD_RE.search(text):
        return True
    if _LATEX_DELIM_RE.search(text):
        return True
    return False


def _is_relative_multisegment_path(text: str) -> bool:
    """Return True for relative path-like text with 3+ slash-separated segments."""
    if "/" not in text and "\\" not in text:
        return False

    segments = [seg.strip() for seg in _PATH_SEGMENT_SPLIT_RE.split(text) if seg.strip()]
    if len(segments) < 3:
        return False

    for seg in segments:
        if seg in {".", "..", "~"}:
            continue
        if seg.endswith(":"):
            return False
        if not _re.fullmatch(r"[\w.\- ]+", seg):
            return False

    return True


def _classify_file_or_folder_path(name: str) -> tuple[bool, str]:
    """Classify whether *name* is path-like and return (is_path, reason_code)."""
    if not name:
        return False, "empty"

    cleaned = name.strip().strip("\"'`")
    if not cleaned:
        return False, "empty"

    lower = cleaned.lower()

    # NOT_PATH whitelist (highest priority)
    if _URL_SCHEME_RE.match(cleaned):
        return False, "url_scheme_whitelist"
    if _looks_like_math_expression(cleaned):
        return False, "math_whitelist"
    if lower in _SLASH_PHRASE_WHITELIST or _re.match(r"^[A-Za-z]/[A-Za-z]$", cleaned):
        return False, "slash_phrase_whitelist"

    # HARD_PATH rules
    if _re.match(r"^[A-Za-z]:[\\/]", cleaned):
        return True, "win_abs"
    if _re.match(r"^(?:\\\\|//)[^\\/\s]+[\\/][^\\/\s]+", cleaned):
        return True, "unc_abs"
    if cleaned.startswith("/"):
        return True, "posix_abs"
    if cleaned.startswith(("./", "../", "~/", ".\\", "..\\", "~\\")):
        return True, "relative_prefix"
    if _is_relative_multisegment_path(cleaned):
        return True, "relative_multisegment"
    if _BARE_FILENAME_RE.match(cleaned):
        return True, "bare_filename"

    return False, "not_path"


def _entity_lexical_key(text: str) -> str:
    key = _re.sub(r"[_\-\s]+", " ", text.strip()).casefold()
    return _re.sub(r"\s+", " ", key).strip()


def _extract_path_fragment_keys(path_text: str) -> set[str]:
    if not path_text:
        return set()
    try:
        normalized = unicodedata.normalize("NFKC", str(path_text))
    except Exception:
        normalized = str(path_text)

    parts = [p for p in _PATH_SEGMENT_SPLIT_RE.split(normalized) if p]
    fragment_keys: set[str] = set()
    for part in parts:
        token = part.strip().strip("\"'`")
        if not token or token in {".", ".."}:
            continue

        # Remove file extension for leaf path token
        if "." in token and not token.startswith("."):
            stem = token.rsplit(".", 1)[0]
            if stem:
                token = stem

        token = _PATH_SEGMENT_TOKEN_RE.sub(" ", token)
        key = _entity_lexical_key(token)
        if not key:
            continue

        # Skip trivial path atoms
        if len(key) <= 2 and key.isalpha():
            continue
        if key.isdigit():
            continue

        fragment_keys.add(key)

    return fragment_keys


def _extract_path_like_strings_from_text(source_text: str) -> set[str]:
    if not source_text:
        return set()
    try:
        normalized = unicodedata.normalize("NFKC", str(source_text))
    except Exception:
        normalized = str(source_text)

    candidates: set[str] = set()
    # Exclude complete URL spans before path-pattern scanning. This prevents
    # URL-only tokens from becoming path fragments (e.g., "unilm" from
    # https://github.com/microsoft/unilm).
    masked_for_path_scan = _URL_IN_TEXT_RE.sub(" ", normalized)

    # Prioritize explicit "Image Path: ..." lines from chunk templates.
    for match in _IMAGE_PATH_LINE_RE.finditer(normalized):
        raw = match.group(1).strip().strip("\"'`")
        if not raw:
            continue
        raw = raw.split("[", 1)[0].strip()
        if not raw or _URL_SCHEME_RE.match(raw):
            continue
        candidates.add(raw)

    for pattern in (
        _CONTENT_WINDOWS_PATH_RE,
        _CONTENT_POSIX_PATH_RE,
        _CONTENT_REL_PATH_RE,
    ):
        for match in pattern.finditer(masked_for_path_scan):
            raw = match.group(0).strip().strip("\"'`")
            if not raw or _URL_SCHEME_RE.match(raw):
                continue
            candidates.add(raw)

    return candidates


def _collect_path_fragment_keys(source_text: str = "") -> set[str]:
    fragment_keys: set[str] = set()
    for path_candidate in _extract_path_like_strings_from_text(source_text):
        fragment_keys.update(_extract_path_fragment_keys(path_candidate))
    return fragment_keys


def _is_path_fragment_entity(
    name: str, path_fragment_keys: set[str] | None = None
) -> bool:
    if not name or not path_fragment_keys:
        return False
    key = _entity_lexical_key(name)
    return bool(key and key in path_fragment_keys)


def _looks_like_acronym(word: str) -> bool:
    if not word.isalpha():
        return False
    if not (2 <= len(word) <= 5):
        return False
    vowels = sum(ch in "aeiou" for ch in word.lower())
    return vowels == 0


def _normalize_uppercase_allowlist(raw_allowlist: Any) -> set[str]:
    if raw_allowlist is None:
        return set()

    allowlist_items: list[str] = []
    if isinstance(raw_allowlist, str):
        raw_text = raw_allowlist.strip()
        if not raw_text:
            return set()
        if raw_text.startswith("[") and raw_text.endswith("]"):
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    allowlist_items = [str(item) for item in parsed]
                else:
                    allowlist_items = [raw_text]
            except json.JSONDecodeError:
                allowlist_items = [item.strip() for item in raw_text.split(",")]
        else:
            allowlist_items = [item.strip() for item in raw_text.split(",")]
    elif isinstance(raw_allowlist, (list, tuple, set)):
        allowlist_items = [str(item) for item in raw_allowlist]
    else:
        return set()

    normalized = set()
    for item in allowlist_items:
        cleaned = str(item).strip()
        if cleaned:
            normalized.add(cleaned.lower())
    return normalized


def _normalize_word_case(word: str, uppercase_allowlist: set[str]) -> str:
    if not word:
        return word

    pieces = _re.split(r"([\-_/])", word)
    normalized_pieces: list[str] = []
    for piece in pieces:
        if piece in {"-", "_", "/"}:
            normalized_pieces.append(piece)
            continue
        if not piece:
            continue

        lowered_piece = piece.lower()
        alnum_key = _re.sub(r"[^a-z0-9]", "", lowered_piece)
        if alnum_key and alnum_key in uppercase_allowlist:
            normalized_pieces.append(piece.upper())
            continue
        # Preserve words with meaningful internal capitals (OpenAI, iPhone).
        has_upper = any(ch.isupper() for ch in piece)
        has_lower = any(ch.islower() for ch in piece)
        if has_upper and has_lower and (
            piece[:1].islower() or any(ch.isupper() for ch in piece[1:])
        ):
            normalized_pieces.append(piece)
            continue
        if piece.isupper() and any(ch.isalpha() for ch in piece):
            normalized_pieces.append(piece)
            continue
        if _looks_like_acronym(lowered_piece):
            normalized_pieces.append(piece.upper())
            continue
        normalized_pieces.append(piece.capitalize())

    return "".join(normalized_pieces)


def _normalize_entity_surface(
    name: str,
    uppercase_allowlist: set[str] | None = None,
) -> str:
    # Normalize Unicode and separator artifacts first.
    normalized = unicodedata.normalize("NFKC", name or "")
    normalized = _re.sub(r"(?<=\w)_(?=\w)", " ", normalized.strip())
    normalized = _re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", normalized)
    normalized = _re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    # Rule-based canonical casing:
    # - allowlist/acronym words => uppercase
    # - words with meaningful internal capitals => preserved
    # - otherwise => title-case word by word
    normalized_allowlist = uppercase_allowlist or set()
    words = normalized.split(" ")
    normalized = " ".join(
        _normalize_word_case(word, normalized_allowlist) for word in words
    )

    if _MODAL_ENTITY_SUFFIX_RE.search(normalized):
        return normalized
    if _TABLE_WITH_TITLE_RE.match(normalized):
        return f"{normalized} (table)"
    if _FIGURE_WITH_TITLE_RE.match(normalized):
        return f"{normalized} (image)"
    if _EQUATION_WITH_TITLE_RE.match(normalized):
        return f"{normalized} (equation)"

    return normalized


def _normalize_high_level_keyword(
    keyword: str,
    uppercase_allowlist: set[str],
) -> str:
    normalized = unicodedata.normalize("NFKC", keyword or "")
    normalized = _re.sub(r"\s+", " ", normalized.strip())
    if not normalized:
        return ""

    words = normalized.split(" ")
    lowered_words: list[str] = []
    for word in words:
        canonical_word = _normalize_word_case(word, uppercase_allowlist)
        # Keep explicit uppercase/mixed-case signals; lowercase the rest.
        has_upper = any(ch.isupper() for ch in canonical_word)
        has_lower = any(ch.islower() for ch in canonical_word)
        if canonical_word.isupper() or (has_upper and has_lower and _re.search(r"[A-Z]", canonical_word[1:])):
            lowered_words.append(canonical_word)
        else:
            lowered_words.append(canonical_word.lower())
    return " ".join(lowered_words)


def _normalize_keyword_list(
    keywords: list[Any],
    *,
    keyword_kind: str,
    uppercase_allowlist: set[str],
) -> list[str]:
    normalized_map: dict[str, str] = {}
    for item in keywords or []:
        raw_keyword = str(item).strip()
        if not raw_keyword:
            continue

        if keyword_kind == "low_level":
            normalized_keyword = _normalize_entity_surface(
                raw_keyword, uppercase_allowlist
            )
        else:
            normalized_keyword = _normalize_high_level_keyword(
                raw_keyword,
                uppercase_allowlist,
            )
        if not normalized_keyword:
            continue

        dedupe_key = normalized_keyword.casefold()
        if dedupe_key not in normalized_map:
            normalized_map[dedupe_key] = normalized_keyword
    return list(normalized_map.values())


def _merge_relation_keywords(
    keyword_items: list[str],
    *,
    uppercase_allowlist: set[str],
    enable_case_normalization: bool,
) -> str:
    normalized_map: dict[str, str] = {}
    for keyword_str in keyword_items:
        if not keyword_str:
            continue
        for raw_keyword in str(keyword_str).split(","):
            keyword = raw_keyword.strip()
            if not keyword:
                continue

            normalized_keyword = (
                _normalize_high_level_keyword(keyword, uppercase_allowlist)
                if enable_case_normalization
                else keyword
            )
            dedupe_key = (
                normalized_keyword.casefold()
                if enable_case_normalization
                else normalized_keyword
            )
            if dedupe_key not in normalized_map:
                normalized_map[dedupe_key] = normalized_keyword

    if not normalized_map:
        return ""

    sorted_items = sorted(
        normalized_map.values(),
        key=lambda item: item.casefold() if enable_case_normalization else item,
    )
    return ",".join(sorted_items)


def _strip_modal_suffix(name: str) -> str:
    return _MODAL_ENTITY_SUFFIX_RE.sub("", name or "").strip()


def _has_semantic_locator_title(name: str) -> bool:
    return bool(_LOCATOR_WITH_TITLE_RE.match(_strip_modal_suffix(name)))


def _is_bare_locator_label(name: str) -> bool:
    cleaned = _strip_modal_suffix(name)
    if _LOCATOR_REF_BRACKET_RE.match(cleaned):
        return True
    return bool(_LOCATOR_BARE_LABEL_RE.match(cleaned))


def _classify_low_quality_entity(
    name: str,
    path_fragment_keys: set[str] | None = None,
) -> tuple[bool, str]:
    if not name:
        return True, "empty"
    cleaned = name.strip()
    key = _entity_lexical_key(_strip_modal_suffix(cleaned))

    if _has_semantic_locator_title(cleaned):
        # Keep high-value locator entities with semantic titles, e.g.
        # "Table 7: Ablation Results (table)".
        return False, "locator_with_semantic_title"
    if _is_bare_locator_label(cleaned):
        return True, "bare_locator_label"
    if cleaned.casefold().startswith("image path"):
        return True, "image_path_label"
    if key:
        if path_fragment_keys and key in path_fragment_keys:
            return True, "path_fragment_from_source"
    if key in _LAYOUT_METADATA_ENTITIES:
        return True, "layout_metadata"
    if key in _AMBIGUOUS_REFERENCE_ENTITIES:
        return True, "ambiguous_reference"
    if key in _GENERIC_NOISE_ENTITIES:
        return True, "generic_noise"
    if _PAGE_NUMBER_ENTITY_RE.match(cleaned):
        return True, "page_number_label"
    if _COORDINATE_ENTITY_RE.match(cleaned):
        return True, "coordinate_label"
    if ("bounding box" in key or "coordinate" in key) and _COORDINATE_BRACKET_RE.search(
        cleaned
    ):
        return True, "coordinate_metadata"
    if _IMAGE_STEM_NO_EXT_RE.match(cleaned):
        return True, "image_stem_no_ext"
    if "published by" in key:
        return True, "publisher_phrase_noise"
    if len(key) <= 2 and key.isalpha():
        return True, "short_alpha_noise"

    return False, "ok"


def _build_effective_history_messages(query_param: QueryParam) -> list[dict[str, str]]:
    """Compose effective history payload from summary + raw turns."""
    history_messages: list[dict[str, str]] = []

    history_summary = str(getattr(query_param, "history_summary", "") or "").strip()
    if history_summary:
        history_messages.append(
            {
                "role": "assistant",
                "content": f"Conversation summary:\n{history_summary}",
            }
        )

    for msg in getattr(query_param, "conversation_history", []) or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        content = msg.get("content", "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        else:
            content = str(content)
        content = content.strip()
        if not content:
            continue
        history_messages.append({"role": role, "content": content})

    return history_messages


def _estimate_history_tokens(
    tokenizer: Tokenizer, history_messages: list[dict[str, str]]
) -> int:
    if not history_messages:
        return 0
    # Add a small structural overhead per message for role framing.
    per_message_overhead = 4
    total = 0
    for msg in history_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        total += len(tokenizer.encode(role))
        total += len(tokenizer.encode(content))
        total += per_message_overhead
    return total


def _history_messages_signature(history_messages: list[dict[str, str]]) -> str:
    if not history_messages:
        return ""
    payload = json.dumps(history_messages, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _prompt_signature(prompt: str) -> str:
    if not prompt:
        return ""
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _build_query_cache_params(
    query_param: QueryParam,
    *,
    history_signature: str,
    user_prompt: str,
    system_prompt: str,
    hl_keywords_str: str = "",
    ll_keywords_str: str = "",
) -> dict[str, Any]:
    """Build a canonical cache-parameter payload for query-level LLM cache."""
    return {
        "mode": query_param.mode,
        "response_type": query_param.response_type,
        "stream": query_param.stream,
        "top_k": query_param.top_k,
        "chunk_top_k": query_param.chunk_top_k,
        "rerank_score_scope": query_param.rerank_score_scope,
        "max_entity_tokens": query_param.max_entity_tokens,
        "max_relation_tokens": query_param.max_relation_tokens,
        "max_total_tokens": query_param.max_total_tokens,
        "hl_keywords": hl_keywords_str,
        "ll_keywords": ll_keywords_str,
        "user_prompt": user_prompt,
        "enable_rerank": query_param.enable_rerank,
        "include_references": query_param.include_references,
        "multimodal_top_k": query_param.multimodal_top_k,
        "enable_image_token_budget": query_param.enable_image_token_budget,
        "image_token_estimate_method": query_param.image_token_estimate_method,
        "image_token_model_name_or_path": query_param.image_token_model_name_or_path,
        "image_wrapper_tokens_per_image": query_param.image_wrapper_tokens_per_image,
        "enable_multi_hop": query_param.enable_multi_hop,
        "multi_hop_depth": query_param.multi_hop_depth,
        "ppr_damping": query_param.ppr_damping,
        "ppr_top_k": query_param.ppr_top_k,
        "passage_node_weight": query_param.passage_node_weight,
        "history_signature": history_signature,
        "system_prompt_signature": _prompt_signature(system_prompt),
    }


def _compute_query_cache_args_hash(query: str, query_cache_params: dict[str, Any]) -> str:
    """Compute query-cache hash from user query + canonical cache params."""
    payload = {"query": query, **query_cache_params}
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return compute_args_hash(payload_json)


_QWEN_IMAGE_PROCESSOR_CACHE: dict[str, Any] = {}


def _resolve_image_token_estimate_method(query_param: QueryParam) -> str:
    method = str(
        getattr(query_param, "image_token_estimate_method", "qwen_vl") or "qwen_vl"
    ).strip().lower()
    if method == "qwen_vl":
        return method
    raise ValueError(
        "Unsupported image_token_estimate_method: "
        f"{method}. Set image_token_estimate_method=qwen_vl."
    )


def _get_qwen_image_processor(model_name_or_path: str):
    if not model_name_or_path:
        raise ValueError(
            "Qwen-VL image token estimation requires image_token_model_name_or_path."
        )
    if AutoImageProcessor is None:
        raise RuntimeError(
            "transformers is required for Qwen-VL image token estimation."
        )

    if model_name_or_path not in _QWEN_IMAGE_PROCESSOR_CACHE:
        _QWEN_IMAGE_PROCESSOR_CACHE[model_name_or_path] = (
            AutoImageProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        )
    return _QWEN_IMAGE_PROCESSOR_CACHE[model_name_or_path]


def _estimate_qwen_image_tokens(
    image_path: Path,
    image_processor: Any,
    wrapper_tokens: int,
) -> int:
    if PILImage is None:
        raise RuntimeError("Pillow is required for Qwen-VL image token estimation.")

    with PILImage.open(image_path) as image:
        image_rgb = image.convert("RGB")
        processed = image_processor(images=image_rgb, return_tensors="pt")

    image_grid_thw = processed.get("image_grid_thw")
    if image_grid_thw is None or len(image_grid_thw) <= 0:
        raise RuntimeError("Qwen image processor did not return image_grid_thw.")

    grid_thw = image_grid_thw[0]
    merge_size = max(1, int(getattr(image_processor, "merge_size", 2)))
    visual_tokens = int(grid_thw[0] * grid_thw[1] * grid_thw[2]) // (merge_size**2)
    if visual_tokens <= 0:
        raise RuntimeError(
            f"Invalid Qwen visual token estimate for image: {image_path.as_posix()}"
        )
    return visual_tokens + wrapper_tokens


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _chunk_token_cost(tokenizer: Tokenizer, chunk: dict) -> int:
    # Keep token accounting aligned with process_chunks_unified Step 4,
    # where truncation happens before synthetic "id" is added in Step 5.
    if "id" in chunk:
        chunk = {k: v for k, v in chunk.items() if k != "id"}
    return len(
        tokenizer.encode(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in [chunk])
        )
    )


def _extract_image_paths_from_chunk(chunk: dict) -> list[str]:
    content = str(chunk.get("content", "") or "")
    image_paths: list[str] = []
    seen_paths: set[str] = set()
    for match in _QUERY_IMAGE_PATH_RE.finditer(content):
        image_path = match.group(1).strip().strip("\"'`")
        if not image_path or image_path in seen_paths:
            continue
        seen_paths.add(image_path)
        image_paths.append(image_path)
    return image_paths


def _build_lazy_qwen_image_token_estimator(
    query_param: QueryParam,
) -> Callable[[str], int | None]:
    wrapper_tokens = max(
        0, int(getattr(query_param, "image_wrapper_tokens_per_image", 2))
    )
    token_cache: dict[str, int | None] = {}
    image_processor: Any = None
    processor_initialized = False

    def _ensure_processor():
        nonlocal image_processor, processor_initialized
        if processor_initialized:
            return
        processor_initialized = True
        if PILImage is None:
            raise RuntimeError("Pillow is required for Qwen-VL image token estimation.")
        _resolve_image_token_estimate_method(query_param)
        model_name_or_path = str(
            getattr(query_param, "image_token_model_name_or_path", "")
            or os.getenv("IMAGE_TOKEN_MODEL_NAME_OR_PATH", "")
            or os.getenv("VISION_MODEL_NAME", "")
        ).strip()
        image_processor = _get_qwen_image_processor(model_name_or_path)

    def _estimate(image_path: str) -> int | None:
        if image_path in token_cache:
            return token_cache[image_path]

        path_obj = Path(image_path)
        if not path_obj.exists():
            token_cache[image_path] = None
            return None

        _ensure_processor()
        try:
            token_value = _estimate_qwen_image_tokens(
                image_path=path_obj,
                image_processor=image_processor,
                wrapper_tokens=wrapper_tokens,
            )
            token_cache[image_path] = token_value
            return token_value
        except Exception:
            token_cache[image_path] = None
            return None

    return _estimate


def _is_file_or_folder_path(name: str) -> bool:
    """Backward-compatible bool wrapper for path-like classification."""
    is_path, _reason = _classify_file_or_folder_path(name)
    return is_path


def _truncate_entity_identifier(
    identifier: str, limit: int, chunk_key: str, identifier_role: str
) -> str:
    """Truncate entity identifiers that exceed the configured length limit."""

    if len(identifier) <= limit:
        return identifier

    display_value = identifier[:limit]
    preview = identifier[:20]  # Show first 20 characters as preview
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        chunk_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=chunk[:120],
                    )
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    for start in range(
                        0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + chunk_token_size]
                        )
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append(
                {
                    "tokens": min(chunk_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    seperator: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    if len(description_list) == 1:
        return description_list[0], False

    # Get configuration
    tokenizer: Tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list
        total_tokens = sum(len(tokenizer.encode(desc)) for desc in current_list)

        # If total length is within limits, perform final summarization
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                final_description = seperator.join(current_list)
                return final_description if final_description else "", llm_was_used
            else:
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list
        for i, desc in enumerate(current_list):
            desc_tokens = len(tokenizer.encode(desc))

            # If adding current description would exceed limit, finalize current chunk
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # curren_chunk is ready for summary in reduce phase
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # Reduce phase: summarize each group from chunks
        new_summaries = []
        for chunk in chunks:
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    global_config,
                    llm_response_cache,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        current_list = new_summaries


async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        descriptions: List of description strings to summarize
        global_config: Global configuration containing LLM function and settings
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Summarized description string
    """
    use_llm_func: callable = global_config["llm_model_func"]
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=8)

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    summary_length_recommended = global_config["summary_length_recommended"]

    prompt_template = PROMPTS["summarize_entity_descriptions"]

    # Convert descriptions to JSONL format and apply token-based truncation
    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]

    # Create list of JSON objects with "Description" field
    json_descriptions = [{"Description": desc} for desc in description_list]

    # Use truncate_list_by_token_size for length truncation
    truncated_json_descriptions = truncate_list_by_token_size(
        json_descriptions,
        key=lambda x: json.dumps(x, ensure_ascii=False),
        max_token_size=summary_context_size,
        tokenizer=tokenizer,
    )

    # Convert to JSONL format (one JSON object per line)
    joined_descriptions = "\n".join(
        json.dumps(desc, ensure_ascii=False) for desc in truncated_json_descriptions
    )

    # Prepare context for the prompt
    context_base = dict(
        description_type=description_type,
        description_name=description_name,
        description_list=joined_descriptions,
        summary_length=summary_length_recommended,
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)

    # Use LLM function with cache (higher priority for summary generation)
    summary, _ = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="summary",
    )

    # Check summary token length against embedding limit
    embedding_token_limit = global_config.get("embedding_token_limit")
    if embedding_token_limit is not None and summary:
        tokenizer = global_config["tokenizer"]
        summary_token_count = len(tokenizer.encode(summary))
        threshold = int(embedding_token_limit)

        if summary_token_count > threshold:
            logger.warning(
                f"Summary tokens({summary_token_count}) exceeds embedding_token_limit({embedding_token_limit}) "
                f" for {description_type}: {description_name}"
            )

    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    path_fragment_keys: set[str] | None = None,
    enable_entity_surface_normalization: bool = False,
    entity_uppercase_allowlist: set[str] | None = None,
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 feilds on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.info(
                f"Empty entity name found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        # Filter out file/folder path entities
        is_path_entity, entity_path_reason = _classify_file_or_folder_path(entity_name)
        if is_path_entity:
            logger.info(
                f"Filtered file path entity [reason={entity_path_reason}]: '{entity_name}'"
            )
            return None

        if _is_path_fragment_entity(entity_name, path_fragment_keys):
            logger.info(
                f"Filtered path fragment entity [reason=path_fragment_from_source]: '{entity_name}'"
            )
            return None

        if enable_entity_surface_normalization:
            entity_name = _normalize_entity_surface(
                entity_name, entity_uppercase_allowlist
            )
            if not entity_name:
                logger.info(
                    f"Empty entity name after normalization. Original: '{record_attributes[1]}'"
                )
                return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        # Handle comma-separated entity types by finding the first non-empty token
        if "," in entity_type:
            original = entity_type
            tokens = [t.strip() for t in entity_type.split(",")]
            non_empty = [t for t in tokens if t]
            if not non_empty:
                logger.warning(
                    f"Entity extraction error: all tokens empty after comma-split: '{original}'"
                )
                return None
            entity_type = non_empty[0]
            logger.warning(
                f"Entity type contains comma, taking first non-empty token: '{original}' -> '{entity_type}'"
            )

        # Remove spaces and convert to lowercase
        entity_type = entity_type.replace(" ", "").lower()

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    path_fragment_keys: set[str] | None = None,
    enable_entity_surface_normalization: bool = False,
    entity_uppercase_allowlist: set[str] | None = None,
):
    if (
        len(record_attributes) != 5 or "relation" not in record_attributes[0]
    ):  # treat "relationship" and "relation" interchangeable
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on REALTION `{record_attributes[1]}`~`{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        # Validate entity names after all cleaning steps
        if not source:
            logger.info(
                f"Empty source entity found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.info(
                f"Empty target entity found after sanitization. Original: '{record_attributes[2]}'"
            )
            return None

        # Filter out relationships involving file path entities
        source_is_path, source_path_reason = _classify_file_or_folder_path(source)
        target_is_path, target_path_reason = _classify_file_or_folder_path(target)
        if source_is_path or target_is_path:
            reason_parts = []
            if source_is_path:
                reason_parts.append(f"src:{source_path_reason}")
            if target_is_path:
                reason_parts.append(f"tgt:{target_path_reason}")
            logger.info(
                f"Filtered relationship with path entity [reason={', '.join(reason_parts)}]: '{source}' -> '{target}'"
            )
            return None

        source_is_path_fragment = _is_path_fragment_entity(source, path_fragment_keys)
        target_is_path_fragment = _is_path_fragment_entity(target, path_fragment_keys)
        if source_is_path_fragment or target_is_path_fragment:
            reason_parts = []
            if source_is_path_fragment:
                reason_parts.append("src:path_fragment_from_source")
            if target_is_path_fragment:
                reason_parts.append("tgt:path_fragment_from_source")
            logger.info(
                f"Filtered relationship with path-fragment entity [reason={', '.join(reason_parts)}]: '{source}' -> '{target}'"
            )
            return None

        if enable_entity_surface_normalization:
            source = _normalize_entity_surface(source, entity_uppercase_allowlist)
            target = _normalize_entity_surface(target, entity_uppercase_allowlist)
            if not source or not target:
                logger.info(
                    f"Empty relation endpoint after normalization in chunk {chunk_key}"
                )
                return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        # Process keywords with same cleaning pipeline
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        # Process relationship description with same cleaning pipeline
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])

        edge_source_id = chunk_key
        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )

        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            keywords=edge_keywords,
            source_id=edge_source_id,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, list[str]],
    relationships_to_rebuild: dict[tuple[str, str], list[str]],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
) -> None:
    """Rebuild entity and relationship descriptions from cached extraction results with parallel processing

    This method uses cached LLM extraction results instead of calling LLM again,
    following the same approach as the insert process. Now with parallel processing
    controlled by llm_model_max_async and using get_storage_keyed_lock for data consistency.

    Args:
        entities_to_rebuild: Dict mapping entity_name -> list of remaining chunk_ids
        relationships_to_rebuild: Dict mapping (src, tgt) -> list of remaining chunk_ids
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_storage: Text chunks storage
        llm_response_cache: LLM response cache
        global_config: Global configuration containing llm_model_max_async
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        entity_chunks_storage: KV storage maintaining full chunk IDs per entity
        relation_chunks_storage: KV storage maintaining full chunk IDs per relation
    """
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Get all referenced chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)

    status_message = f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk extractions (parallel processing)"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Get cached extraction results for these chunks using storage
    # cached_results： chunk_id -> [list of (extraction_result, create_time) from LLM cache sorted by create_time of the first extraction_result]
    cached_results = await _get_cached_extraction_results(
        llm_response_cache,
        all_referenced_chunk_ids,
        text_chunks_storage=text_chunks_storage,
    )

    if not cached_results:
        status_message = (
            "No cached extraction results found, falling back to source-only rebuild"
        )
        logger.warning(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)

    # Process cached results to get entities and relationships for each chunk
    chunk_entities = {}  # chunk_id -> {entity_name: [entity_data]}
    chunk_relationships = {}  # chunk_id -> {(src, tgt): [relationship_data]}

    for chunk_id, results in cached_results.items():
        try:
            # Handle multiple extraction results per chunk
            chunk_entities[chunk_id] = defaultdict(list)
            chunk_relationships[chunk_id] = defaultdict(list)

            # process multiple LLM extraction results for a single chunk_id
            for result in results:
                entities, relationships = await _rebuild_from_extraction_result(
                    text_chunks_storage=text_chunks_storage,
                    chunk_id=chunk_id,
                    extraction_result=result[0],
                    timestamp=result[1],
                    global_config=global_config,
                )

                # Merge entities and relationships from this extraction result
                # Compare description lengths and keep the better version for the same chunk_id
                for entity_name, entity_list in entities.items():
                    if entity_name not in chunk_entities[chunk_id]:
                        # New entity for this chunk_id
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    elif len(chunk_entities[chunk_id][entity_name]) == 0:
                        # Empty list, add the new entities
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_entities[chunk_id][entity_name][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(entity_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new entity that has longer description
                            chunk_entities[chunk_id][entity_name] = list(entity_list)
                        # Otherwise keep existing version

                # Compare description lengths and keep the better version for the same chunk_id
                for rel_key, rel_list in relationships.items():
                    if rel_key not in chunk_relationships[chunk_id]:
                        # New relationship for this chunk_id
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    elif len(chunk_relationships[chunk_id][rel_key]) == 0:
                        # Empty list, add the new relationships
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_relationships[chunk_id][rel_key][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(rel_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new relationship that has longer description
                            chunk_relationships[chunk_id][rel_key] = list(rel_list)
                        # Otherwise keep existing version

        except Exception as e:
            status_message = (
                f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
            )
            logger.info(status_message)  # Per requirement, change to info
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            continue

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0

    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        entity_chunks_storage=entity_chunks_storage,
                    )
                    rebuilt_entities_count += 1
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild `{entity_name}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # Sort src and tgt to ensure order-independent lock key generation
            sorted_key_parts = sorted([src, tgt])
            async with get_storage_keyed_lock(
                sorted_key_parts,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        entities_vdb=entities_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        relation_chunks_storage=relation_chunks_storage,
                        entity_chunks_storage=entity_chunks_storage,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                    )
                    rebuilt_relationships_count += 1
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild `{src}`~`{tgt}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    # Create tasks for parallel processing
    tasks = []

    # Add entity rebuilding tasks
    for entity_name, chunk_ids in entities_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
        tasks.append(task)

    # Add relationship rebuilding tasks
    for (src, tgt), chunk_ids in relationships_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
        tasks.append(task)

    # Log parallel processing start
    status_message = f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Execute all tasks in parallel with semaphore control and early failure detection
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                # Task completed successfully, retrieve result to mark as processed
                task.result()
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Re-raise the first exception to notify the caller
        raise first_exception

    # Final status report
    status_message = f"KG rebuild completed: {rebuilt_entities_count} entities and {rebuilt_relationships_count} relationships rebuilt successfully."
    if failed_entities_count > 0 or failed_relationships_count > 0:
        status_message += f" Failed: {failed_entities_count} entities, {failed_relationships_count} relationships."

    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _get_cached_extraction_results(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
) -> dict[str, list[str]]:
    """Get cached extraction results for specific chunk IDs

    This function retrieves cached LLM extraction results for the given chunk IDs and returns
    them sorted by creation time. The results are sorted at two levels:
    1. Individual extraction results within each chunk are sorted by create_time (earliest first)
    2. Chunks themselves are sorted by the create_time of their earliest extraction result

    Args:
        llm_response_cache: LLM response cache storage
        chunk_ids: Set of chunk IDs to get cached results for
        text_chunks_storage: Text chunks storage for retrieving chunk data and LLM cache references

    Returns:
        Dict mapping chunk_id -> list of extraction_result_text, where:
        - Keys (chunk_ids) are ordered by the create_time of their first extraction result
        - Values (extraction results) are ordered by create_time within each chunk
    """
    cached_results = {}

    # Collect all LLM cache IDs from chunks
    all_cache_ids = set()

    # Read from storage
    chunk_data_list = await text_chunks_storage.get_by_ids(list(chunk_ids))
    for chunk_data in chunk_data_list:
        if chunk_data and isinstance(chunk_data, dict):
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if llm_cache_list:
                all_cache_ids.update(llm_cache_list)
        else:
            logger.warning(f"Chunk data is invalid or None: {chunk_data}")

    if not all_cache_ids:
        logger.warning(f"No LLM cache IDs found for {len(chunk_ids)} chunk IDs")
        return cached_results

    # Batch get LLM cache entries
    cache_data_list = await llm_response_cache.get_by_ids(list(all_cache_ids))

    # Process cache entries and group by chunk_id
    valid_entries = 0
    for cache_entry in cache_data_list:
        if (
            cache_entry is not None
            and isinstance(cache_entry, dict)
            and cache_entry.get("cache_type") == "extract"
            and cache_entry.get("chunk_id") in chunk_ids
        ):
            chunk_id = cache_entry["chunk_id"]
            extraction_result = cache_entry["return"]
            create_time = cache_entry.get(
                "create_time", 0
            )  # Get creation time, default to 0
            valid_entries += 1

            # Support multiple LLM caches per chunk
            if chunk_id not in cached_results:
                cached_results[chunk_id] = []
            # Store tuple with extraction result and creation time for sorting
            cached_results[chunk_id].append((extraction_result, create_time))

    # Sort extraction results by create_time for each chunk and collect earliest times
    chunk_earliest_times = {}
    for chunk_id in cached_results:
        # Sort by create_time (x[1]), then extract only extraction_result (x[0])
        cached_results[chunk_id].sort(key=lambda x: x[1])
        # Store the earliest create_time for this chunk (first item after sorting)
        chunk_earliest_times[chunk_id] = cached_results[chunk_id][0][1]

    # Sort cached_results by the earliest create_time of each chunk
    sorted_chunk_ids = sorted(
        chunk_earliest_times.keys(), key=lambda chunk_id: chunk_earliest_times[chunk_id]
    )

    # Rebuild cached_results in sorted order
    sorted_cached_results = {}
    for chunk_id in sorted_chunk_ids:
        sorted_cached_results[chunk_id] = cached_results[chunk_id]

    logger.info(
        f"Found {valid_entries} valid cache entries, {len(sorted_cached_results)} chunks with results"
    )
    return sorted_cached_results  # each item: list(extraction_result, create_time)


async def _process_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    source_text: str = "",
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
    enable_entity_surface_normalization: bool = DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION,
    entity_uppercase_allowlist: Any = None,
) -> tuple[dict, dict]:
    """Process a single extraction result (either initial or gleaning)
    Args:
        result (str): The extraction result to process
        chunk_key (str): The chunk key for source tracking
        file_path (str): The file path for citation
        tuple_delimiter (str): Delimiter for tuple fields
        record_delimiter (str): Delimiter for records
        completion_delimiter (str): Delimiter for completion
    Returns:
        tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
    """
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    path_fragment_keys = _collect_path_fragment_keys(source_text)
    normalized_allowlist = _normalize_uppercase_allowlist(
        entity_uppercase_allowlist
        if entity_uppercase_allowlist is not None
        else DEFAULT_ENTITY_UPPERCASE_ALLOWLIST
    )

    if completion_delimiter not in result:
        logger.warning(
            f"{chunk_key}: Complete delimiter can not be found in extraction result"
        )

    # Split LLL output result to records by "\n"
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )

    # Fix LLM output format error which use tuple_delimiter to seperate record instead of "\n"
    fixed_records = []
    for record in records:
        record = record.strip()
        if record is None:
            continue
        entity_records = split_string_by_multi_markers(
            record, [f"{tuple_delimiter}entity{tuple_delimiter}"]
        )
        for entity_record in entity_records:
            if not entity_record.startswith("entity") and not entity_record.startswith(
                "relation"
            ):
                entity_record = f"entity<|{entity_record}"
            entity_relation_records = split_string_by_multi_markers(
                # treat "relationship" and "relation" interchangeable
                entity_record,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            for entity_relation_record in entity_relation_records:
                if not entity_relation_record.startswith(
                    "entity"
                ) and not entity_relation_record.startswith("relation"):
                    entity_relation_record = (
                        f"relation{tuple_delimiter}{entity_relation_record}"
                    )
                fixed_records = fixed_records + [entity_relation_record]

    if len(fixed_records) != len(records):
        logger.warning(
            f"{chunk_key}: LLM output format error; find LLM use {tuple_delimiter} as record seperators instead new-line"
        )

    for record in fixed_records:
        record = record.strip()
        if record is None:
            continue

        # Fix various forms of tuple_delimiter corruption from the LLM output using the dedicated function
        delimiter_core = tuple_delimiter[2:-2]  # Extract "#" from "<|#|>"
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        if delimiter_core != delimiter_core.lower():
            # change delimiter_core to lower case, and fix again
            delimiter_core = delimiter_core.lower()
            record = fix_tuple_delimiter_corruption(
                record, delimiter_core, tuple_delimiter
            )

        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        # Try to parse as entity
        entity_data = await _handle_single_entity_extraction(
            record_attributes,
            chunk_key,
            timestamp,
            file_path,
            path_fragment_keys,
            enable_entity_surface_normalization=enable_entity_surface_normalization,
            entity_uppercase_allowlist=normalized_allowlist,
        )
        if entity_data is not None:
            truncated_name = _truncate_entity_identifier(
                entity_data["entity_name"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            continue

        # Try to parse as relationship
        relationship_data = await _handle_single_relationship_extraction(
            record_attributes,
            chunk_key,
            timestamp,
            file_path,
            path_fragment_keys,
            enable_entity_surface_normalization=enable_entity_surface_normalization,
            entity_uppercase_allowlist=normalized_allowlist,
        )
        if relationship_data is not None:
            truncated_source = _truncate_entity_identifier(
                relationship_data["src_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            truncated_target = _truncate_entity_identifier(
                relationship_data["tgt_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            relationship_data["src_id"] = truncated_source
            relationship_data["tgt_id"] = truncated_target
            maybe_edges[(truncated_source, truncated_target)].append(relationship_data)

    return dict(maybe_nodes), dict(maybe_edges)


async def _rebuild_from_extraction_result(
    text_chunks_storage: BaseKVStorage,
    extraction_result: str,
    chunk_id: str,
    timestamp: int,
    global_config: dict[str, Any],
) -> tuple[dict, dict]:
    """Parse cached extraction result using the same logic as extract_entities

    Args:
        text_chunks_storage: Text chunks storage to get chunk data
        extraction_result: The cached LLM extraction result
        chunk_id: The chunk ID for source tracking

    Returns:
        Tuple of (entities_dict, relationships_dict)
    """

    # Get chunk data for file_path from storage
    chunk_data = await text_chunks_storage.get_by_id(chunk_id)
    file_path = (
        chunk_data.get("file_path", "unknown_source")
        if chunk_data
        else "unknown_source"
    )
    chunk_content = chunk_data.get("content", "") if chunk_data else ""

    # Call the shared processing function
    return await _process_extraction_result(
        extraction_result,
        chunk_id,
        timestamp,
        file_path,
        source_text=chunk_content,
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        enable_entity_surface_normalization=bool(
            global_config.get(
                "enable_entity_surface_normalization",
                DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION,
            )
        ),
        entity_uppercase_allowlist=global_config.get(
            "entity_uppercase_allowlist",
            DEFAULT_ENTITY_UPPERCASE_ALLOWLIST,
        ),
    )


async def _rebuild_single_entity(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name: str,
    chunk_ids: list[str],
    chunk_entities: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    entity_chunks_storage: BaseKVStorage | None = None,
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild a single entity from cached extraction results"""

    # Get current entity data
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    if not current_entity:
        return

    # Helper function to update entity in both graph and vector storage
    async def _update_entity_storage(
        final_description: str,
        entity_type: str,
        file_paths: list[str],
        source_chunk_ids: list[str],
        truncation_info: str = "",
    ):
        try:
            # Update entity in graph storage (critical path)
            updated_entity_data = {
                **current_entity,
                "description": final_description,
                "entity_type": entity_type,
                "source_id": GRAPH_FIELD_SEP.join(source_chunk_ids),
                "file_path": GRAPH_FIELD_SEP.join(file_paths)
                if file_paths
                else current_entity.get("file_path", "unknown_source"),
                "created_at": int(time.time()),
                "truncate": truncation_info,
            }
            _disambig = global_config.get("enable_entity_disambiguation", True)
            composite_id = compute_entity_id(entity_name, entity_type, _disambig)
            updated_entity_data["entity_id"] = composite_id
            await knowledge_graph_inst.upsert_node(composite_id, updated_entity_data)

            # Update entity in vector database (equally critical)
            entity_vdb_id = compute_entity_vdb_id(entity_name, entity_type, _disambig)
            entity_content = f"{entity_name}\n{final_description}"

            vdb_data = {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_id": composite_id,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }

            # Use safe operation wrapper - VDB failure must throw exception
            await safe_vdb_operation_with_exception(
                operation=lambda: entities_vdb.upsert(vdb_data),
                operation_name="rebuild_entity_upsert",
                entity_name=entity_name,
                max_retries=3,
                retry_delay=0.1,
            )

        except Exception as e:
            error_msg = f"Failed to update entity storage for `{entity_name}`: {e}"
            logger.error(error_msg)
            raise  # Re-raise exception

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if entity_chunks_storage is not None and normalized_chunk_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_entity"],
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # Collect all entity data from relevant (limited) chunks
    all_entity_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_entities and entity_name in chunk_entities[chunk_id]:
            all_entity_data.extend(chunk_entities[chunk_id][entity_name])

    if not all_entity_data:
        logger.warning(
            f"No entity data found for `{entity_name}`, trying to rebuild from relationships"
        )

        # Get all edges connected to this entity
        edges = await knowledge_graph_inst.get_node_edges(entity_name)
        if not edges:
            logger.warning(f"No relations attached to entity `{entity_name}`")
            return

        # Collect relationship data to extract entity information
        relationship_descriptions = []
        file_paths = set()

        # Get edge data for all connected relationships
        for src_id, tgt_id in edges:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data:
                if edge_data.get("description"):
                    relationship_descriptions.append(edge_data["description"])

                if edge_data.get("file_path"):
                    edge_file_paths = edge_data["file_path"].split(GRAPH_FIELD_SEP)
                    file_paths.update(edge_file_paths)

        # deduplicate descriptions
        description_list = list(dict.fromkeys(relationship_descriptions))

        # Generate final description from relationships or fallback to current
        if description_list:
            final_description, _ = await _handle_entity_relation_summary(
                "Entity",
                entity_name,
                description_list,
                GRAPH_FIELD_SEP,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            final_description = current_entity.get("description", "")

        entity_type = current_entity.get("entity_type", "UNKNOWN")
        await _update_entity_storage(
            final_description,
            entity_type,
            file_paths,
            limited_chunk_ids,
        )
        return

    # Process cached entity data
    descriptions = []
    entity_types = []
    file_paths_list = []
    seen_paths = set()

    for entity_data in all_entity_data:
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        if entity_data.get("file_path"):
            file_path = entity_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply MAX_FILE_PATHS limit
    max_file_paths = global_config.get("max_file_paths")
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{entity_name}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    entity_types = list(dict.fromkeys(entity_types))

    # Get most common entity type
    entity_type = (
        max(set(entity_types), key=entity_types.count)
        if entity_types
        else current_entity.get("entity_type", "UNKNOWN")
    )

    # Generate final description from entities or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = current_entity.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    await _update_entity_storage(
        final_description,
        entity_type,
        file_paths_list,
        limited_chunk_ids,
        truncation_info,
    )

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{entity_name}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    logger.info(status_message)
    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _rebuild_single_relationship(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    src: str,
    tgt: str,
    chunk_ids: list[str],
    chunk_relationships: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild a single relationship from cached extraction results

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    if not current_relationship:
        return

    strict_endpoint_match = bool(
        global_config.get(
            "strict_relation_endpoint_entity_match",
            DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH,
        )
    )
    if strict_endpoint_match:
        missing_endpoints: list[str] = []
        if not await knowledge_graph_inst.has_node(src):
            missing_endpoints.append(src)
        if not await knowledge_graph_inst.has_node(tgt):
            missing_endpoints.append(tgt)
        if missing_endpoints:
            await _remove_relation_edge_and_vector(
                knowledge_graph_inst=knowledge_graph_inst,
                relationships_vdb=relationships_vdb,
                src_id=src,
                tgt_id=tgt,
            )
            status_message = (
                f"Skipped rebuild `{src}`~`{tgt}`: strict endpoint match enabled, "
                f"missing endpoints={','.join(sorted(set(missing_endpoints)))}"
            )
            logger.info(status_message)
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            return

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if relation_chunks_storage is not None and normalized_chunk_ids:
        storage_key = make_relation_chunk_key(src, tgt)
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_relation"],
        limit_method,
        identifier=f"`{src}`~`{tgt}`",
    )

    # Collect all relationship data from relevant chunks
    all_relationship_data = []
    for chunk_id in limited_chunk_ids:
        if chunk_id in chunk_relationships:
            # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
            for edge_key in [(src, tgt), (tgt, src)]:
                if edge_key in chunk_relationships[chunk_id]:
                    all_relationship_data.extend(
                        chunk_relationships[chunk_id][edge_key]
                    )

    current_is_factual = _is_factual_or_legacy_edge(current_relationship)
    fallback_without_cache = not all_relationship_data
    if fallback_without_cache:
        logger.warning(
            f"No relation data found for `{src}-{tgt}`, falling back to source-only rebuild"
        )

    # Merge descriptions and keywords
    descriptions = []
    keywords = []
    weights = []
    file_paths_list = []
    seen_paths = set()

    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            keywords.append(rel_data["keywords"])
        if "weight" in rel_data and rel_data.get("weight") is not None:
            weights.append(rel_data.get("weight"))
        if rel_data.get("file_path"):
                file_path = rel_data["file_path"]
                if file_path and file_path not in seen_paths:
                    file_paths_list.append(file_path)
                    seen_paths.add(file_path)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths")
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{src}`~`{tgt}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    keywords = list(dict.fromkeys(keywords))
    enable_keyword_case_normalization = bool(
        global_config.get(
            "enable_keyword_case_normalization",
            DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION,
        )
    )
    uppercase_allowlist = _normalize_uppercase_allowlist(
        global_config.get(
            "entity_uppercase_allowlist",
            DEFAULT_ENTITY_UPPERCASE_ALLOWLIST,
        )
    )
    if keywords:
        combined_keywords = _merge_relation_keywords(
            keywords,
            uppercase_allowlist=uppercase_allowlist,
            enable_case_normalization=enable_keyword_case_normalization,
        )
    else:
        combined_keywords = (
            current_relationship.get("keywords", "") if current_is_factual else ""
        )
        if enable_keyword_case_normalization and combined_keywords:
            combined_keywords = _merge_relation_keywords(
                [combined_keywords],
                uppercase_allowlist=uppercase_allowlist,
                enable_case_normalization=True,
            )

    if weights:
        weight_raw = sum(_to_non_negative_float(weight, default=1.0) for weight in weights)
    else:
        current_raw = (
            _extract_existing_factual_weight_raw(current_relationship)
            if current_is_factual
            else 1.0
        )
        if fallback_without_cache and current_is_factual:
            current_source_ids = split_string_by_multi_markers(
                str(current_relationship.get("source_id", "") or ""), [GRAPH_FIELD_SEP]
            )
            current_source_count = len([chunk_id for chunk_id in current_source_ids if chunk_id])
            remaining_source_count = len([chunk_id for chunk_id in limited_chunk_ids if chunk_id])
            if current_source_count > 0 and remaining_source_count > 0:
                weight_raw = current_raw * (remaining_source_count / current_source_count)
            elif remaining_source_count > 0:
                weight_raw = float(remaining_source_count)
            else:
                weight_raw = current_raw
        else:
            weight_raw = current_raw
    weight = _factual_weight_from_raw(weight_raw)

    # Generate final description from relations or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Relation",
            f"{src}-{tgt}",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = (
            current_relationship.get("description", "") if current_is_factual else ""
        )

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    edge_created_at = current_relationship.get("created_at")
    if edge_created_at is None:
        edge_created_at = int(time.time())

    # Update relationship in graph storage with a factual full template.
    updated_relationship_data = {
        "description": final_description,
        "keywords": combined_keywords,
        "weight": weight,
        "weight_raw": weight_raw,
        "source_id": GRAPH_FIELD_SEP.join(limited_chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join([fp for fp in file_paths_list if fp])
        if file_paths_list
        else (
            current_relationship.get("file_path", "unknown_source")
            if current_is_factual
            else "unknown_source"
        ),
        "truncate": truncation_info,
        "created_at": edge_created_at,
        "edge_type": FACTUAL_EDGE_TYPE,
        "provenance": FACTUAL_EDGE_PROVENANCE,
    }

    # Ensure both endpoint nodes exist before writing the edge back
    # (certain storage backends require pre-existing nodes).
    node_description = (
        updated_relationship_data["description"]
        if updated_relationship_data.get("description")
        else (current_relationship.get("description", "") if current_is_factual else "")
    )
    node_source_id = updated_relationship_data.get("source_id", "")
    node_file_path = updated_relationship_data.get("file_path", "unknown_source")

    for node_id in {src, tgt}:
        if not (await knowledge_graph_inst.has_node(node_id)):
            node_created_at = int(time.time())
            node_data = {
                "entity_id": node_id,
                "source_id": node_source_id,
                "description": node_description,
                "entity_type": "UNKNOWN",
                "file_path": node_file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(node_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None and limited_chunk_ids:
                await entity_chunks_storage.upsert(
                    {
                        node_id: {
                            "chunk_ids": limited_chunk_ids,
                            "count": len(limited_chunk_ids),
                        }
                    }
                )

            # Update entity_vdb for the newly created entity
            if entities_vdb is not None:
                _disambig = global_config.get("enable_entity_disambiguation", True)
                entity_vdb_id = compute_entity_vdb_id(node_id, "UNKNOWN", _disambig)
                _composite_id = compute_entity_id(node_id, "UNKNOWN", _disambig)
                entity_content = f"{node_id}\n{node_description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_id": _composite_id,
                        "entity_name": node_id,
                        "source_id": node_source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": node_file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entities_vdb.upsert(payload),
                    operation_name="rebuild_added_entity_upsert",
                    entity_name=node_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    # Sort src and tgt to ensure consistent ordering (smaller string first)
    if src > tgt:
        src, tgt = tgt, src
    try:
        rel_vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt + src, prefix="rel-")

        # Delete old vector records first (both directions to be safe)
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )

        # Insert new vector record
        rel_content = f"{combined_keywords}\t{src}\n{tgt}\n{final_description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": updated_relationship_data["source_id"],
                "content": rel_content,
                "keywords": combined_keywords,
                "description": final_description,
                "weight": weight,
                "weight_raw": weight_raw,
                "file_path": updated_relationship_data["file_path"],
                "edge_type": FACTUAL_EDGE_TYPE,
                "provenance": FACTUAL_EDGE_PROVENANCE,
            }
        }

        # Use safe operation wrapper - VDB failure must throw exception
        await safe_vdb_operation_with_exception(
            operation=lambda: relationships_vdb.upsert(vdb_data),
            operation_name="rebuild_relationship_upsert",
            entity_name=f"{src}-{tgt}",
            max_retries=3,
            retry_delay=0.2,
        )

    except Exception as e:
        error_msg = f"Failed to rebuild relationship storage for `{src}-{tgt}`: {e}"
        logger.error(error_msg)
        raise  # Re-raise exception

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{src}`~`{tgt}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    # Add truncation info from apply_source_ids_limit if truncation occurred
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f" ({limit_method}:{len(limited_chunk_ids)}/{len(normalized_chunk_ids)})"
        )
        status_message += truncation_info

    logger.info(status_message)

    # Update pipeline status
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    incoming_entity_name = ""
    for dp in nodes_data:
        candidate_name = dp.get("entity_name") if isinstance(dp, dict) else None
        if isinstance(candidate_name, str) and candidate_name.strip():
            incoming_entity_name = candidate_name.strip()
            break

    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    # 1. Get existing node data from knowledge graph
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node:
        existing_entity_type = already_node.get("entity_type")
        # Coerce to str before any string operations: non-string values from
        # API/custom graph paths would otherwise raise TypeError on the comma check.
        if (
            not isinstance(existing_entity_type, str)
            or not existing_entity_type.strip()
        ):
            existing_entity_type = "UNKNOWN"
        # Sanitize entity_type read back from DB to prevent dirty data from propagating
        if "," in existing_entity_type:
            original = existing_entity_type
            tokens = [t.strip() for t in existing_entity_type.split(",")]
            non_empty = [t for t in tokens if t]
            existing_entity_type = non_empty[0] if non_empty else "UNKNOWN"
            logger.warning(
                f"Entity type read from DB contains comma, taking first non-empty token: '{original}' -> '{existing_entity_type}'"
            )
        already_entity_types.append(existing_entity_type)
        already_source_ids.extend(already_node["source_id"].split(GRAPH_FIELD_SEP))
        already_file_paths.extend(already_node["file_path"].split(GRAPH_FIELD_SEP))
        already_description.extend(already_node["description"].split(GRAPH_FIELD_SEP))

    new_source_ids = [dp["source_id"] for dp in nodes_data if dp.get("source_id")]

    existing_full_source_ids = []
    if entity_chunks_storage is not None:
        stored_chunks = await entity_chunks_storage.get_by_id(entity_name)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id for chunk_id in stored_chunks.get("chunk_ids", []) if chunk_id
            ]

    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merging new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    if entity_chunks_storage is not None and full_source_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_entity")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # 4. Only keep nodes not filter by apply_source_ids_limit if limit_method is KEEP
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        filtered_nodes = []
        for dp in nodes_data:
            source_id = dp.get("source_id")
            # Skip descriptions sourced from chunks dropped by the limitation cap
            if (
                source_id
                and source_id not in allowed_source_ids
                and source_id not in existing_full_source_ids
            ):
                continue
            filtered_nodes.append(dp)
        nodes_data = filtered_nodes
    else:  # In FIFO mode, keep all nodes - truncation happens at source_ids level only
        nodes_data = list(nodes_data)

    # 5. Keep merge idempotent by accepting only first-time source evidence.
    # Also collapse same-source repeats inside one merge pass to the richest description.
    existing_source_id_set = {
        chunk_id for chunk_id in existing_full_source_ids if chunk_id
    }
    best_node_by_source: dict[str, dict] = {}
    source_less_nodes: list[dict] = []

    for node_data in nodes_data:
        source_id_value = str(node_data.get("source_id", "") or "").strip()
        if not source_id_value:
            source_less_nodes.append(node_data)
            continue
        if source_id_value in existing_source_id_set:
            continue

        previous = best_node_by_source.get(source_id_value)
        if previous is None:
            best_node_by_source[source_id_value] = node_data
            continue

        previous_desc_len = len(str(previous.get("description", "") or ""))
        current_desc_len = len(str(node_data.get("description", "") or ""))
        if current_desc_len > previous_desc_len:
            best_node_by_source[source_id_value] = node_data

    nodes_data = source_less_nodes + list(best_node_by_source.values())

    # Nothing new to merge: keep existing entity as-is.
    if not nodes_data:
        if already_node:
            if (
                limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
                and len(existing_full_source_ids) >= max_source_limit
            ):
                logger.info(
                    f"Skipped `{entity_name}`: KEEP old chunks {already_source_ids}/{len(full_source_ids)}"
                )
            else:
                logger.debug(
                    f"Skipped `{entity_name}`: no new source evidence (idempotent replay)"
                )
            unchanged_node = dict(already_node)
            if incoming_entity_name and not unchanged_node.get("entity_name"):
                unchanged_node["entity_name"] = incoming_entity_name
            unchanged_node["_changed"] = False
            return unchanged_node
        logger.error(f"Internal Error: already_node missing for `{entity_name}`")
        raise ValueError(f"Internal Error: already_node missing for `{entity_name}`")

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize entity type by highest count
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 7. Deduplicate nodes by description, keeping first occurrence in the same document
    unique_nodes = {}
    for dp in nodes_data:
        desc = dp.get("description")
        if not desc:
            continue
        if desc not in unique_nodes:
            unique_nodes[desc] = dp

    # Sort description by timestamp, then by description length when timestamps are the same
    sorted_nodes = sorted(
        unique_nodes.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    sorted_descriptions = [dp["description"] for dp in sorted_nodes]

    # Combine already_description with sorted new sorted descriptions
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Entity {entity_name} has no description")
        raise ValueError(f"Entity {entity_name} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during entity summary")

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        entity_name,
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Indicating file_path has been truncated before

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in nodes_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )
        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{entity_name}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10.Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(nodes_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    _disambig = global_config.get("enable_entity_disambiguation", True)
    # Recover plain human-readable name from the first node record.
    canonical_entity_name = entity_name
    if nodes_data:
        first_entity_name = nodes_data[0].get("entity_name")
        if isinstance(first_entity_name, str) and first_entity_name:
            canonical_entity_name = first_entity_name

    # entity_name is already the correct graph node key:
    #   - disambiguation ON:  entity_name = "name|type"  (composite, set by caller)
    #   - disambiguation OFF: entity_name = "name"
    # Do NOT call compute_entity_id again — that would produce "name|type|type".
    composite_id = entity_name
    node_data = dict(
        entity_id=composite_id,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
        truncate=truncation_info,
    )
    await knowledge_graph_inst.upsert_node(
        composite_id,
        node_data=node_data,
    )
    node_data["entity_name"] = canonical_entity_name
    if entity_vdb is not None:
        entity_vdb_id = compute_entity_vdb_id(
            canonical_entity_name, entity_type, _disambig
        )
        entity_content = f"{canonical_entity_name}\n{description}"
        data_for_vdb = {
            entity_vdb_id: {
                "entity_id": composite_id,
                "entity_name": canonical_entity_name,
                "entity_type": entity_type,
                "content": entity_content,
                "source_id": source_id,
                "file_path": file_path,
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
            operation_name="entity_upsert",
            entity_name=canonical_entity_name,
            max_retries=3,
            retry_delay=0.1,
        )
    node_data["_changed"] = True
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,  # New parameter to track entities added during edge processing
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_skip_stats: dict[str, Any] | None = None,
    allowed_relation_endpoint_ids: set[str] | None = None,
):
    if src_id == tgt_id:
        return None

    strict_endpoint_match = bool(
        global_config.get(
            "strict_relation_endpoint_entity_match",
            DEFAULT_STRICT_RELATION_ENDPOINT_ENTITY_MATCH,
        )
    )
    if strict_endpoint_match:
        missing_endpoints: list[str] = []
        endpoint_scope = (
            "doc-scope" if allowed_relation_endpoint_ids is not None else "graph"
        )
        if allowed_relation_endpoint_ids is not None:
            if src_id not in allowed_relation_endpoint_ids:
                missing_endpoints.append(src_id)
            if tgt_id not in allowed_relation_endpoint_ids:
                missing_endpoints.append(tgt_id)
        else:
            if not await knowledge_graph_inst.has_node(src_id):
                missing_endpoints.append(src_id)
            if not await knowledge_graph_inst.has_node(tgt_id):
                missing_endpoints.append(tgt_id)
        if missing_endpoints:
            if allowed_relation_endpoint_ids is None:
                await _remove_relation_edge_and_vector(
                    knowledge_graph_inst=knowledge_graph_inst,
                    relationships_vdb=relationships_vdb,
                    src_id=src_id,
                    tgt_id=tgt_id,
                )
            if relation_skip_stats is not None:
                relation_skip_stats["doc_scope_missing_endpoint_relations"] = (
                    int(
                        relation_skip_stats.get(
                            "doc_scope_missing_endpoint_relations", 0
                        )
                    )
                    + 1
                )
                missing_counter = relation_skip_stats.get("missing_endpoints")
                if not isinstance(missing_counter, Counter):
                    missing_counter = Counter()
                    relation_skip_stats["missing_endpoints"] = missing_counter
                missing_counter.update(sorted(set(missing_endpoints)))
            logger.info(
                "Skipped relation `%s`~`%s`: strict %s endpoint match enabled, missing endpoints=%s",
                src_id,
                tgt_id,
                endpoint_scope,
                ",".join(sorted(set(missing_endpoints))),
            )
            return None

    already_edge = None
    already_weight_raw = 0.0
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    # 1. Get existing edge data from graph storage
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            if _is_factual_or_legacy_edge(already_edge):
                # Only factual history participates in factual merge.
                # Existing synonym metadata must not leak into factual updates.
                already_weight_raw = _extract_existing_factual_weight_raw(already_edge)

                if already_edge.get("source_id") is not None:
                    already_source_ids.extend(
                        already_edge["source_id"].split(GRAPH_FIELD_SEP)
                    )

                if already_edge.get("file_path") is not None:
                    already_file_paths.extend(
                        already_edge["file_path"].split(GRAPH_FIELD_SEP)
                    )

                if already_edge.get("description") is not None:
                    already_description.extend(
                        already_edge["description"].split(GRAPH_FIELD_SEP)
                    )

                if already_edge.get("keywords") is not None:
                    already_keywords.extend(
                        split_string_by_multi_markers(
                            already_edge["keywords"], [GRAPH_FIELD_SEP]
                        )
                    )

    new_source_ids = [dp["source_id"] for dp in edges_data if dp.get("source_id")]

    storage_key = make_relation_chunk_key(src_id, tgt_id)
    existing_full_source_ids = []
    if relation_chunks_storage is not None:
        stored_chunks = await relation_chunks_storage.get_by_id(storage_key)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id for chunk_id in stored_chunks.get("chunk_ids", []) if chunk_id
            ]

    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merge new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    if relation_chunks_storage is not None and full_source_ids:
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_relation")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{src_id}`~`{tgt_id}`",
    )
    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    # 4. Only keep edges with source_id in the final source_ids list if in KEEP mode
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        filtered_edges = []
        for dp in edges_data:
            source_id = dp.get("source_id")
            # Skip relationship fragments sourced from chunks dropped by keep oldest cap
            if (
                source_id
                and source_id not in allowed_source_ids
                and source_id not in existing_full_source_ids
            ):
                continue
            filtered_edges.append(dp)
        edges_data = filtered_edges
    else:  # In FIFO mode, keep all edges - truncation happens at source_ids level only
        edges_data = list(edges_data)

    # 5. Keep merge idempotent by accepting only first-time source evidence.
    existing_source_id_set = {
        chunk_id for chunk_id in existing_full_source_ids if chunk_id
    }
    source_weight_increments: dict[str, float] = {}
    filtered_incremental_edges: list[dict] = []

    for edge_data in edges_data:
        source_id_value = str(edge_data.get("source_id", "") or "").strip()
        if source_id_value and source_id_value in existing_source_id_set:
            continue

        filtered_incremental_edges.append(edge_data)

        if source_id_value:
            edge_weight = edge_data.get("weight", 1.0)
            parsed_weight = _to_non_negative_float(edge_weight, default=1.0)
            source_weight_increments[source_id_value] = max(
                source_weight_increments.get(source_id_value, 0.0),
                parsed_weight,
            )

    edges_data = filtered_incremental_edges

    # Nothing new to merge: keep existing edge as-is.
    if not edges_data:
        if already_edge:
            if (
                limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
                and len(existing_full_source_ids) >= max_source_limit
            ):
                logger.info(
                    f"Skipped `{src_id}`~`{tgt_id}`: KEEP old chunks  {already_source_ids}/{len(full_source_ids)}"
                )
            else:
                logger.debug(
                    f"Skipped `{src_id}`~`{tgt_id}`: no new source evidence (idempotent replay)"
                )
            return dict(already_edge)
        logger.error(f"Internal Error: already_edge missing for `{src_id}`~`{tgt_id}`")
        raise ValueError(
            f"Internal Error: already_edge missing for `{src_id}`~`{tgt_id}`"
        )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize factual weight:
    # raw evidence is additive; retrieval weight uses log1p scaling.
    source_less_weight_sum = 0.0
    for edge_data in edges_data:
        source_id_value = str(edge_data.get("source_id", "") or "").strip()
        if source_id_value:
            continue
        edge_weight = edge_data.get("weight", 1.0)
        source_less_weight_sum += _to_non_negative_float(edge_weight, default=1.0)
    weight_raw = (
        already_weight_raw + sum(source_weight_increments.values()) + source_less_weight_sum
    )
    weight = _factual_weight_from_raw(weight_raw)

    # 6.3 Finalize keywords by merging existing and new keywords.
    # Optional case normalization keeps relation keyword style stable.
    enable_keyword_case_normalization = bool(
        global_config.get(
            "enable_keyword_case_normalization",
            DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION,
        )
    )
    uppercase_allowlist = _normalize_uppercase_allowlist(
        global_config.get(
            "entity_uppercase_allowlist",
            DEFAULT_ENTITY_UPPERCASE_ALLOWLIST,
        )
    )
    raw_keyword_items = [keyword for keyword in already_keywords if keyword]
    raw_keyword_items.extend(
        str(edge.get("keywords", "") or "")
        for edge in edges_data
        if edge.get("keywords")
    )
    keywords = _merge_relation_keywords(
        raw_keyword_items,
        uppercase_allowlist=uppercase_allowlist,
        enable_case_normalization=enable_keyword_case_normalization,
    )

    # 7. Deduplicate by description, keeping first occurrence in the same document
    unique_edges = {}
    for dp in edges_data:
        description_value = dp.get("description")
        if not description_value:
            continue
        if description_value not in unique_edges:
            unique_edges[description_value] = dp

    # Sort description by timestamp, then by description length (largest to smallest) when timestamps are the same
    sorted_edges = sorted(
        unique_edges.values(),
        key=lambda x: (x.get("timestamp", 0), -len(x.get("description", ""))),
    )
    sorted_descriptions = [dp["description"] for dp in sorted_edges]

    # Combine already_description with sorted new descriptions
    description_list = already_description + sorted_descriptions
    if not description_list:
        logger.error(f"Relation {src_id}~{tgt_id} has no description")
        raise ValueError(f"Relation {src_id}~{tgt_id} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during relation summary"
                )

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Relation",
        f"({src_id}, {tgt_id})",
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS limit
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Track if already_file_paths contains placeholder

    max_file_paths = global_config.get("max_file_paths", DEFAULT_MAX_FILE_PATHS)
    file_path_placeholder = global_config.get(
        "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    )

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        # Check if this is a placeholder record
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for dp in edges_data:
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths")

    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        file_path_placeholder = global_config.get(
            "file_path_more_placeholder", DEFAULT_FILE_PATH_MORE_PLACEHOLDER
        )

        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+" if has_placeholder else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{src_id}`~`{tgt_id}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10. Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(edges_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    _disambig = global_config.get("enable_entity_disambiguation", True)
    for need_insert_id in [src_id, tgt_id]:
        # Optimization: Use get_node instead of has_node + get_node
        existing_node = await knowledge_graph_inst.get_node(need_insert_id)

        if existing_node is None:
            # Node doesn't exist - create new node
            edge_entity_name = need_insert_id
            edge_entity_type = "UNKNOWN"
            if _disambig and "|" in need_insert_id:
                parsed_name, parsed_type = need_insert_id.rsplit("|", 1)
                if parsed_name and parsed_type:
                    edge_entity_name = parsed_name
                    edge_entity_type = parsed_type

            if _disambig and need_insert_id != edge_entity_name:
                edge_entity_id = need_insert_id
            else:
                edge_entity_id = compute_entity_id(
                    edge_entity_name, edge_entity_type, _disambig
                )
            node_created_at = int(time.time())
            node_data = {
                "entity_id": edge_entity_id,
                "source_id": source_id,
                "description": description,
                "entity_type": edge_entity_type,
                "file_path": file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(edge_entity_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None:
                chunk_ids = [chunk_id for chunk_id in full_source_ids if chunk_id]
                if chunk_ids:
                    await entity_chunks_storage.upsert(
                        {
                            edge_entity_id: {
                                "chunk_ids": chunk_ids,
                                "count": len(chunk_ids),
                            }
                        }
                    )

            if entity_vdb is not None:
                entity_vdb_id = compute_entity_vdb_id(
                    edge_entity_name, edge_entity_type, _disambig
                )
                entity_content = f"{edge_entity_name}\n{description}"
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_id": edge_entity_id,
                        "entity_name": edge_entity_name,
                        "source_id": source_id,
                        "entity_type": edge_entity_type,
                        "file_path": file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                    operation_name="added_entity_upsert",
                    entity_name=edge_entity_name,
                    max_retries=3,
                    retry_delay=0.1,
                )

            # Track entities added during edge processing
            if added_entities is not None:
                entity_data = {
                    "entity_id": edge_entity_id,
                    "entity_name": edge_entity_name,
                    "entity_type": edge_entity_type,
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": node_created_at,
                }
                added_entities.append(entity_data)
        else:
            # Node exists - update its source_ids by merging with new source_ids
            updated = False  # Track if any update occurred
            existing_entity_id = existing_node.get("entity_id", need_insert_id)

            # 1. Get existing full source_ids from entity_chunks_storage
            existing_full_source_ids = []
            if entity_chunks_storage is not None:
                stored_chunks = await entity_chunks_storage.get_by_id(existing_entity_id)
                if stored_chunks and isinstance(stored_chunks, dict):
                    existing_full_source_ids = [
                        chunk_id
                        for chunk_id in stored_chunks.get("chunk_ids", [])
                        if chunk_id
                    ]

            # If not in entity_chunks_storage, get from graph database
            if not existing_full_source_ids:
                if existing_node.get("source_id"):
                    existing_full_source_ids = existing_node["source_id"].split(
                        GRAPH_FIELD_SEP
                    )

            # 2. Merge with new source_ids from this relationship
            new_source_ids_from_relation = [
                chunk_id for chunk_id in source_ids if chunk_id
            ]
            merged_full_source_ids = merge_source_ids(
                existing_full_source_ids, new_source_ids_from_relation
            )

            # 3. Save merged full list to entity_chunks_storage (conditional)
            if (
                entity_chunks_storage is not None
                and merged_full_source_ids != existing_full_source_ids
            ):
                updated = True
                await entity_chunks_storage.upsert(
                    {
                        existing_entity_id: {
                            "chunk_ids": merged_full_source_ids,
                            "count": len(merged_full_source_ids),
                        }
                    }
                )

            # 4. Apply source_ids limit for graph and vector db
            limit_method = global_config.get(
                "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
            )
            max_source_limit = global_config.get("max_source_ids_per_entity")
            limited_source_ids = apply_source_ids_limit(
                merged_full_source_ids,
                max_source_limit,
                limit_method,
                identifier=f"`{existing_entity_id}`",
            )

            # 5. Update graph database and vector database with limited source_ids (conditional)
            limited_source_id_str = GRAPH_FIELD_SEP.join(limited_source_ids)

            if limited_source_id_str != existing_node.get("source_id", ""):
                updated = True
                updated_node_data = {
                    **existing_node,
                    "source_id": limited_source_id_str,
                }
                await knowledge_graph_inst.upsert_node(
                    existing_entity_id, node_data=updated_node_data
                )

                # Update vector database
                if entity_vdb is not None:
                    exist_entity_type = existing_node.get("entity_type", "UNKNOWN")
                    exist_entity_name = existing_node.get(
                        "entity_name", existing_entity_id
                    )
                    if (
                        _disambig
                        and exist_entity_name == existing_entity_id
                        and "|" in existing_entity_id
                    ):
                        parsed_name, _ = existing_entity_id.rsplit("|", 1)
                        if parsed_name:
                            exist_entity_name = parsed_name
                    entity_vdb_id = compute_entity_vdb_id(
                        exist_entity_name, exist_entity_type, _disambig
                    )
                    entity_content = (
                        f"{exist_entity_name}\n{existing_node.get('description', '')}"
                    )
                    vdb_data = {
                        entity_vdb_id: {
                            "content": entity_content,
                            "entity_id": existing_entity_id,
                            "entity_name": exist_entity_name,
                            "source_id": limited_source_id_str,
                            "entity_type": exist_entity_type,
                            "file_path": existing_node.get(
                                "file_path", "unknown_source"
                            ),
                        }
                    }
                    await safe_vdb_operation_with_exception(
                        operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                        operation_name="existing_entity_update",
                        entity_name=exist_entity_name,
                        max_retries=3,
                        retry_delay=0.1,
                    )

            # 6. Log once at the end if any update occurred
            if updated:
                status_message = (
                    f"Chunks appended to relation endpoint entity: `{existing_entity_id}` "
                    f"from relation `{src_id}`~`{tgt_id}`"
                )
                logger.info(status_message)
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = status_message
                        pipeline_status["history_messages"].append(status_message)

    edge_created_at = int(time.time())
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            weight_raw=weight_raw,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=edge_created_at,
            truncate=truncation_info,
            edge_type=FACTUAL_EDGE_TYPE,
            provenance=FACTUAL_EDGE_PROVENANCE,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=edge_created_at,
        truncate=truncation_info,
        weight=weight,
        weight_raw=weight_raw,
        edge_type=FACTUAL_EDGE_TYPE,
        provenance=FACTUAL_EDGE_PROVENANCE,
    )

    # Sort src_id and tgt_id to ensure consistent ordering (smaller string first)
    if src_id > tgt_id:
        src_id, tgt_id = tgt_id, src_id

    if relationships_vdb is not None:
        rel_vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt_id + src_id, prefix="rel-")
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )
        rel_content = f"{keywords}\t{src_id}\n{tgt_id}\n{description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": source_id,
                "content": rel_content,
                "keywords": keywords,
                "description": description,
                "weight": weight,
                "weight_raw": weight_raw,
                "file_path": file_path,
                "edge_type": FACTUAL_EDGE_TYPE,
                "provenance": FACTUAL_EDGE_PROVENANCE,
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=vdb_data: relationships_vdb.upsert(payload),
            operation_name="relationship_upsert",
            entity_name=f"{src_id}-{tgt_id}",
            max_retries=3,
            retry_delay=0.2,
        )

    return edge_data


async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,
    full_relations_storage: BaseKVStorage = None,
    doc_id: str = None,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
) -> dict[str, list[str]]:
    """Two-phase merge: process all entities first, then all relationships

    This approach ensures data consistency by:
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)
    3. Phase 3: Update full_entities and full_relations storage with final results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        full_entities_storage: Storage for document entity lists
        full_relations_storage: Storage for document relation lists
        doc_id: Document ID for storage indexing
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
        entity_chunks_storage: Storage tracking full chunk lists per entity
        relation_chunks_storage: Storage tracking full chunk lists per relation
        current_file_number: Current file number for logging
        total_files: Total files for logging
        file_path: File path for logging

    Returns:
        dict containing ``changed_entity_ids`` (newly added or updated entity IDs)
        for incremental synonym-linking query-side selection.
    """

    # Check for cancellation at the start of merge
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during merge phase")

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    _disambig = global_config.get("enable_entity_disambiguation", True)
    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        if _disambig:
            # V1: group by composite key (name|type) for disambiguation
            for entity_name, entities in maybe_nodes.items():
                for entity in entities:
                    group_key = compute_entity_id(
                        entity.get("entity_name", entity_name),
                        entity.get("entity_type", ""),
                        True,
                    )
                    all_nodes[group_key].append(entity)
        else:
            # Original path: batch extend by entity_name — identical to upstream
            for entity_name, entities in maybe_nodes.items():
                all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)

    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)
    total_relation_records_count = sum(len(edges) for edges in all_edges.values())

    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # ===== Phase 1: Process all entities concurrently =====
    log_message = f"Phase 1: Processing {total_entities_count} entities from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_entity_name(entity_name, entities):
        async with semaphore:
            # Check for cancellation before processing entity
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during entity merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    logger.debug(f"Processing entity {entity_name}")
                    entity_data = await _merge_nodes_then_upsert(
                        entity_name,
                        entities,
                        knowledge_graph_inst,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        entity_chunks_storage,
                    )

                    return entity_data

                except Exception as e:
                    error_msg = f"Error processing entity `{entity_name}`: {e}"
                    logger.error(error_msg)

                    # Try to update pipeline status, but don't let status update failure affect main exception
                    try:
                        if (
                            pipeline_status is not None
                            and pipeline_status_lock is not None
                        ):
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)
                    except Exception as status_error:
                        logger.error(
                            f"Failed to update pipeline status: {status_error}"
                        )

                    # Re-raise the original exception with a prefix
                    prefixed_exception = create_prefixed_exception(
                        e, f"`{entity_name}`"
                    )
                    raise prefixed_exception from e

    # Create entity processing tasks
    entity_tasks = []
    for entity_name, entities in all_nodes.items():
        task = asyncio.create_task(_locked_process_entity_name(entity_name, entities))
        entity_tasks.append(task)

    # Execute entity tasks with error handling
    processed_entities = []
    if entity_tasks:
        done, pending = await asyncio.wait(
            entity_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        first_exception = None
        processed_entities = []

        for task in done:
            try:
                result = task.result()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
            else:
                processed_entities.append(result)

        if pending:
            for task in pending:
                task.cancel()
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                else:
                    processed_entities.append(result)

        if first_exception is not None:
            raise first_exception

    # Build name→composite_id mapping for edge key resolution
    _entity_name_to_composite = {}
    doc_relation_endpoint_ids: set[str] = set()
    if _disambig:
        for ent in processed_entities:
            if ent and isinstance(ent, dict):
                ename = ent.get("entity_name", "")
                eid = ent.get("entity_id", ename)
                if ename and eid:
                    _entity_name_to_composite[ename] = eid
                if eid:
                    doc_relation_endpoint_ids.add(eid)

        # Remap edge keys to use composite IDs
        remapped_edges = defaultdict(list)
        for edge_key, edges in all_edges.items():
            new_src = _entity_name_to_composite.get(edge_key[0], edge_key[0])
            new_tgt = _entity_name_to_composite.get(edge_key[1], edge_key[1])
            new_key = tuple(sorted((new_src, new_tgt)))
            remapped_edges[new_key].extend(edges)
        all_edges = remapped_edges
        total_relations_count = len(all_edges)
        total_relation_records_count = sum(len(edges) for edges in all_edges.values())
    else:
        for ent in processed_entities:
            if ent and isinstance(ent, dict):
                eid = ent.get("entity_id") or ent.get("entity_name")
                if eid:
                    doc_relation_endpoint_ids.add(eid)

    # ===== Phase 2: Process all relationships concurrently =====
    log_message = f"Phase 2: Processing {total_relations_count} relations from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_edges(edge_key, edges):
        async with semaphore:
            # Check for cancellation before processing edges
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during relation merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            sorted_edge_key = sorted([edge_key[0], edge_key[1]])

            async with get_storage_keyed_lock(
                sorted_edge_key,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    added_entities = []  # Track entities added during edge processing

                    logger.debug(f"Processing relation {sorted_edge_key}")
                    edge_data = await _merge_edges_then_upsert(
                        edge_key[0],
                        edge_key[1],
                        edges,
                        knowledge_graph_inst,
                        relationships_vdb,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        added_entities,  # Pass list to collect added entities
                        relation_chunks_storage,
                        entity_chunks_storage,  # Add entity_chunks_storage parameter
                        relation_skip_stats,
                        doc_relation_endpoint_ids,
                    )

                    if edge_data is None:
                        return None, []

                    return edge_data, added_entities

                except Exception as e:
                    error_msg = f"Error processing relation `{sorted_edge_key}`: {e}"
                    logger.error(error_msg)

                    # Try to update pipeline status, but don't let status update failure affect main exception
                    try:
                        if (
                            pipeline_status is not None
                            and pipeline_status_lock is not None
                        ):
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(error_msg)
                    except Exception as status_error:
                        logger.error(
                            f"Failed to update pipeline status: {status_error}"
                        )

                    # Re-raise the original exception with a prefix
                    prefixed_exception = create_prefixed_exception(
                        e, f"{sorted_edge_key}"
                    )
                    raise prefixed_exception from e

    relation_skip_stats: dict[str, Any] = {
        "doc_scope_missing_endpoint_relations": 0,
        "missing_endpoints": Counter(),
    }

    # Create relationship processing tasks
    edge_tasks = []
    for edge_key, edges in all_edges.items():
        task = asyncio.create_task(_locked_process_edges(edge_key, edges))
        edge_tasks.append(task)

    # Execute relationship tasks with error handling
    processed_edges = []
    all_added_entities = []

    if edge_tasks:
        done, pending = await asyncio.wait(
            edge_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        first_exception = None

        for task in done:
            try:
                edge_data, added_entities = task.result()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
            else:
                if edge_data is not None:
                    processed_edges.append(edge_data)
                all_added_entities.extend(added_entities)

        if pending:
            for task in pending:
                task.cancel()
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                else:
                    edge_data, added_entities = result
                    if edge_data is not None:
                        processed_edges.append(edge_data)
                    all_added_entities.extend(added_entities)

        if first_exception is not None:
            raise first_exception

    doc_scope_skipped_relations = int(
        relation_skip_stats.get("doc_scope_missing_endpoint_relations", 0)
    )
    missing_endpoint_counter = relation_skip_stats.get("missing_endpoints")
    if not isinstance(missing_endpoint_counter, Counter):
        missing_endpoint_counter = Counter()
    skip_rate = (
        doc_scope_skipped_relations / total_relations_count
        if total_relations_count
        else 0.0
    )
    top_missing_endpoints = [
        {"endpoint": endpoint, "count": count}
        for endpoint, count in missing_endpoint_counter.most_common(10)
    ]
    strict_endpoint_match_enabled = bool(
        global_config.get("strict_relation_endpoint_entity_match", False)
    )
    log_message = (
        "Relation endpoint doc-scope strict-match summary: "
        f"doc_id={doc_id}, "
        f"strict_relation_endpoint_entity_match={strict_endpoint_match_enabled}, "
        f"doc_entity_ids={len(doc_relation_endpoint_ids)}, "
        f"extracted_relation_pairs={total_relations_count}, "
        f"extracted_relation_records={total_relation_records_count}, "
        f"skipped_relation_pairs={doc_scope_skipped_relations}, "
        f"skip_rate={skip_rate:.2%}, "
        f"missing_endpoint_top={json.dumps(top_missing_endpoints, ensure_ascii=False)}"
    )
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    changed_entity_ids: set[str] = set()
    for entity_data in processed_entities:
        if not entity_data:
            continue
        if entity_data.get("_changed", True):
            entity_id = entity_data.get("entity_id") or entity_data.get("entity_name")
            if entity_id:
                changed_entity_ids.add(entity_id)

    for added_entity in all_added_entities:
        if not added_entity:
            continue
        entity_id = added_entity.get("entity_id") or added_entity.get("entity_name")
        if entity_id:
            changed_entity_ids.add(entity_id)

    # ===== Phase 3: Update full_entities and full_relations storage =====
    if full_entities_storage and full_relations_storage and doc_id:
        try:
            # Collect entities produced in this merge call.
            final_entity_names = set()
            for entity_data in processed_entities:
                if not entity_data:
                    continue
                entity_id = entity_data.get("entity_id") or entity_data.get(
                    "entity_name"
                )
                if entity_id:
                    final_entity_names.add(entity_id)
            for added_entity in all_added_entities:
                if not added_entity:
                    continue
                entity_id = added_entity.get("entity_id") or added_entity.get(
                    "entity_name"
                )
                if entity_id:
                    final_entity_names.add(entity_id)

            # Collect relation pairs produced in this merge call.
            final_relation_pairs = set()
            for edge_data in processed_edges:
                if not edge_data:
                    continue
                src_id = edge_data.get("src_id")
                tgt_id = edge_data.get("tgt_id")
                if src_id and tgt_id:
                    final_relation_pairs.add(tuple(sorted([src_id, tgt_id])))

            # Merge with existing doc-level indexes instead of overwrite.
            existing_entities_data = await full_entities_storage.get_by_id(doc_id)
            existing_relations_data = await full_relations_storage.get_by_id(doc_id)

            merged_entity_names: list[str] = []
            seen_entity_names: set[str] = set()
            if isinstance(existing_entities_data, dict):
                for existing_entity in existing_entities_data.get("entity_names", []):
                    entity_key = str(existing_entity).strip()
                    if entity_key and entity_key not in seen_entity_names:
                        seen_entity_names.add(entity_key)
                        merged_entity_names.append(entity_key)
            for entity_id in sorted(final_entity_names):
                if entity_id not in seen_entity_names:
                    seen_entity_names.add(entity_id)
                    merged_entity_names.append(entity_id)

            merged_relation_pairs: list[list[str]] = []
            seen_relation_pairs: set[tuple[str, str]] = set()

            def _append_relation_pair(raw_pair: Any) -> None:
                if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) < 2:
                    return
                src_value = str(raw_pair[0]).strip()
                tgt_value = str(raw_pair[1]).strip()
                if not src_value or not tgt_value:
                    return
                normalized_pair = tuple(sorted((src_value, tgt_value)))
                if normalized_pair in seen_relation_pairs:
                    return
                seen_relation_pairs.add(normalized_pair)
                merged_relation_pairs.append([normalized_pair[0], normalized_pair[1]])

            if isinstance(existing_relations_data, dict):
                for existing_pair in existing_relations_data.get("relation_pairs", []):
                    _append_relation_pair(existing_pair)
            for relation_pair in sorted(final_relation_pairs):
                _append_relation_pair(relation_pair)

            log_message = (
                f"Phase 3: Updating final "
                f"{len(merged_entity_names)} entities "
                f"({len(processed_entities)}+{len(all_added_entities)} new candidates) "
                f"and {len(merged_relation_pairs)} relations from {doc_id}"
            )
            logger.info(log_message)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            if merged_entity_names:
                entity_payload = {
                    "entity_names": merged_entity_names,
                    "count": len(merged_entity_names),
                }
                if isinstance(existing_entities_data, dict):
                    for key, value in existing_entities_data.items():
                        if key not in {"entity_names", "count"}:
                            entity_payload[key] = value
                await full_entities_storage.upsert({doc_id: entity_payload})

            if merged_relation_pairs:
                relation_payload = {
                    "relation_pairs": merged_relation_pairs,
                    "count": len(merged_relation_pairs),
                }
                if isinstance(existing_relations_data, dict):
                    for key, value in existing_relations_data.items():
                        if key not in {"relation_pairs", "count"}:
                            relation_payload[key] = value
                await full_relations_storage.upsert({doc_id: relation_payload})

            logger.debug(
                f"Updated entity-relation index for document {doc_id}: "
                f"{len(merged_entity_names)} entities, {len(merged_relation_pairs)} relations"
            )

        except Exception as e:
            logger.error(
                f"Failed to update entity-relation index for document {doc_id}: {e}"
            )
            # Don't raise exception to avoid affecting main flow

    log_message = f"Completed merging: {len(processed_entities)} entities, {len(all_added_entities)} extra entities, {len(processed_edges)} relations"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    return {"changed_entity_ids": sorted(changed_entity_ids)}


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
) -> list:
    # Check for cancellation at the start of entity extraction
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during entity extraction"
                )

    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)
    entity_types = global_config["addon_params"].get(
        "entity_types", DEFAULT_ENTITY_TYPES
    )
    enable_surface_normalization = bool(
        global_config.get(
            "enable_entity_surface_normalization",
            DEFAULT_ENABLE_ENTITY_SURFACE_NORMALIZATION,
        )
    )
    entity_uppercase_allowlist = global_config.get(
        "entity_uppercase_allowlist",
        DEFAULT_ENTITY_UPPERCASE_ALLOWLIST,
    )
    base_extraction_examples: list[str] = list(PROMPTS["entity_extraction_examples"])
    example_limit_raw = global_config.get("entity_extraction_example_limit", 10)
    try:
        example_limit = int(example_limit_raw)
    except (TypeError, ValueError):
        example_limit = 10
    if example_limit > 0:
        base_extraction_examples = base_extraction_examples[:example_limit]

    extraction_examples: list[str] = base_extraction_examples
    if enable_surface_normalization:
        normalization_examples = PROMPTS.get("entity_extraction_normalization_examples")
        if normalization_examples:
            extraction_examples.extend(normalization_examples)
        else:
            fallback_examples = PROMPTS.get(
                "entity_extraction_normalization_examples_fallback"
            )
            if fallback_examples:
                logger.warning(
                    "Using fallback normalization examples because "
                    "'entity_extraction_normalization_examples' is missing or empty."
                )
                extraction_examples.extend(fallback_examples)
            elif normalization_examples is None and fallback_examples is None:
                logger.warning(
                    "Surface normalization is enabled but normalization-example "
                    "prompt keys are missing; falling back to base extraction "
                    "examples only."
                )

    examples = "\n".join(extraction_examples)

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
        entity_name_case_rule=(
            PROMPTS["ENTITY_NAME_CASE_RULE_NORMALIZED"]
            if enable_surface_normalization
            else PROMPTS["ENTITY_NAME_CASE_RULE_DEFAULT"]
        ),
        relation_endpoint_case_rule=(
            PROMPTS["RELATION_ENDPOINT_CASE_RULE_NORMALIZED"]
            if enable_surface_normalization
            else PROMPTS["RELATION_ENDPOINT_CASE_RULE_DEFAULT"]
        ),
    )

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")

        # Create cache keys collector for batch processing
        cache_keys_collector = []

        # Get initial extraction
        # Format system prompt without input_text for each chunk (enables OpenAI prompt caching across chunks)
        entity_extraction_system_prompt = PROMPTS[
            "entity_extraction_system_prompt"
        ].format(**context_base)
        # Format user prompts with input_text for each chunk
        entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            **{**context_base, "input_text": content}
        )
        entity_continue_extraction_user_prompt = PROMPTS[
            "entity_continue_extraction_user_prompt"
        ].format(**{**context_base, "input_text": content})

        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )

        history = pack_user_ass_to_openai_messages(
            entity_extraction_user_prompt, final_result
        )

        # Process initial extraction with file path
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result,
            chunk_key,
            timestamp,
            file_path,
            source_text=content,
            tuple_delimiter=context_base["tuple_delimiter"],
            completion_delimiter=context_base["completion_delimiter"],
            enable_entity_surface_normalization=enable_surface_normalization,
            entity_uppercase_allowlist=entity_uppercase_allowlist,
        )

        # Process additional gleaning results only 1 time when entity_extract_max_gleaning is greater than zero.
        if entity_extract_max_gleaning > 0:
            glean_result, timestamp = await use_llm_func_with_cache(
                entity_continue_extraction_user_prompt,
                use_llm_func,
                system_prompt=entity_extraction_system_prompt,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                chunk_id=chunk_key,
                cache_keys_collector=cache_keys_collector,
            )

            # Process gleaning result separately with file path
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result,
                chunk_key,
                timestamp,
                file_path,
                source_text=content,
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
                enable_entity_surface_normalization=enable_surface_normalization,
                entity_uppercase_allowlist=entity_uppercase_allowlist,
            )

            # Merge results - compare description lengths to choose better version
            for entity_name, glean_entities in glean_nodes.items():
                if entity_name in maybe_nodes:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_nodes[entity_name][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_entities[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_nodes[entity_name] = list(glean_entities)
                    # Otherwise keep original version
                else:
                    # New entity from gleaning stage
                    maybe_nodes[entity_name] = list(glean_entities)

            for edge_key, glean_edges in glean_edges.items():
                if edge_key in maybe_edges:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_edges[edge_key][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_edges[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_edges[edge_key] = list(glean_edges)
                    # Otherwise keep original version
                else:
                    # New edge from gleaning stage
                    maybe_edges[edge_key] = list(glean_edges)

        # Batch update chunk's llm_cache_list with all collected cache keys
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel {chunk_key}"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            # Check for cancellation before processing chunk
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during chunk processing"
                        )

            try:
                return await _process_single_content(chunk)
            except Exception as e:
                chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
                prefixed_exception = create_prefixed_exception(e, chunk_id)
                raise prefixed_exception from e

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for tasks to complete or for the first exception to occur
    # This allows us to cancel remaining tasks if any task fails
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None
    chunk_results = []

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                chunk_results.append(task.result())
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Add progress prefix to the exception message
        progress_prefix = f"C[{processed_chunks + 1}/{total_chunks}]"

        # Re-raise the original exception with a prefix
        prefixed_exception = create_prefixed_exception(first_exception, progress_prefix)
        raise prefixed_exception from first_exception

    # If all tasks completed successfully, chunk_results already contains the results
    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: BaseVectorStorage = None,
) -> QueryResult | None:
    """
    Execute knowledge graph query and return unified QueryResult object.

    Args:
        query: Query string
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_db: Text chunks storage
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt
        chunks_vdb: Document chunks vector database

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Based on different query_param settings, different fields will be populated:
        - only_need_context=True: content contains context string
        - only_need_prompt=True: content contains complete prompt
        - stream=True: response_iterator contains streaming response, raw_data contains complete data
        - default: content contains LLM response text, raw_data contains complete data

        Returns None when no relevant context could be constructed for the query.
    """
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix", "rrf", "ppr", "ppr_local"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix", "rrf", "ppr", "ppr_local"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            logger.warning(f"Forced low_level_keywords to origin query: {query}")
            ll_keywords = [query]
        else:
            return QueryResult(content=PROMPTS["fail_response"])

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build query context (unified interface)
    context_result = await _build_query_context(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    if context_result is None:
        logger.info("[kg_query] No query context could be built; returning no-result.")
        return None

    # Return different content based on query parameters
    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(
            content=context_result.context, raw_data=context_result.raw_data
        )

    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Build system prompt
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=response_type,
        user_prompt=user_prompt,
        context_data=context_result.context,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=context_result.raw_data)

    # Call LLM
    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

    # Handle cache
    effective_history_messages = _build_effective_history_messages(query_param)
    history_signature = _history_messages_signature(effective_history_messages)

    query_cache_params = _build_query_cache_params(
        query_param,
        history_signature=history_signature,
        user_prompt=query_param.user_prompt or "",
        system_prompt=sys_prompt,
        hl_keywords_str=hl_keywords_str,
        ll_keywords_str=ll_keywords_str,
    )
    args_hash = _compute_query_cache_args_hash(query, query_cache_params)

    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )

    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=effective_history_messages,
            enable_cot=True,
            stream=query_param.stream,
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = dict(query_cache_params)
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return QueryResult(content=response, raw_data=context_result.raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response,
            raw_data=context_result.raw_data,
            is_streaming=True,
        )


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    return hl_keywords, ll_keywords


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Build the examples
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)
    enable_keyword_case_normalization = bool(
        global_config.get(
            "enable_keyword_case_normalization",
            DEFAULT_ENABLE_KEYWORD_CASE_NORMALIZATION,
        )
    )
    uppercase_allowlist = _normalize_uppercase_allowlist(
        global_config.get(
            "entity_uppercase_allowlist",
            DEFAULT_ENTITY_UPPERCASE_ALLOWLIST,
        )
    )

    # 2. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(
        param.mode,
        text,
        language,
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        try:
            keywords_data = json_repair.loads(cached_response)
            cached_hl_keywords = keywords_data.get("high_level_keywords", [])
            cached_ll_keywords = keywords_data.get("low_level_keywords", [])
            if enable_keyword_case_normalization:
                cached_hl_keywords = _normalize_keyword_list(
                    cached_hl_keywords,
                    keyword_kind="high_level",
                    uppercase_allowlist=uppercase_allowlist,
                )
                cached_ll_keywords = _normalize_keyword_list(
                    cached_ll_keywords,
                    keyword_kind="low_level",
                    uppercase_allowlist=uppercase_allowlist,
                )
            return cached_hl_keywords, cached_ll_keywords
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 3. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text,
        examples=examples,
        language=language,
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(kw_prompt))
    logger.debug(
        f"[extract_keywords] Sending to LLM: {len_of_prompts:,} tokens (Prompt: {len_of_prompts})"
    )

    # 4. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 5. Parse out JSON from the LLM response
    result = remove_think_tags(result)
    try:
        keywords_data = json_repair.loads(result)
        if not keywords_data:
            logger.error("No JSON-like structure found in the LLM respond.")
            return [], []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"LLM respond: {result}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])
    if enable_keyword_case_normalization:
        hl_keywords = _normalize_keyword_list(
            hl_keywords,
            keyword_kind="high_level",
            uppercase_allowlist=uppercase_allowlist,
        )
        ll_keywords = _normalize_keyword_list(
            ll_keywords,
            keyword_kind="low_level",
            uppercase_allowlist=uppercase_allowlist,
        )

    # 6. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv.global_config.get("enable_llm_cache"):
            # Save to cache with query parameters
            queryparam_dict = {
                "mode": param.mode,
                "response_type": param.response_type,
                "top_k": param.top_k,
                "chunk_top_k": param.chunk_top_k,
                "max_entity_tokens": param.max_entity_tokens,
                "max_relation_tokens": param.max_relation_tokens,
                "max_total_tokens": param.max_total_tokens,
                "user_prompt": param.user_prompt or "",
                "enable_rerank": param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    mode=param.mode,
                    cache_type="keywords",
                    queryparam=queryparam_dict,
                ),
            )

    return hl_keywords, ll_keywords


async def _query_vector_storage(
    vector_storage: BaseVectorStorage,
    query: str,
    top_k: int,
    query_param: QueryParam,
    query_embedding: list[float] = None,
) -> list[dict[str, Any]]:
    if vector_storage.__class__.__name__ == "QdrantVectorDBStorage":
        return await vector_storage.query(
            query,
            top_k=top_k,
            query_embedding=query_embedding,
            qdrant_retrieval_mode=query_param.qdrant_retrieval_mode,
        )
    return await vector_storage.query(
        query,
        top_k=top_k,
        query_embedding=query_embedding,
    )


async def _get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    This function performs vector search to find relevant text chunks for a query.
    Reranking and truncation will be handled later in the unified processing.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including chunk_top_k and ids
        query_embedding: Optional pre-computed query embedding to avoid redundant embedding calls

    Returns:
        List of text chunks with metadata
    """
    try:
        # naive_top_k controls VDB retrieval size; chunk_top_k controls post-rerank window.
        # Fall back chain: naive_top_k → chunk_top_k → top_k
        search_top_k = (
            getattr(query_param, "naive_top_k", None)
            or query_param.chunk_top_k
            or query_param.top_k
        )
        cosine_threshold = chunks_vdb.cosine_better_than_threshold

        results = await _query_vector_storage(
            chunks_vdb,
            query,
            search_top_k,
            query_param,
            query_embedding,
        )
        if not results:
            logger.info(
                f"Naive query: 0 chunks (naive_top_k:{search_top_k} cosine:{cosine_threshold})"
            )
            return []

        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",  # Mark the source type
                    "chunk_id": result.get("id"),  # Add chunk_id for deduplication
                    "is_multimodal": result.get("is_multimodal", False),
                    "page_idx": result.get("page_idx"),
                }
                valid_chunks.append(chunk_with_metadata)

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks (naive_top_k:{search_top_k} cosine:{cosine_threshold})"
        )
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return []


async def _perform_kg_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
) -> dict[str, Any]:
    """
    Pure search logic that retrieves raw entities, relations, and vector chunks.
    No token truncation or formatting - just raw search results.
    """

    # Initialize result containers
    local_entities = []
    local_relations = []
    global_entities = []
    global_relations = []
    vector_chunks = []
    chunk_tracking = {}

    # Handle different query modes

    # Track chunk sources and metadata for final logging
    chunk_tracking = {}  # chunk_id -> {source, frequency, order}

    # Pre-compute query embedding once for all vector operations
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    query_embedding = None
    if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
        actual_embedding_func = text_chunks_db.embedding_func
        if actual_embedding_func:
            try:
                query_embedding = await actual_embedding_func([query])
                query_embedding = query_embedding[
                    0
                ]  # Extract first embedding from batch result
                logger.debug("Pre-computed query embedding for all vector operations")
            except Exception as e:
                logger.warning(f"Failed to pre-compute query embedding: {e}")
                query_embedding = None

    # Handle local and global modes
    _ppr_mode = query_param.mode in ("ppr", "ppr_local")

    if query_param.mode == "local" and len(ll_keywords) > 0:
        local_entities, local_relations = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )

    elif query_param.mode == "global" and len(hl_keywords) > 0:
        global_relations, global_entities = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

    else:  # hybrid / mix / rrf / ppr / ppr_local — or any unrecognised mode
        if len(ll_keywords) > 0:
            local_entities, local_relations = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                query_param,
            )
        if len(hl_keywords) > 0:
            global_relations, global_entities = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                query_param,
            )

        # Get vector chunks for mix/rrf mode (PPR modes handle chunks internally)
        if query_param.mode in ("mix", "rrf") and chunks_vdb:
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
                query_embedding,
            )
            # Track vector chunks with source metadata
            for i, chunk in enumerate(vector_chunks):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id:
                    chunk_tracking[chunk_id] = {
                        "source": "C",
                        "frequency": 1,  # Vector chunks always have frequency 1
                        "order": i + 1,  # 1-based order in vector search results
                    }
                else:
                    logger.warning(f"Vector chunk missing chunk_id: {chunk}")

    # Round-robin merge entities
    final_entities = []
    seen_entities = set()
    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        # First from local
        if i < len(local_entities):
            entity = local_entities[i]
            entity_key = entity.get("entity_id") or entity.get("entity_name")
            if entity_key and entity_key not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_key)

        # Then from global
        if i < len(global_entities):
            entity = global_entities[i]
            entity_key = entity.get("entity_id") or entity.get("entity_name")
            if entity_key and entity_key not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_key)

    # Round-robin merge relations
    final_relations = []
    seen_relations = set()
    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        # First from local
        if i < len(local_relations):
            relation = local_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

        # Then from global
        if i < len(global_relations):
            relation = global_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    logger.info(
        f"Raw search results: {len(final_entities)} entities, {len(final_relations)} relations, {len(vector_chunks)} vector chunks"
    )

    # V3: PPR chunk ranking
    # Triggered by mode="ppr" / mode="ppr_local" or the legacy enable_multi_hop flag.
    ppr_chunks = []
    _run_ppr = _ppr_mode or query_param.enable_multi_hop
    _use_global_ppr = query_param.mode == "ppr"

    if _run_ppr and chunks_vdb:
        all_entities = final_entities
        if all_entities:
            ppr_chunks = await _ppr_rank_chunks(
                query=query,
                node_datas=all_entities,
                knowledge_graph_inst=knowledge_graph_inst,
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                chunks_vdb=chunks_vdb,
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                query_embedding=query_embedding,
                use_global=_use_global_ppr,
            )

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "ppr_chunks": ppr_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


async def _apply_token_truncation(
    search_result: dict[str, Any],
    query_param: QueryParam,
    global_config: dict[str, str],
) -> dict[str, Any]:
    """
    Apply token-based truncation to entities and relations for LLM efficiency.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.warning("No tokenizer found, skipping truncation")
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": search_result["final_entities"],
            "filtered_relations": search_result["final_relations"],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }

    # Get token limits from query_param with fallbacks
    max_entity_tokens = getattr(
        query_param,
        "max_entity_tokens",
        global_config.get("max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS),
    )
    max_relation_tokens = getattr(
        query_param,
        "max_relation_tokens",
        global_config.get("max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS),
    )

    # HippoRAG2 alignment: PPR modes surface only chunk text to the LLM.
    # Entity/relation context adds token overhead without improving PPR recall.
    if getattr(query_param, "mode", None) in ("ppr", "ppr_local"):
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": [],
            "filtered_relations": [],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }

    final_entities = search_result["final_entities"]
    final_relations = search_result["final_relations"]

    # Create mappings from entity/relation identifiers to original data
    entity_id_to_original = {}
    relation_id_to_original = {}

    # Generate entities context for truncation
    entities_context = []
    for i, entity in enumerate(final_entities):
        entity_name = entity["entity_name"]
        created_at = entity.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Store mapping from entity name to original data
        entity_id_to_original[entity_name] = entity

        entities_context.append(
            {
                "entity": entity_name,
                "type": entity.get("entity_type", "UNKNOWN"),
                "description": entity.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": entity.get("file_path", "unknown_source"),
            }
        )

    # Generate relations context for truncation
    relations_context = []
    for i, relation in enumerate(final_relations):
        created_at = relation.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Handle different relation data formats
        if "src_tgt" in relation:
            entity1, entity2 = relation["src_tgt"]
        else:
            entity1, entity2 = relation.get("src_id"), relation.get("tgt_id")

        # Store mapping from relation pair to original data
        relation_key = (entity1, entity2)
        relation_id_to_original[relation_key] = relation

        relations_context.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "description": relation.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": relation.get("file_path", "unknown_source"),
            }
        )

    logger.debug(
        f"Before truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Apply token-based truncation
    if entities_context:
        # Remove file_path and created_at for token calculation
        entities_context_for_truncation = []
        for entity in entities_context:
            entity_copy = entity.copy()
            entity_copy.pop("file_path", None)
            entity_copy.pop("created_at", None)
            entities_context_for_truncation.append(entity_copy)

        entities_context = truncate_list_by_token_size(
            entities_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

    if relations_context:
        # Remove file_path and created_at for token calculation
        relations_context_for_truncation = []
        for relation in relations_context:
            relation_copy = relation.copy()
            relation_copy.pop("file_path", None)
            relation_copy.pop("created_at", None)
            relations_context_for_truncation.append(relation_copy)

        relations_context = truncate_list_by_token_size(
            relations_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

    logger.info(
        f"After truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Create filtered original data based on truncated context
    filtered_entities = []
    filtered_entity_id_to_original = {}
    if entities_context:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for entity in final_entities:
            name = entity.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                filtered_entities.append(entity)
                filtered_entity_id_to_original[name] = entity
                seen_nodes.add(name)

    filtered_relations = []
    filtered_relation_id_to_original = {}
    if relations_context:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        for relation in final_relations:
            src, tgt = relation.get("src_id"), relation.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = relation.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                filtered_relations.append(relation)
                filtered_relation_id_to_original[pair] = relation
                seen_edges.add(pair)

    return {
        "entities_context": entities_context,
        "relations_context": relations_context,
        "filtered_entities": filtered_entities,
        "filtered_relations": filtered_relations,
        "entity_id_to_original": filtered_entity_id_to_original,
        "relation_id_to_original": filtered_relation_id_to_original,
    }


def _rrf_merge(ranking_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked chunk lists.

    Each chunk's RRF score = sum of 1 / (k + rank_i) across all lists it appears in.
    Chunks are deduplicated by chunk_id; the first occurrence's metadata is kept.
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for ranked in ranking_lists:
        for rank, chunk in enumerate(ranked):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if chunk_id not in meta:
                meta[chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    result = []
    for cid in sorted_ids:
        chunk = meta[cid]
        result.append(
            {
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": cid,
                "is_multimodal": chunk.get("is_multimodal", False),
                "page_idx": chunk.get("page_idx"),
                "rrf_score": scores[cid],
            }
        )
    return result


async def _merge_all_chunks(
    filtered_entities: list[dict],
    filtered_relations: list[dict],
    vector_chunks: list[dict],
    query: str = "",
    knowledge_graph_inst: BaseGraphStorage = None,
    text_chunks_db: BaseKVStorage = None,
    query_param: QueryParam = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding: list[float] = None,
    ppr_chunks: list[dict] = None,
) -> list[dict]:
    """
    Merge chunks from different sources: vector_chunks + entity_chunks + relation_chunks.
    When PPR chunks are available (V3 enable_multi_hop), they take priority.
    """
    if chunk_tracking is None:
        chunk_tracking = {}

    # V3: When PPR chunks are available, they replace entity/relation chunk selection.
    # PPR scores already encode the graph structure signal; vector_chunks supplement.
    if ppr_chunks:
        # HippoRAG2 qa_top_k: slice PPR candidates to the LLM-context budget.
        # ppr_top_k controls retrieval breadth; ppr_qa_top_k controls LLM input size.
        ppr_qa_top_k = getattr(query_param, "ppr_qa_top_k", None) if query_param else None
        if ppr_qa_top_k and ppr_qa_top_k > 0:
            ppr_chunks = ppr_chunks[:ppr_qa_top_k]

        merged_chunks = []
        seen_chunk_ids = set()

        # PPR chunks first (highest priority — graph-based ranking)
        for chunk in ppr_chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "is_multimodal": chunk.get("is_multimodal", False),
                        "page_idx": chunk.get("page_idx"),
                        "ppr_score": chunk.get("ppr_score"),
                    }
                )

        # Vector chunks as supplement
        for chunk in vector_chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "is_multimodal": chunk.get("is_multimodal", False),
                        "page_idx": chunk.get("page_idx"),
                    }
                )

        logger.info(
            f"PPR-priority merged chunks: {len(ppr_chunks)} PPR (qa_top_k={ppr_qa_top_k}) + {len(vector_chunks)} vector -> {len(merged_chunks)} (deduplicated)"
        )
        return merged_chunks

    # Get chunks from entities (shared by both RRF and round-robin paths)
    entity_chunks = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await _find_related_text_unit_from_entities(
            filtered_entities,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Get chunks from relations (shared by both RRF and round-robin paths)
    relation_chunks = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await _find_related_text_unit_from_relations(
            filtered_relations,
            query_param,
            text_chunks_db,
            entity_chunks,  # For deduplication
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # RRF path: Reciprocal Rank Fusion merge
    if query_param is not None and query_param.mode == "rrf":
        ranking_lists = [lst for lst in [vector_chunks, entity_chunks, relation_chunks] if lst]
        if not ranking_lists:
            return []
        rrf_k = getattr(query_param, "rrf_k", 60)
        merged_chunks = _rrf_merge(ranking_lists, k=rrf_k)
        origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)
        logger.info(
            f"RRF merged chunks: {origin_len} -> {len(merged_chunks)} (k={rrf_k}, "
            f"sources: vector={len(vector_chunks)}, entity={len(entity_chunks)}, relation={len(relation_chunks)})"
        )
        return merged_chunks

    # Default path: round-robin merge from entity/relation/vector sources
    merged_chunks = []
    seen_chunk_ids = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)

    for i in range(max_len):
        # Add from vector chunks first (Naive mode)
        if i < len(vector_chunks):
            chunk = vector_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "is_multimodal": chunk.get("is_multimodal", False),
                        "page_idx": chunk.get("page_idx"),
                    }
                )

        # Add from entity chunks (Local mode)
        if i < len(entity_chunks):
            chunk = entity_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "is_multimodal": chunk.get("is_multimodal", False),
                        "page_idx": chunk.get("page_idx"),
                    }
                )

        # Add from relation chunks (Global mode)
        if i < len(relation_chunks):
            chunk = relation_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "is_multimodal": chunk.get("is_multimodal", False),
                        "page_idx": chunk.get("page_idx"),
                    }
                )

    logger.info(
        f"Round-robin merged chunks: {origin_len} -> {len(merged_chunks)} (deduplicated {origin_len - len(merged_chunks)})"
    )

    return merged_chunks


async def _build_context_str(
    entities_context: list[dict],
    relations_context: list[dict],
    merged_chunks: list[dict],
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    chunk_tracking: dict = None,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the final LLM context string with token processing.
    This includes dynamic token calculation and final chunk truncation.
    """
    def _empty_rerank_chunk_debug() -> dict[str, Any]:
        raw_scope = str(getattr(query_param, "rerank_score_scope", "all") or "all")
        scope = raw_scope.strip().lower()
        if scope not in {"top_k", "all"}:
            scope = "all"
        try:
            min_rerank_score = float(global_config.get("min_rerank_score", 0.5))
        except (TypeError, ValueError):
            min_rerank_score = 0.5
        return {
            "enabled": bool(query_param.enable_rerank and query),
            "scope": scope,
            "min_rerank_score": min_rerank_score,
            "scores_all": [],
            "scores_after_threshold": [],
            "scores_final": [],
            "chunk_ids_all": [],
            "chunk_ids_after_threshold": [],
            "chunk_ids_after_chunk_top_k": [],
            "chunk_ids_final": [],
            "count_input": 0,
            "count_after_rerank": 0,
            "count_after_threshold": 0,
            "count_after_chunk_top_k": 0,
            "count_final": 0,
        }

    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.error("Missing tokenizer, cannot build LLM context")
        # Return empty raw data structure when no tokenizer
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Missing tokenizer, cannot build LLM context."
        metadata = empty_raw_data.setdefault("metadata", {})
        metadata["rerank_chunk_debug"] = _empty_rerank_chunk_debug()
        return "", empty_raw_data

    # Get token limits
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Get the system prompt template from PROMPTS or global_config
    sys_prompt_template = global_config.get(
        "system_prompt_template", PROMPTS["rag_response"]
    )

    kg_context_template = PROMPTS["kg_query_context"]
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    entities_str = "\n".join(
        json.dumps(entity, ensure_ascii=False) for entity in entities_context
    )
    relations_str = "\n".join(
        json.dumps(relation, ensure_ascii=False) for relation in relations_context
    )

    # Calculate preliminary kg context tokens
    pre_kg_context = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str="",
        reference_list_str="",
    )
    kg_context_tokens = len(tokenizer.encode(pre_kg_context))

    # Calculate preliminary system prompt tokens
    pre_sys_prompt = sys_prompt_template.format(
        context_data="",  # Empty for overhead calculation
        response_type=response_type,
        user_prompt=user_prompt,
    )
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))

    # Calculate fixed overhead before chunk packing
    query_tokens = len(tokenizer.encode(query))
    history_messages = _build_effective_history_messages(query_param)
    history_tokens = _estimate_history_tokens(tokenizer, history_messages)
    enable_image_budget = _coerce_bool(
        getattr(
            query_param,
            "enable_image_token_budget",
            DEFAULT_ENABLE_IMAGE_TOKEN_BUDGET,
        ),
        default=DEFAULT_ENABLE_IMAGE_TOKEN_BUDGET,
    )
    budget_history_tokens = history_tokens
    buffer_tokens = 200  # reserved for reference list and safety buffer
    fixed_overhead_tokens = (
        sys_prompt_tokens
        + kg_context_tokens
        + query_tokens
        + budget_history_tokens
        + buffer_tokens
    )
    available_chunk_tokens = max_total_tokens - fixed_overhead_tokens
    if available_chunk_tokens < 0:
        logger.warning(
            "Chunk token budget below zero (%s); clamped to 0",
            available_chunk_tokens,
        )
        available_chunk_tokens = 0

    rerank_chunk_debug: dict[str, Any] = _empty_rerank_chunk_debug()
    if not enable_image_budget:
        # Official-style fallback path
        truncated_chunks = await process_chunks_unified(
            query=query,
            unique_chunks=merged_chunks,
            query_param=query_param,
            global_config=global_config,
            source_type=query_param.mode,
            chunk_token_limit=available_chunk_tokens,
            rerank_debug=rerank_chunk_debug,
        )
        image_tokens = 0
        image_count = 0
    else:
        ordered_candidates = await process_chunks_unified(
            query=query,
            unique_chunks=merged_chunks,
            query_param=query_param,
            global_config=global_config,
            source_type=query_param.mode,
            chunk_token_limit=2**31 - 1,
            rerank_debug=rerank_chunk_debug,
        )

        image_cap = int(getattr(query_param, "multimodal_top_k", 0) or 0)
        estimate_image_tokens = (
            _build_lazy_qwen_image_token_estimator(query_param)
            if image_cap > 0
            else None
        )

        truncated_chunks = []
        selected_image_paths: set[str] = set()
        skipped_image_count = 0
        total_tokens_used = fixed_overhead_tokens
        image_tokens = 0

        for chunk in ordered_candidates:
            chunk_text_tokens = _chunk_token_cost(tokenizer, chunk)
            chunk_image_tokens = 0
            new_paths_for_chunk: list[str] = []

            if estimate_image_tokens is not None and len(selected_image_paths) < image_cap:
                for image_path in _extract_image_paths_from_chunk(chunk):
                    if image_path in selected_image_paths or image_path in new_paths_for_chunk:
                        continue
                    if len(selected_image_paths) + len(new_paths_for_chunk) >= image_cap:
                        break

                    token_value = estimate_image_tokens(image_path)
                    if token_value is None:
                        skipped_image_count += 1
                        continue
                    chunk_image_tokens += token_value
                    new_paths_for_chunk.append(image_path)

            chunk_total_tokens = chunk_text_tokens + chunk_image_tokens
            if total_tokens_used + chunk_total_tokens > max_total_tokens:
                break

            truncated_chunks.append(chunk)
            total_tokens_used += chunk_total_tokens
            image_tokens += chunk_image_tokens
            if new_paths_for_chunk:
                selected_image_paths.update(new_paths_for_chunk)

        image_count = len(selected_image_paths)
        if skipped_image_count > 0:
            logger.debug(
                "Skipped %s images during per-chunk Qwen token estimation (path unreadable or image parse failed).",
                skipped_image_count,
            )

    per_image_tokens = image_tokens // image_count if image_count > 0 else 0

    logger.debug(
        "Token allocation - Total: %s, SysPrompt: %s, Query: %s, KG: %s, "
        "History: %s, Image: %s (%s * %s), Buffer: %s, Available for chunks: %s",
        max_total_tokens,
        sys_prompt_tokens,
        query_tokens,
        kg_context_tokens,
        budget_history_tokens,
        image_tokens,
        image_count,
        per_image_tokens,
        buffer_tokens,
        available_chunk_tokens,
    )

    # Generate reference list from truncated chunks using the new common function
    reference_list, truncated_chunks = generate_reference_list_from_chunks(
        truncated_chunks
    )

    # Rebuild chunks_context with truncated chunks
    # The actual tokens may be slightly less than available_chunk_tokens due to deduplication logic
    chunks_context = []
    for i, chunk in enumerate(truncated_chunks):
        chunks_context.append(
            {
                "id": chunk.get("id", ""),        # chunk-level inline citation identifier
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(chunks_context)} chunks"
    )

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context and not chunks_context:
        # Return empty raw data structure when no entities/relations
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            query_param.mode,
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        metadata = empty_raw_data.setdefault("metadata", {})
        metadata["rerank_chunk_debug"] = rerank_chunk_debug
        return "", empty_raw_data

    # output chunks tracking infomations
    # format: <source><frequency>/<order> (e.g., E5/2 R2/1 C1/1)
    if truncated_chunks and chunk_tracking:
        chunk_tracking_log = []
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_tracking:
                tracking_info = chunk_tracking[chunk_id]
                source = tracking_info["source"]
                frequency = tracking_info["frequency"]
                order = tracking_info["order"]
                chunk_tracking_log.append(f"{source}{frequency}/{order}")
            else:
                chunk_tracking_log.append("?0/0")

        if chunk_tracking_log:
            logger.info(f"Final chunks S+F/O: {' '.join(chunk_tracking_log)}")

    result = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    # Always return both context and complete data structure (unified approach)
    logger.debug(
        f"[_build_context_str] Converting to user format: {len(entities_context)} entities, {len(relations_context)} relations, {len(truncated_chunks)} chunks"
    )
    final_data = convert_to_user_format(
        entities_context,
        relations_context,
        truncated_chunks,
        reference_list,
        query_param.mode,
        entity_id_to_original,
        relation_id_to_original,
    )
    metadata = final_data.setdefault("metadata", {})
    metadata["rerank_chunk_debug"] = rerank_chunk_debug
    logger.debug(
        f"[_build_context_str] Final data after conversion: {len(final_data.get('entities', []))} entities, {len(final_data.get('relationships', []))} relationships, {len(final_data.get('chunks', []))} chunks"
    )
    return result, final_data


async def _rerank_kg_results(
    query: str,
    search_result: dict,
    query_param: QueryParam,
    global_config: dict,
) -> dict:
    """
    Rerank KG search results (entities + relations) with CrossEncoder.
    Replaces ordering of final_entities / final_relations in search_result.
    Only called when enable_rerank=True and rerank_model_func is configured.
    """
    final_entities = search_result.get("final_entities", [])
    final_relations = search_result.get("final_relations", [])

    # Rerank entities
    if final_entities:
        for e in final_entities:
            e["content"] = f"{e['entity_name']}: {e.get('description', '')}"
        reranked = await apply_rerank_if_enabled(
            query,
            final_entities,
            global_config,
            enable_rerank=True,
            top_n=len(final_entities),
            item_label="entities",
        )
        search_result["final_entities"] = reranked

    # Rerank relations
    if final_relations:
        for r in final_relations:
            src = r.get("src_id") or (r["src_tgt"][0] if "src_tgt" in r else "")
            tgt = r.get("tgt_id") or (r["src_tgt"][1] if "src_tgt" in r else "")
            r["content"] = f"{src} → {tgt}: {r.get('description', '')}"
        reranked = await apply_rerank_if_enabled(
            query,
            final_relations,
            global_config,
            enable_rerank=True,
            top_n=len(final_relations),
            item_label="relations",
        )
        search_result["final_relations"] = reranked

    return search_result


# Now let's update the old _build_query_context to use the new architecture
async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
) -> QueryContextResult | None:
    """
    Main query context building function using the new 4-stage architecture:
    1. Search -> 2. Truncate -> 3. Merge chunks -> 4. Build LLM context

    Returns unified QueryContextResult containing both context and raw_data.
    """

    if not query:
        logger.warning("Query is empty, skipping context building")
        return None

    # Stage 1: Pure search
    search_result = await _perform_kg_search(
        query,
        ll_keywords,
        hl_keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode not in ("mix", "rrf", "ppr", "ppr_local"):
            return None
        else:
            if not search_result["chunk_tracking"] and not search_result.get("ppr_chunks"):
                return None

    # Stage 1.5 (optional): Rerank entities and relations by query relevance.
    # PPR modes surface chunk-only context after graph propagation, so KG rerank
    # here is unnecessary overhead for mode="ppr"/"ppr_local".
    # enable_kg_rerank is independent of enable_rerank (chunk rerank).
    _kg_rerank_enabled = getattr(query_param, "enable_kg_rerank", query_param.enable_rerank)
    if (
        query_param.mode not in ("ppr", "ppr_local")
        and _kg_rerank_enabled
        and text_chunks_db.global_config.get("rerank_model_func")
    ):
        search_result = await _rerank_kg_results(
            query, search_result, query_param, text_chunks_db.global_config
        )

    # Stage 2: Apply token truncation for LLM efficiency
    truncation_result = await _apply_token_truncation(
        search_result,
        query_param,
        text_chunks_db.global_config,
    )

    # Stage 3: Merge chunks using filtered entities/relations
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query,
        knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=chunks_vdb,
        chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
        ppr_chunks=search_result.get("ppr_chunks"),
    )

    if (
        not merged_chunks
        and not truncation_result["entities_context"]
        and not truncation_result["relations_context"]
    ):
        return None

    # Stage 4: Build final LLM context with dynamic token processing
    # _build_context_str now always returns tuple[str, dict]
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks,
        query=query,
        query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
    )

    # Convert keywords strings to lists and add complete metadata to raw_data
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []

    # Add complete metadata to raw_data (preserve existing metadata including query_mode)
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}

    # Update keywords while preserving existing metadata
    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "entities_after_truncation": len(
            truncation_result.get("filtered_entities", [])
        ),
        "relations_after_truncation": len(
            truncation_result.get("filtered_relations", [])
        ),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }

    logger.debug(
        f"[_build_query_context] Context length: {len(context) if context else 0}"
    )
    logger.debug(
        f"[_build_query_context] Raw data entities: {len(raw_data.get('data', {}).get('entities', []))}, relationships: {len(raw_data.get('data', {}).get('relationships', []))}, chunks: {len(raw_data.get('data', {}).get('chunks', []))}"
    )

    return QueryContextResult(context=context, raw_data=raw_data)


def _min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    """Normalise score dict to [0, 1].

    Edge case: if all scores are identical and > 0, returns uniform 1.0 to
    preserve seed signal rather than collapsing everything to 0.
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        uniform = 1.0 if hi > 0.0 else 0.0
        return {k: uniform for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


async def _recognition_memory_filter(
    query: str,
    node_datas: list[dict],
    rel_results: list[dict],
    llm_model_func: Callable,
    linking_top_k: int = 5,
    tokenizer: Tokenizer | None = None,
    recognition_prompt_max_tokens: int = DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS,
    recognition_prompt_output_max_tokens: int = DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS,
    recognition_prompt_reserved_tokens: int = DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS,
) -> dict[str, float]:
    """HippoRAG2-style recognition memory filter for global PPR entity seeds.

    Three-step hybrid filter:
      1. Normalisation step — scores already retrieved by hybrid search (no new VDB calls)
      2. LLM step           — unified candidate list sent to LLM for relevance judgment
      3. Difflib step       — LLM text output remapped to graph entity_ids

    Args:
        query:             User query string.
        node_datas:        Entity VDB results (each dict has "entity_id", "vdb_score").
        rel_results:       Relation VDB results (each dict has "src_id", "tgt_id",
                           "description", "distance").
        llm_model_func:    Async callable — global_config["llm_model_func"].
        (recognition_top_k removed — all hybrid candidates passed directly; caller guards with recognition_top_k > 0)
        linking_top_k:     Max entity seeds returned (HippoRAG2 link_top_k). 0 = no cap.
        tokenizer:         Tokenizer for prompt token budgeting. If None, no budget guard.
        recognition_prompt_max_tokens: Hard cap for recognition prompt token length.
        recognition_prompt_output_max_tokens: LLM output token upper bound for recognition step.
        recognition_prompt_reserved_tokens: Safety reserve subtracted from hard cap.

    Returns:
        {entity_id: normalised_weight} for LLM-recognised entities.
        Empty dict signals fallback to direct score merge in the caller.
    """
    import difflib

    # --- Step 1: Candidate pool sizing ---
    # Pass all hybrid-retrieved candidates directly to LLM (no recognition_top_k truncation).
    # recognition_top_k is retained as an enable/disable flag only (checked by caller).
    top_rels = rel_results
    top_nodes = node_datas

    # --- Step 2: fact_scores — max across triplets for same entity ---
    fact_scores: dict[str, float] = {}
    for rel in top_rels:
        for eid in (rel.get("src_id"), rel.get("tgt_id")):
            if eid:
                fact_scores[eid] = max(fact_scores.get(eid, 0.0), rel.get("distance", 0.0))

    # --- Step 3: Independent min-max normalisation ---
    norm_vdb = _min_max_norm({
        nd["entity_id"]: nd.get("vdb_score", 0.0)
        for nd in top_nodes if nd.get("entity_id")
    })
    norm_fact = _min_max_norm(fact_scores)

    # --- Step 4: Build unified candidate list for difflib matching ---
    # entity_vdb_ids: bare IDs used for difflib matching and weight lookup
    # entity_display_lines: "id: description" strings shown in the LLM prompt
    entity_id_desc_pairs = [
        (nd["entity_id"], (nd.get("description") or "").strip()[:120])
        for nd in top_nodes if nd.get("entity_id")
    ]
    entity_vdb_ids = [eid for eid, _ in entity_id_desc_pairs]
    entity_display_lines = [
        f"{eid}: {desc}" if desc else eid
        for eid, desc in entity_id_desc_pairs
    ]

    # triplet endpoints may include entities absent from entity VDB results
    triplet_eids = list(dict.fromkeys(
        eid for rel in top_rels
        for eid in (rel.get("src_id"), rel.get("tgt_id")) if eid
    ))
    all_candidate_ids = list(dict.fromkeys(entity_vdb_ids + triplet_eids))

    if not all_candidate_ids:
        return {}

    # --- Step 5: Build LLM prompt (with token budget guard) ---
    prompt_middle = "\n\nRetrieved facts (src | relation | tgt):\n"
    prompt_suffix = (
        "\n\nReturn the relevant entity identifiers only, one per line. "
        "Output ONLY the identifier (the part before ':' if a description follows). "
        "Entities appearing as src or tgt endpoints in the facts section are also valid candidates. "
        "If none are relevant, return an empty response."
    )

    triplet_lines = [
        f"{r.get('src_id', '')} | {r.get('description', '')} | {r.get('tgt_id', '')}"
        for r in top_rels
    ]

    def _compose_prompt(entity_lines: list[str], fact_lines: list[str]) -> str:
        prompt_prefix = (
            "You are an entity relevance judge.\n\n"
            f"Query: {query}\n\n"
            "Below are retrieved entities (with descriptions) and relational facts. "
            "Select ONLY those entity identifiers directly relevant to answering the query.\n"
            "You MUST copy each entity identifier EXACTLY as it appears "
            '(including any "|TYPE" suffix). '
            "Do not rephrase, abbreviate, or invent new identifiers.\n\n"
            "Entities:\n"
        )
        standalone_block = "\n".join(entity_lines) if entity_lines else "(none)"
        triplet_block = "\n".join(fact_lines) if fact_lines else "(none)"
        return (
            prompt_prefix
            + standalone_block
            + prompt_middle
            + triplet_block
            + prompt_suffix
        )

    prompt: str
    if tokenizer is None:
        prompt = _compose_prompt(entity_display_lines, triplet_lines)
        completion_max_tokens = max(
            1,
            _to_non_negative_int(
                recognition_prompt_output_max_tokens,
                default=DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS,
            ),
        )
    else:
        max_context_tokens = max(
            512,
            _to_non_negative_int(
                recognition_prompt_max_tokens,
                default=DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS,
            ),
        )
        completion_max_tokens = max(
            1,
            _to_non_negative_int(
                recognition_prompt_output_max_tokens,
                default=DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS,
            ),
        )
        reserved_tokens = _to_non_negative_int(
            recognition_prompt_reserved_tokens,
            default=DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS,
        )
        # Keep a guaranteed prompt floor while respecting model context limits.
        min_prompt_tokens = 256
        max_completion_tokens = max(1, max_context_tokens - min_prompt_tokens)
        completion_max_tokens = min(completion_max_tokens, max_completion_tokens)
        max_reserved_tokens = max(0, max_context_tokens - completion_max_tokens - min_prompt_tokens)
        reserved_tokens = min(reserved_tokens, max_reserved_tokens)
        token_budget = max(
            min_prompt_tokens,
            max_context_tokens - completion_max_tokens - reserved_tokens,
        )

        selected_entities: list[str] = []
        selected_facts: list[str] = []

        def _fits_budget(candidate_entities: list[str], candidate_facts: list[str]) -> bool:
            candidate_prompt = _compose_prompt(candidate_entities, candidate_facts)
            return len(tokenizer.encode(candidate_prompt)) <= token_budget

        # Ensure both sections have a chance to contribute before filling remaining budget.
        if entity_display_lines and _fits_budget([entity_display_lines[0]], []):
            selected_entities.append(entity_display_lines[0])
        if triplet_lines and _fits_budget(selected_entities, [triplet_lines[0]]):
            selected_facts.append(triplet_lines[0])

        for line in entity_display_lines[1:] if selected_entities else entity_display_lines:
            candidate_entities = selected_entities + [line]
            if _fits_budget(candidate_entities, selected_facts):
                selected_entities = candidate_entities

        for fact_line in triplet_lines[1:] if selected_facts else triplet_lines:
            candidate_facts = selected_facts + [fact_line]
            if _fits_budget(selected_entities, candidate_facts):
                selected_facts = candidate_facts

        prompt = _compose_prompt(selected_entities, selected_facts)
        prompt_tokens = len(tokenizer.encode(prompt))
        if (
            len(selected_entities) < len(entity_vdb_ids)
            or len(selected_facts) < len(triplet_lines)
        ):
            logger.warning(
                "PPR(global): recognition prompt truncated by token budget "
                "(prompt_tokens=%d, budget=%d, completion_max_tokens=%d, reserved_tokens=%d, entities=%d/%d, facts=%d/%d)",
                prompt_tokens,
                token_budget,
                completion_max_tokens,
                reserved_tokens,
                len(selected_entities),
                len(entity_vdb_ids),
                len(selected_facts),
                len(triplet_lines),
            )
        if prompt_tokens > token_budget:
            logger.warning(
                "PPR(global): recognition prompt still exceeds budget after truncation "
                "(prompt_tokens=%d, budget=%d, completion_max_tokens=%d, reserved_tokens=%d)",
                prompt_tokens,
                token_budget,
                completion_max_tokens,
                reserved_tokens,
            )

    # --- Step 6: LLM call ---
    try:
        llm_output: str = await llm_model_func(
            prompt, max_tokens=completion_max_tokens
        )
    except TypeError:
        # Backward compatibility for custom llm_model_func without max_tokens kwarg.
        llm_output = await llm_model_func(prompt)
    except Exception as e:
        logger.warning(f"PPR: recognition memory LLM call failed: {e}")
        return {}

    # --- Step 7: Difflib mapping ---
    # Strip description suffix (": ...") the LLM may have echoed back, then fuzzy-match.
    # cutoff=0.6 tolerates minor formatting drift (e.g. missing |TYPE suffix) while
    # still rejecting clearly wrong identifiers.
    recognized_ids: set[str] = set()
    for line in llm_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove anything after the first ": " that the LLM may have copied from the prompt
        candidate_id = line.split(": ")[0].strip()
        matches = difflib.get_close_matches(candidate_id, all_candidate_ids, n=1, cutoff=0.6)
        if matches:
            recognized_ids.add(matches[0])

    # --- Step 8: Merge weights ---
    result: dict[str, float] = {}
    for eid in recognized_ids:
        w = max(norm_vdb.get(eid, 0.0), norm_fact.get(eid, 0.0))
        if w > 0.0:
            result[eid] = w

    # --- Step 9: Truncate to top linking_top_k (HippoRAG2 link_top_k) ---
    if linking_top_k > 0 and len(result) > linking_top_k:
        result = dict(
            sorted(result.items(), key=lambda x: x[1], reverse=True)[:linking_top_k]
        )
        logger.debug(
            f"PPR(global): recognition memory truncated to linking_top_k={linking_top_k} seeds"
        )

    return result


def _direct_merge_seeds(
    node_datas: list[dict],
    rel_results: list[dict],
) -> dict[str, float]:
    """Direct max-merge of entity VDB + relation VDB scores into seed weights.

    Used as the fallback when recognition memory is disabled, unavailable, or
    returns an empty result.
    """
    weights: dict[str, float] = {}
    for nd in node_datas:
        eid = nd.get("entity_id", nd.get("entity_name", ""))
        if eid:
            weights[eid] = max(weights.get(eid, 0), nd.get("vdb_score", 0.0))
    for rel in rel_results:
        score = rel.get("distance", 0.0)
        for field_name in ("src_id", "tgt_id"):
            eid = rel.get(field_name)
            if eid:
                weights[eid] = max(weights.get(eid, 0), score)
    return weights


async def _ppr_rank_chunks_global(
    query: str,
    entity_seed_weights: dict[str, float],
    knowledge_graph_inst: BaseGraphStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    query_embedding: list[float] = None,
) -> list[dict]:
    """Global PPR path: full-graph engine, skips BFS subgraph extraction.

    entity_seed_weights must already have hub-penalty applied (raw, not yet normalised).
    """
    from lightrag.ppr_engine import get_engine

    engine = get_engine(knowledge_graph_inst)

    passage_node_weight = query_param.passage_node_weight

    # Normalise entity seeds to sum=1
    entity_total = sum(entity_seed_weights.values())
    if entity_total > 0:
        entity_seed_weights = {k: v / entity_total for k, v in entity_seed_weights.items()}

    await engine._ensure_loaded()

    # HippoRAG2-aligned full-graph chunk pool: ALL chunks in the knowledge graph are
    # included as PPR nodes, not just neighbours of seed entities.  This enables true
    # multi-hop passage discovery: PPR energy propagates from entity seeds through the
    # entity graph and can reach any passage node regardless of VDB similarity.
    chunk_to_entities: dict[str, list[str]] = dict(engine._chunk_to_entities)

    if not chunk_to_entities:
        logger.debug("PPR(global): no chunk mappings in graph")
        return []

    # DPR chunk seeds: VDB scores over the full chunk pool.
    # Chunks in the VDB result that exist in the graph receive an initial DPR weight;
    # the remaining graph chunks start with zero DPR weight but still participate as
    # PPR nodes and can accumulate score through entity-graph energy propagation.
    chunk_seed_weights: dict[str, float] = {}
    try:
        chunk_results = await _query_vector_storage(
            chunks_vdb,
            query,
            query_param.ppr_top_k * 4,
            query_param,
            query_embedding,
        )
        if chunk_results:
            scores = [c.get("distance", 0.0) for c in chunk_results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            for c in chunk_results:
                cid = c.get("chunk_id") or c.get("id")
                if cid and cid in chunk_to_entities:
                    normalized = (c.get("distance", 0.0) - min_s) / range_s
                    chunk_seed_weights[cid] = normalized
    except Exception as e:
        logger.warning(f"PPR(global): chunks VDB query failed: {e}")

    # Normalise chunk seeds to sum=1, then scale by passage_node_weight
    chunk_total = sum(chunk_seed_weights.values())
    if chunk_total > 0:
        chunk_seed_weights = {
            k: (v / chunk_total) * passage_node_weight
            for k, v in chunk_seed_weights.items()
        }

    exclude_synonym_edges = _should_exclude_synonym_edges(query_param)
    ppr_ranked = await engine.run_ppr(
        entity_seed_weights=entity_seed_weights,
        chunk_seed_weights=chunk_seed_weights,
        chunk_to_entities=chunk_to_entities,
        damping=query_param.ppr_damping,
        top_k=query_param.ppr_top_k,
        ppr_synonym_weight_mode=query_param.ppr_synonym_weight_mode,
        exclude_synonym_edges=exclude_synonym_edges,
    )

    if not ppr_ranked:
        return []

    ranked_chunk_ids = [cid for cid, _ in ppr_ranked]
    chunk_data_list = await text_chunks_db.get_by_ids(ranked_chunk_ids)

    result_chunks = []
    for (chunk_id, ppr_score), chunk_data in zip(ppr_ranked, chunk_data_list):
        if chunk_data and chunk_data.get("content"):
            chunk = chunk_data.copy()
            chunk["chunk_id"] = chunk_id
            chunk["source_type"] = "ppr"
            chunk["ppr_score"] = ppr_score
            result_chunks.append(chunk)

    logger.info(
        f"PPR(global): {len(result_chunks)} chunks ranked, "
        f"{len(chunk_to_entities)} graph chunk nodes (full-graph pool), "
        f"{len(chunk_seed_weights)} DPR seeds"
    )
    return result_chunks


async def _ppr_rank_chunks(
    query: str,
    node_datas: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    query_embedding: list[float] = None,
    use_global: bool = False,
) -> list[dict]:
    """PPR chunk ranking — local BFS (use_global=False) or full-graph (use_global=True).

    Local path: BFS subgraph from seed entities → nx.pagerank on heterogeneous graph.
    Global path: full entity graph cached in GlobalPPREngine → pagerank_power via
                 scipy CSR + fast-pagerank.

    Returns:
        Ranked list of chunk dicts with ``ppr_score`` and ``source_type="ppr"``.
    """
    from lightrag.ppr import personalized_pagerank

    # Step 1: Build entity seed weights
    # Global PPR path: recognition memory filters seeds via LLM (when enabled).
    # Local PPR path: direct max-merge (unchanged behaviour).
    entity_seed_weights: dict[str, float] = {}

    # Always fetch relation VDB results — used by both paths
    rel_results: list[dict] = []
    try:
        rel_results = await _query_vector_storage(
            relationships_vdb,
            query,
            query_param.top_k,
            query_param,
            query_embedding,
        )
    except Exception as e:
        logger.warning(f"PPR: relation VDB query failed: {e}")

    exclude_synonym_edges = _should_exclude_synonym_edges(query_param)
    if exclude_synonym_edges and rel_results:
        before = len(rel_results)
        rel_results = [r for r in rel_results if not _is_synonym_edge(r)]
        filtered = before - len(rel_results)
        if filtered > 0:
            logger.info(f"PPR: excluded {filtered} SYNONYM relation seeds by query flag")

    if use_global and query_param.recognition_top_k > 0:
        # --- Recognition Memory path (global PPR only) ---
        llm_func = text_chunks_db.global_config.get("llm_model_func")
        tokenizer = text_chunks_db.global_config.get("tokenizer")
        recognition_prompt_max_tokens = _to_non_negative_int(
            text_chunks_db.global_config.get(
                "recognition_prompt_max_tokens",
                DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS,
            ),
            default=DEFAULT_RECOGNITION_PROMPT_MAX_TOKENS,
        )
        recognition_prompt_output_max_tokens = _to_non_negative_int(
            text_chunks_db.global_config.get(
                "recognition_prompt_output_max_tokens",
                DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS,
            ),
            default=DEFAULT_RECOGNITION_PROMPT_OUTPUT_MAX_TOKENS,
        )
        recognition_prompt_reserved_tokens = _to_non_negative_int(
            text_chunks_db.global_config.get(
                "recognition_prompt_reserved_tokens",
                DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS,
            ),
            default=DEFAULT_RECOGNITION_PROMPT_RESERVED_TOKENS,
        )
        if llm_func and (node_datas or rel_results):
            try:
                recognized = await _recognition_memory_filter(
                    query=query,
                    node_datas=node_datas,
                    rel_results=rel_results,
                    llm_model_func=llm_func,
                    linking_top_k=getattr(query_param, "linking_top_k", 5),
                    tokenizer=tokenizer,
                    recognition_prompt_max_tokens=recognition_prompt_max_tokens,
                    recognition_prompt_output_max_tokens=recognition_prompt_output_max_tokens,
                    recognition_prompt_reserved_tokens=recognition_prompt_reserved_tokens,
                )
            except Exception as e:
                logger.warning(
                    f"PPR(global): recognition memory failed, falling back to direct merge: {e}"
                )
                recognized = {}
            if recognized:
                entity_seed_weights = recognized
                seed_ids = list(recognized.keys())
                logger.info(
                    f"PPR(global): recognition memory accepted {len(recognized)} seeds: {seed_ids}"
                )
            else:
                logger.info(
                    "PPR(global): recognition memory returned empty; using direct seed merge"
                )
                entity_seed_weights = _direct_merge_seeds(node_datas, rel_results)
        else:
            # No LLM configured — direct merge
            logger.info("PPR(global): no LLM configured; using direct seed merge")
            entity_seed_weights = _direct_merge_seeds(node_datas, rel_results)
    else:
        # Local PPR path OR recognition_top_k=0 (disabled): direct max-merge
        entity_seed_weights = _direct_merge_seeds(node_datas, rel_results)

    if not entity_seed_weights:
        return []

    # Step 1b: Hub penalty – down-weight high-degree generic entities
    if query_param.hub_penalty_threshold > 0:
        seed_ids_for_degree = list(entity_seed_weights.keys())
        try:
            degrees = await knowledge_graph_inst.node_degrees_batch(seed_ids_for_degree)
            for eid in seed_ids_for_degree:
                deg = degrees.get(eid, 0)
                if deg > query_param.hub_penalty_threshold:
                    entity_seed_weights[eid] /= math.log(1 + deg)
        except Exception as e:
            logger.warning(f"PPR: hub penalty degree query failed: {e}")

    # ------------------------------------------------------------------ #
    # Global PPR path: skip BFS, use full-graph GlobalPPREngine            #
    # ------------------------------------------------------------------ #
    if use_global:
        return await _ppr_rank_chunks_global(
            query=query,
            entity_seed_weights=entity_seed_weights,
            knowledge_graph_inst=knowledge_graph_inst,
            chunks_vdb=chunks_vdb,
            text_chunks_db=text_chunks_db,
            query_param=query_param,
            query_embedding=query_embedding,
        )

    # Step 2: Get subgraph (entities + edges)
    seed_ids = list(entity_seed_weights.keys())
    subgraph_nodes, subgraph_edges = await knowledge_graph_inst.get_subgraph_for_ppr(
        seed_ids, max_depth=query_param.multi_hop_depth
    )
    if exclude_synonym_edges and subgraph_edges:
        before = len(subgraph_edges)
        subgraph_edges = [e for e in subgraph_edges if not _is_synonym_edge(e)]
        filtered = before - len(subgraph_edges)
        if filtered > 0:
            logger.info(
                f"PPR(local): excluded {filtered} SYNONYM edges from subgraph by query flag"
            )

    # Step 3: Build virtual chunk nodes + chunk-entity edges from source_id
    chunk_to_entities: dict[str, list[str]] = {}
    for node in subgraph_nodes:
        eid = node.get("entity_id")
        source_ids = node.get("source_id", "")
        if eid and source_ids:
            for chunk_id in split_string_by_multi_markers(
                source_ids, [GRAPH_FIELD_SEP]
            ):
                chunk_id = chunk_id.strip()
                if chunk_id:
                    chunk_to_entities.setdefault(chunk_id, []).append(eid)

    # Also include chunks referenced by edges (relation source_id)
    for edge in subgraph_edges:
        edge_source_ids = edge.get("source_id", "")
        if edge_source_ids:
            src, tgt = edge.get("src"), edge.get("tgt")
            for chunk_id in split_string_by_multi_markers(
                edge_source_ids, [GRAPH_FIELD_SEP]
            ):
                chunk_id = chunk_id.strip()
                if chunk_id:
                    if src:
                        chunk_to_entities.setdefault(chunk_id, []).append(src)
                    if tgt:
                        chunk_to_entities.setdefault(chunk_id, []).append(tgt)

    if not chunk_to_entities:
        logger.debug("PPR: no chunk-entity mappings found in subgraph")
        return []

    chunk_nodes = [{"chunk_id": cid} for cid in chunk_to_entities]
    chunk_entity_edges = [
        {"chunk_id": cid, "entity_id": eid}
        for cid, eids in chunk_to_entities.items()
        for eid in eids
    ]

    # Step 4: Get DPR chunk scores (passage_node_weight)
    chunk_seed_weights: dict[str, float] = {}
    passage_node_weight = query_param.passage_node_weight

    try:
        chunk_results = await _query_vector_storage(
            chunks_vdb,
            query,
            query_param.ppr_top_k * 2,
            query_param,
            query_embedding,
        )
        if chunk_results:
            scores = [c.get("distance", 0.0) for c in chunk_results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0

            for c in chunk_results:
                cid = c.get("chunk_id") or c.get("id")
                if cid and cid in chunk_to_entities:
                    normalized = (c.get("distance", 0.0) - min_s) / range_s
                    chunk_seed_weights[cid] = normalized  # raw score; scaled below
    except Exception as e:
        logger.warning(f"PPR: chunks VDB query failed: {e}")

    # Normalize entity seeds to sum=1 (after hub penalty)
    entity_seed_total = sum(entity_seed_weights.values())
    if entity_seed_total > 0:
        entity_seed_weights = {
            k: v / entity_seed_total for k, v in entity_seed_weights.items()
        }

    # Normalize chunk seeds to sum=1, then scale the whole group by passage_node_weight
    chunk_seed_total = sum(chunk_seed_weights.values())
    if chunk_seed_total > 0:
        chunk_seed_weights = {
            k: (v / chunk_seed_total) * passage_node_weight
            for k, v in chunk_seed_weights.items()
        }

    # Step 5: Run PPR
    ppr_ranked = personalized_pagerank(
        entity_nodes=subgraph_nodes,
        entity_edges=subgraph_edges,
        chunk_nodes=chunk_nodes,
        chunk_entity_edges=chunk_entity_edges,
        entity_seed_weights=entity_seed_weights,
        chunk_seed_weights=chunk_seed_weights,
        damping=query_param.ppr_damping,
        top_k=query_param.ppr_top_k,
        ppr_synonym_weight_mode=query_param.ppr_synonym_weight_mode,
        exclude_synonym_edges=exclude_synonym_edges,
    )

    if not ppr_ranked:
        return []

    # Step 6: Fetch chunk content and return
    ranked_chunk_ids = [cid for cid, _ in ppr_ranked]
    chunk_data_list = await text_chunks_db.get_by_ids(ranked_chunk_ids)

    result_chunks = []
    for (chunk_id, ppr_score), chunk_data in zip(ppr_ranked, chunk_data_list):
        if chunk_data and chunk_data.get("content"):
            chunk = chunk_data.copy()
            chunk["chunk_id"] = chunk_id
            chunk["source_type"] = "ppr"
            chunk["ppr_score"] = ppr_score
            result_chunks.append(chunk)

    logger.info(
        f"PPR chunk ranking: {len(result_chunks)} chunks from {len(subgraph_nodes)} entities, "
        f"{len(chunk_to_entities)} virtual chunks"
    )

    return result_chunks


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query} (top_k:{query_param.top_k}, cosine:{entities_vdb.cosine_better_than_threshold})"
    )

    results = await _query_vector_storage(
        entities_vdb,
        query,
        query_param.top_k,
        query_param,
    )

    if not len(results):
        return [], []

    # Extract all entity IDs from your results list
    # Use composite entity_id (from VDB metadata) when available, fall back to entity_name
    node_ids = [r.get("entity_id", r["entity_name"]) for r in results]

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    retrieved_nodes = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in retrieved_nodes]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = []
    for k, n, d in zip(results, retrieved_nodes, node_degrees):
        if n is None:
            continue
        entity_name = k["entity_name"]
        node_datas.append(
            {
                **n,
                "entity_id": n.get("entity_id", k.get("entity_id", entity_name)),
                "entity_name": entity_name,
                "rank": d,
                "vdb_score": k.get("distance", 0.0),
                "created_at": k.get("created_at"),
            }
        )

    # In PPR modes the top-k entity VDB results are the seed candidates; skip
    # one-hop edge expansion to avoid flooding the seed pool with noisy neighbours.
    if query_param.mode in ("ppr", "ppr_local"):
        use_relations = []
    else:
        use_relations = await _find_most_related_edges_from_entities(
            node_datas,
            query_param,
            knowledge_graph_inst,
        )

    logger.info(
        f"Local query: {len(node_datas)} entites, {len(use_relations)} relations"
    )

    # Entities are sorted by cosine similarity
    # Relations are sorted by rank + weight
    return node_datas, use_relations


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    node_ids = []
    for dp in node_datas:
        node_id = dp.get("entity_id") or dp.get("entity_name")
        if node_id:
            node_ids.append(node_id)

    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_ids)

    all_edges = []
    seen = set()

    for node_id in node_ids:
        this_edges = batch_edges_dict.get(node_id, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    exclude_synonym_edges = _should_exclude_synonym_edges(query_param)
    filtered_degree_func = getattr(
        knowledge_graph_inst,
        "edge_degrees_batch_excluding_synonym",
        None,
    )
    edge_degree_coro = (
        filtered_degree_func(edge_pairs_tuples)
        if exclude_synonym_edges and callable(filtered_degree_func)
        else knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples)
    )
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        edge_degree_coro,
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    if exclude_synonym_edges:
        before = len(all_edges_data)
        all_edges_data = [e for e in all_edges_data if not _is_synonym_edge(e)]
        filtered = before - len(all_edges_data)
        if filtered > 0:
            logger.info(
                f"Local query: excluded {filtered} SYNONYM relations by query flag"
            )

    # Weight edges by endpoint entity VDB relevance scores (hub noise fix)
    entity_score_map: dict[str, float] = {}
    for dp in node_datas:
        score = dp.get("vdb_score", 0.0)
        for key in (dp.get("entity_id"), dp.get("entity_name")):
            if key:
                entity_score_map[key] = max(entity_score_map.get(key, 0.0), score)
    for edge in all_edges_data:
        src, tgt = edge["src_tgt"]
        edge["_entity_relevance"] = max(
            entity_score_map.get(src, 0.0), entity_score_map.get(tgt, 0.0)
        )

    # Sort: query relevance first, weight secondary, degree as tiebreaker
    all_edges_data = sorted(
        all_edges_data,
        key=lambda x: (x["_entity_relevance"], x.get("weight", 1.0), x["rank"]),
        reverse=True,
    )

    return all_edges_data


async def _find_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to entities using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(node_datas)} entities")

    if not node_datas:
        return []

    # Step 1: Collect all text chunks for each entity
    entities_with_chunks = []
    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(
                entity["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )

    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned entities)
    chunk_occurrence_count = {}
    for entity_info in entities_with_chunks:
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier entity, so skip it

        # Update entity's chunks to deduplicated chunks
        entity_info["chunks"] = deduplicated_chunks

    # Step 3: Sort chunks for each entity by occurrence count (higher count = higher priority)
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        sorted_chunks = sorted(
            entity_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        entity_info["sorted_chunks"] = sorted_chunks
        total_entity_chunks += len(sorted_chunks)

    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    # Step 4: Apply the selected chunk selection algorithm
    # Pick by vector similarity:
    #     The order of text chunks aligns with the naive retrieval's destination.
    #     When reranking is disabled, the text chunks delivered to the LLM tend to favor naive retrieval.
    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(entities_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=entities_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No entity-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Pick by entity and chunk weight:
        #     When reranking is disabled, delivered more solely KG related chunks to the LLM
        selected_chunk_ids = pick_by_weighted_polling(
            entities_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by weighted polling"
        )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final entity-related results
                }

    return result_chunks


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords} (top_k:{query_param.top_k}, cosine:{relationships_vdb.cosine_better_than_threshold})"
    )

    results = await _query_vector_storage(
        relationships_vdb,
        keywords,
        query_param.top_k,
        query_param,
    )

    if not len(results):
        return [], []

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            # Keep edge data without rank, maintain vector search order
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    if _should_exclude_synonym_edges(query_param):
        before = len(edge_datas)
        edge_datas = [e for e in edge_datas if not _is_synonym_edge(e)]
        filtered = before - len(edge_datas)
        if filtered > 0:
            logger.info(
                f"Global query: excluded {filtered} SYNONYM relations by query flag"
            )

    # Relations maintain vector search order (sorted by similarity)

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entites, {len(edge_datas)} relations"
    )

    return edge_datas, use_entities


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Only get nodes data, no need for node degrees
    nodes_dict = await knowledge_graph_inst.get_nodes_batch(entity_names)

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name, no rank needed
        combined = {**node, "entity_name": entity_name}
        node_datas.append(combined)

    return node_datas


async def _find_related_text_unit_from_relations(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    entity_chunks: list[dict] = None,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to relationships using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(edge_datas)} relations")

    if not edge_datas:
        return []

    # Step 1: Collect all text chunks for each relationship
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(
                relation["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                # Build relation identifier
                if "src_tgt" in relation:
                    rel_key = tuple(sorted(relation["src_tgt"]))
                else:
                    rel_key = tuple(
                        sorted([relation.get("src_id"), relation.get("tgt_id")])
                    )

                relations_with_chunks.append(
                    {
                        "relation_key": rel_key,
                        "chunks": chunks,
                        "relation_data": relation,
                    }
                )

    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned relationships)
    # Also remove duplicates with entity_chunks

    # Extract chunk IDs from entity_chunks for deduplication
    entity_chunk_ids = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)

    chunk_occurrence_count = {}
    # Track unique chunk_ids that have been removed to avoid double counting
    removed_entity_chunk_ids = set()

    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            # Skip chunks that already exist in entity_chunks
            if chunk_id in entity_chunk_ids:
                # Only count each unique chunk_id once
                removed_entity_chunk_ids.add(chunk_id)
                continue

            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier relationship, so skip it

        # Update relationship's chunks to deduplicated chunks
        relation_info["chunks"] = deduplicated_chunks

    # Check if any relations still have chunks after deduplication
    relations_with_chunks = [
        relation_info
        for relation_info in relations_with_chunks
        if relation_info["chunks"]
    ]

    if not relations_with_chunks:
        logger.info(
            f"Find no additional relations-related chunks from {len(edge_datas)} relations"
        )
        return []

    # Step 3: Sort chunks for each relationship by occurrence count (higher count = higher priority)
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        sorted_chunks = sorted(
            relation_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        relation_info["sorted_chunks"] = sorted_chunks
        total_relation_chunks += len(sorted_chunks)

    logger.info(
        f"Find {total_relation_chunks} additional chunks in {len(relations_with_chunks)} relations (deduplicated {len(removed_entity_chunk_ids)})"
    )

    # Step 4: Apply the selected chunk selection algorithm
    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(relations_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=relations_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No relation-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Apply linear gradient weighted polling algorithm
        selected_chunk_ids = pick_by_weighted_polling(
            relations_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by weighted polling"
        )

    logger.debug(
        f"KG related chunks: {len(entity_chunks)} from entitys, {len(selected_chunk_ids)} from relations"
    )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final relation-related results
                }

    return result_chunks


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    return_raw_data: Literal[True] = True,
) -> dict[str, Any]: ...


@overload
async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    return_raw_data: Literal[False] = False,
) -> str | AsyncIterator[str]: ...


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> QueryResult | None:
    """
    Execute naive query and return unified QueryResult object.

    Args:
        query: Query string
        chunks_vdb: Document chunks vector database
        query_param: Query parameters
        global_config: Global configuration
        hashing_kv: Cache storage
        system_prompt: System prompt

    Returns:
        QueryResult | None: Unified query result object containing:
            - content: Non-streaming response text content
            - response_iterator: Streaming response iterator
            - raw_data: Complete structured data (including references and metadata)
            - is_streaming: Whether this is a streaming result

        Returns None when no relevant chunks are retrieved.
    """

    if not query:
        return QueryResult(content=PROMPTS["fail_response"])

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    tokenizer: Tokenizer = global_config["tokenizer"]
    if not tokenizer:
        logger.error("Tokenizer not found in global configuration.")
        return QueryResult(content=PROMPTS["fail_response"])

    chunks = await _get_vector_context(query, chunks_vdb, query_param, None)

    if chunks is None or len(chunks) == 0:
        logger.info(
            "[naive_query] No relevant document chunks found; returning no-result."
        )
        return None

    # Calculate dynamic token limit for chunks
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Calculate system prompt template tokens (excluding context payload fields)
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Use the provided system prompt or default
    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )

    # Create a preliminary system prompt with empty context placeholders to
    # calculate overhead. Keep both keys for backward compatibility with custom
    # templates using either {context_data} or {content_data}.
    pre_sys_prompt = sys_prompt_template.format(
        response_type=response_type,
        user_prompt=user_prompt,
        context_data="",  # Empty for overhead calculation
        content_data="",  # Empty for overhead calculation
    )

    # Calculate fixed overhead before chunk packing
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))
    query_tokens = len(tokenizer.encode(query))
    history_messages = _build_effective_history_messages(query_param)
    history_tokens = _estimate_history_tokens(tokenizer, history_messages)
    enable_image_budget = _coerce_bool(
        getattr(
            query_param,
            "enable_image_token_budget",
            DEFAULT_ENABLE_IMAGE_TOKEN_BUDGET,
        ),
        default=DEFAULT_ENABLE_IMAGE_TOKEN_BUDGET,
    )
    budget_history_tokens = history_tokens
    buffer_tokens = 200  # reserved for reference list and safety buffer
    fixed_overhead_tokens = (
        sys_prompt_tokens
        + query_tokens
        + budget_history_tokens
        + buffer_tokens
    )
    available_chunk_tokens = max_total_tokens - fixed_overhead_tokens
    if available_chunk_tokens < 0:
        logger.warning(
            "Naive chunk token budget below zero (%s); clamped to 0",
            available_chunk_tokens,
        )
        available_chunk_tokens = 0

    rerank_chunk_debug: dict[str, Any] = {}
    if not enable_image_budget:
        # Official-style fallback path
        processed_chunks = await process_chunks_unified(
            query=query,
            unique_chunks=chunks,
            query_param=query_param,
            global_config=global_config,
            source_type="vector",
            chunk_token_limit=available_chunk_tokens,
            rerank_debug=rerank_chunk_debug,
        )
        image_tokens = 0
        image_count = 0
    else:
        ordered_candidates = await process_chunks_unified(
            query=query,
            unique_chunks=chunks,
            query_param=query_param,
            global_config=global_config,
            source_type="vector",
            chunk_token_limit=2**31 - 1,
            rerank_debug=rerank_chunk_debug,
        )

        image_cap = int(getattr(query_param, "multimodal_top_k", 0) or 0)
        estimate_image_tokens = (
            _build_lazy_qwen_image_token_estimator(query_param)
            if image_cap > 0
            else None
        )

        processed_chunks = []
        selected_image_paths: set[str] = set()
        skipped_image_count = 0
        total_tokens_used = fixed_overhead_tokens
        image_tokens = 0

        for chunk in ordered_candidates:
            chunk_text_tokens = _chunk_token_cost(tokenizer, chunk)
            chunk_image_tokens = 0
            new_paths_for_chunk: list[str] = []

            if estimate_image_tokens is not None and len(selected_image_paths) < image_cap:
                for image_path in _extract_image_paths_from_chunk(chunk):
                    if image_path in selected_image_paths or image_path in new_paths_for_chunk:
                        continue
                    if len(selected_image_paths) + len(new_paths_for_chunk) >= image_cap:
                        break

                    token_value = estimate_image_tokens(image_path)
                    if token_value is None:
                        skipped_image_count += 1
                        continue
                    chunk_image_tokens += token_value
                    new_paths_for_chunk.append(image_path)

            chunk_total_tokens = chunk_text_tokens + chunk_image_tokens
            if total_tokens_used + chunk_total_tokens > max_total_tokens:
                break

            processed_chunks.append(chunk)
            total_tokens_used += chunk_total_tokens
            image_tokens += chunk_image_tokens
            if new_paths_for_chunk:
                selected_image_paths.update(new_paths_for_chunk)

        image_count = len(selected_image_paths)
        if skipped_image_count > 0:
            logger.debug(
                "Skipped %s images during per-chunk Qwen token estimation (path unreadable or image parse failed).",
                skipped_image_count,
            )

    per_image_tokens = image_tokens // image_count if image_count > 0 else 0

    logger.debug(
        "Naive query token allocation - Total: %s, SysPrompt: %s, Query: %s, "
        "History: %s, Image: %s (%s * %s), Buffer: %s, Available for chunks: %s",
        max_total_tokens,
        sys_prompt_tokens,
        query_tokens,
        budget_history_tokens,
        image_tokens,
        image_count,
        per_image_tokens,
        buffer_tokens,
        available_chunk_tokens,
    )

    # Generate reference list from processed chunks using the new common function
    reference_list, processed_chunks_with_ref_ids = generate_reference_list_from_chunks(
        processed_chunks
    )

    logger.info(f"Final context: {len(processed_chunks_with_ref_ids)} chunks")

    # Build raw data structure for naive mode using processed chunks with reference IDs
    raw_data = convert_to_user_format(
        [],  # naive mode has no entities
        [],  # naive mode has no relationships
        processed_chunks_with_ref_ids,
        reference_list,
        "naive",
    )

    # Add complete metadata for naive mode
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}
    raw_data["metadata"]["keywords"] = {
        "high_level": [],  # naive mode has no keyword extraction
        "low_level": [],  # naive mode has no keyword extraction
    }
    raw_data["metadata"]["processing_info"] = {
        "total_chunks_found": len(chunks),
        "final_chunks_count": len(processed_chunks_with_ref_ids),
    }
    raw_data["metadata"]["rerank_chunk_debug"] = rerank_chunk_debug

    # Build chunks_context from processed chunks with reference IDs
    chunks_context = []
    for i, chunk in enumerate(processed_chunks_with_ref_ids):
        chunks_context.append(
            {
                "reference_id": chunk["reference_id"],
                "content": chunk["content"],
            }
        )

    text_units_str = "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )

    naive_context_template = PROMPTS["naive_query_context"]
    context_content = naive_context_template.format(
        text_chunks_str=text_units_str,
        reference_list_str=reference_list_str,
    )

    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(content=context_content, raw_data=raw_data)

    sys_prompt = sys_prompt_template.format(
        response_type=query_param.response_type,
        user_prompt=user_prompt,
        context_data=context_content,
        content_data=context_content,
    )

    user_query = query

    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=raw_data)

    # Handle cache
    effective_history_messages = history_messages
    history_signature = _history_messages_signature(effective_history_messages)

    query_cache_params = _build_query_cache_params(
        query_param,
        history_signature=history_signature,
        user_prompt=query_param.user_prompt or "",
        system_prompt=sys_prompt,
    )
    args_hash = _compute_query_cache_args_hash(query, query_cache_params)
    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )
    if cached_result is not None:
        cached_response, _ = cached_result  # Extract content, ignore timestamp
        logger.info(
            " == LLM cache == Query cache hit, using cached response as query result"
        )
        response = cached_response
    else:
        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
            history_messages=effective_history_messages,
            enable_cot=True,
            stream=query_param.stream,
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = dict(query_cache_params)
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    mode=query_param.mode,
                    cache_type="query",
                    queryparam=queryparam_dict,
                ),
            )

    # Return unified result based on actual response type
    if isinstance(response, str):
        # Non-streaming response (string)
        if len(response) > len(sys_prompt):
            response = (
                response[len(sys_prompt) :]
                .replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return QueryResult(content=response, raw_data=raw_data)
    else:
        # Streaming response (AsyncIterator)
        return QueryResult(
            response_iterator=response, raw_data=raw_data, is_streaming=True
        )
