"""
Shared query message repacking helpers.

These helpers normalize LightRAG's prompt layout into a consistent
"system instructions + user context/question" structure for both
non-enhanced and enhanced query paths.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

TEXT_CONTEXT_SUFFIX = "Please answer based on the context provided."
IMAGE_CONTEXT_SUFFIX = "Please answer based on the context and images provided."
_EMPTY_PROMPT_MARKERS = {"", "{user_prompt}", "none", "null", "n/a"}


def normalize_optional_instruction_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in _EMPTY_PROMPT_MARKERS:
        return ""
    return text


def _extract_last_context_segment(raw_text: str) -> Tuple[str, str]:
    # Keep the last ---Context--- block to avoid matching earlier examples.
    context_idx = raw_text.rfind("---Context---")
    if context_idx < 0:
        return "", raw_text.strip()
    prefix = raw_text[:context_idx].strip()
    context = raw_text[context_idx:].strip()
    return prefix, context


def _drop_empty_additional_instructions(text: str) -> str:
    # Remove empty Additional Instructions placeholders from system prompt.
    out_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(
            r"^(?:\d+\.\s*)?Additional Instructions:\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            payload = match.group(1).strip()
            if payload.lower() in _EMPTY_PROMPT_MARKERS:
                continue
        out_lines.append(raw_line)
    return "\n".join(out_lines).strip()


def _compose_final_system(role_prefix: str, upstream_system: str) -> str:
    # Keep LightRAG role instructions; prepend upstream system when provided.
    role_text = _drop_empty_additional_instructions(role_prefix.strip())
    upstream_text = upstream_system.strip()
    if role_text and upstream_text:
        return f"{upstream_text}\n\n{role_text}"
    return role_text or upstream_text


def _clean_context_for_user_text(context_text: str) -> str:
    # Strip embedded user-query blocks and stale answer suffixes.
    cleaned = context_text.strip()

    # Remove everything after the first ---User Query--- marker.
    cleaned = re.split(
        r"(?im)^\s*---User Query---\s*$",
        cleaned,
        maxsplit=1,
    )[0].strip()

    # Remove stale user question/query lines that may be embedded in context.
    cleaned = re.sub(
        r"(?im)^\s*User (?:Question|Query)\s*:\s*.*$",
        "",
        cleaned,
    )

    # Remove both legacy and current suffix templates.
    cleaned = re.sub(
        r"(?im)^\s*Please answer based on the context(?: and images)? provided\.?\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*Please answer based on the provided context\.?\s*$",
        "",
        cleaned,
    )
    return cleaned.strip()


def build_answer_suffix(has_images: bool) -> str:
    return IMAGE_CONTEXT_SUFFIX if has_images else TEXT_CONTEXT_SUFFIX


def repack_query_messages(
    system_prompt: Any,
    prompt: Any,
    *,
    upstream_system: str = "",
    has_images: bool = False,
) -> Optional[Tuple[str, str]]:
    """
    Repack LightRAG prompt into a consistent system/user message pair.

    Returns None when prompt layout is not suitable for repacking.
    """
    raw_system = str(system_prompt or "").strip()
    if "---Context---" not in raw_system:
        return None

    role_prefix, context_segment = _extract_last_context_segment(raw_system)
    if not role_prefix or not context_segment:
        return None

    final_system = _compose_final_system(role_prefix, str(upstream_system or ""))
    cleaned_context = _clean_context_for_user_text(context_segment)
    if not cleaned_context:
        return None

    question = str(prompt or "").strip()
    if question:
        final_user = (
            f"{cleaned_context}\n\n"
            f"User Question: {question}\n\n"
            f"{build_answer_suffix(has_images)}"
        )
    else:
        final_user = cleaned_context
    return final_system, final_user
