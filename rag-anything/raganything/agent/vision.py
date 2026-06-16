"""Generation-time VLM helper: send interleaved text + images to a vision model.

Single responsibility: turn a text prompt + a list of image file paths into a
multimodal message list and call the injected vision function. Mirrors the message
shape used by ``query.py``'s ``aquery_vlm_enhanced`` so it drives the same VLM
endpoint. "Directed" viewing is enforced upstream: ``pack_context`` only collects
images when ``visual_intent`` is set and the owning chunk is relevant (supports a
fact or clears the score floor)."""
from __future__ import annotations

import logging
from typing import Any, Callable

from raganything.utils import encode_image_to_base64, validate_image_file

logger = logging.getLogger(__name__)

MAX_VLM_IMAGES = 6


def build_multimodal_messages(
    prompt: str, image_paths: list[str], *, max_images: int = MAX_VLM_IMAGES,
) -> tuple[list[dict], int]:
    """Build an OpenAI-style multimodal message list: one text part followed by the
    validated / base64-encoded image parts. Returns ``(messages, n_images_encoded)``.
    Invalid or unreadable images are skipped; if none survive, the message is plain
    text so callers degrade gracefully to a text answer."""
    parts: list[dict] = [{"type": "text", "text": prompt}]
    n = 0
    for path in image_paths:
        if n >= max_images:
            break
        p = str(path).strip()
        if not p or not validate_image_file(p):
            continue
        try:
            b64 = encode_image_to_base64(p)
        except Exception:
            logger.warning("VLM image encode failed: %s", p, exc_info=True)
            b64 = ""
        if not b64:
            continue
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        n += 1
    content: Any = parts if n else prompt
    return [{"role": "user", "content": content}], n


async def vlm_generate(
    vision_fn: Callable, prompt: str, image_paths: list[str], *,
    max_images: int = MAX_VLM_IMAGES,
) -> str:
    """Call the vision function with text+images. Matches ``query.py``'s convention
    of ``vision_fn("", messages=messages)``."""
    messages, _ = build_multimodal_messages(prompt, image_paths, max_images=max_images)
    return str(await vision_fn("", messages=messages))
