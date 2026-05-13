from __future__ import annotations

import json
import re
from typing import Any


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def load_json_object(raw: Any) -> dict[str, Any]:
    """Parse a JSON object from normal or lightly wrapped LLM output."""
    if isinstance(raw, dict):
        return raw
    text = raw if isinstance(raw, str) else str(raw)
    text = text.strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _loads_wrapped_object(text)
    if not isinstance(parsed, dict):
        raise TypeError("expected JSON object")
    return parsed


def _loads_wrapped_object(text: str) -> Any:
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        return json.loads(fenced.group(1).strip())

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no JSON object found", text, 0)
