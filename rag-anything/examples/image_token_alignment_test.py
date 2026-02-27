#!/usr/bin/env python3
"""
Compare local image-token estimate vs vLLM prompt-token delta.

This script only tests image token accounting:
1) local estimate uses the same formula as lightrag/operate.py
2) vLLM side uses OpenAI-compatible prompt_tokens delta:
   image_tokens_measured = prompt_tokens(with_image) - prompt_tokens(text_only)
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from PIL import Image


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# =========================
# Local defaults (edit here)
# =========================
# Usage:
#   1) Fill this block once on your Linux machine.
#   2) Run: python image_token_alignment_test.py
# CLI args still override these values when provided.
LOCAL_DEFAULTS = {
    "api_base": "http://127.0.0.1:8001/v1",
    "api_key": "EMPTY",
    "model": "OpenGVLab/InternVL2-26B-AWQ",
    # Example:
    # "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/DocBench/docbench_results/mineru_outputs/docbench_0/P19-1598/hybrid_auto/images"
    "images_dir": "",
    "limit": 20,
    "prompt": "Answer briefly.",
    "max_tokens": 8,
    # InternVL2-26B-AWQ defaults
    "image_size": 448,
    "patch_size": 14,
    "downsample_ratio": 0.5,
    "min_dynamic_patch": 1,
    "max_dynamic_patch": 12,
    "use_thumbnail": "true",
    # Usually 2 means wrapper tokens like <IMG_START>/<IMG_END>
    "wrapper_tokens": 2,
}


def _bool_from_str(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _guess_mime(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".bmp":
        return "image/bmp"
    return "image/jpeg"


def _encode_image_to_data_url(image_path: Path) -> str:
    mime = _guess_mime(image_path)
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _estimate_dynamic_patch_count(
    width: int,
    height: int,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    use_thumbnail: bool,
) -> int:
    # Keep this logic aligned with lightrag/lightrag/operate.py
    if width <= 0 or height <= 0:
        return 1

    image_ratio = width / height
    candidate_ratios: list[tuple[int, int]] = []
    for n in range(min_dynamic_patch, max_dynamic_patch + 1):
        for i in range(1, n + 1):
            if n % i != 0:
                continue
            j = n // i
            candidate_ratios.append((i, j))

    if not candidate_ratios:
        return 1

    candidate_ratios = sorted(set(candidate_ratios), key=lambda x: x[0] * x[1])
    best = min(candidate_ratios, key=lambda x: abs(image_ratio - (x[0] / x[1])))
    blocks = best[0] * best[1]
    if use_thumbnail and blocks != 1:
        blocks += 1
    return blocks


def estimate_image_tokens_local(
    image_path: Path,
    image_size: int,
    patch_size: int,
    downsample_ratio: float,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    use_thumbnail: bool,
    wrapper_tokens: int,
) -> tuple[int, int, int]:
    with Image.open(image_path) as img:
        width, height = img.size

    per_patch_tokens = int(((image_size // patch_size) ** 2) * (downsample_ratio**2))
    if per_patch_tokens <= 0:
        raise ValueError("per_patch_tokens <= 0, check image/patch/downsample settings")

    patches = _estimate_dynamic_patch_count(
        width=width,
        height=height,
        min_dynamic_patch=min_dynamic_patch,
        max_dynamic_patch=max_dynamic_patch,
        use_thumbnail=use_thumbnail,
    )
    total = patches * per_patch_tokens + wrapper_tokens
    return total, patches, per_patch_tokens


def _prompt_tokens_from_resp(resp) -> int:
    usage = getattr(resp, "usage", None)
    if usage is None:
        raise RuntimeError("Response does not contain usage. Ensure vLLM returns usage.")

    if hasattr(usage, "prompt_tokens"):
        return int(usage.prompt_tokens)
    if isinstance(usage, dict) and "prompt_tokens" in usage:
        return int(usage["prompt_tokens"])
    raise RuntimeError("Cannot read usage.prompt_tokens from response.")


def measure_prompt_tokens_delta(
    client: OpenAI,
    model: str,
    text_prompt: str,
    image_data_url: str,
    max_tokens: int,
) -> tuple[int, int, int]:
    text_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text_prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    text_prompt_tokens = _prompt_tokens_from_resp(text_resp)

    image_resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    image_prompt_tokens = _prompt_tokens_from_resp(image_resp)
    image_delta = image_prompt_tokens - text_prompt_tokens
    return text_prompt_tokens, image_prompt_tokens, image_delta


def _collect_images(images: list[str], images_dir: str, limit: int) -> list[Path]:
    paths: list[Path] = []
    for item in images:
        p = Path(item).expanduser().resolve()
        if p.is_file():
            paths.append(p)

    if images_dir:
        root = Path(images_dir).expanduser().resolve()
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    paths.append(p)

    # de-duplicate while preserving order
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)

    if limit > 0:
        dedup = dedup[:limit]
    return dedup


def _iter_rows(paths: Iterable[Path]) -> Iterable[tuple[int, Path]]:
    for idx, path in enumerate(paths, start=1):
        yield idx, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local image token estimate with vLLM prompt token delta."
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("VISION_VLLM_API_BASE", LOCAL_DEFAULTS["api_base"]),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VISION_VLLM_API_KEY", LOCAL_DEFAULTS["api_key"]),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VISION_MODEL_NAME", LOCAL_DEFAULTS["model"]),
    )
    parser.add_argument("--prompt", default=LOCAL_DEFAULTS["prompt"])
    parser.add_argument("--max-tokens", type=int, default=LOCAL_DEFAULTS["max_tokens"])

    parser.add_argument("--image", action="append", default=[], help="Single image path; can be repeated.")
    parser.add_argument(
        "--images-dir",
        default=LOCAL_DEFAULTS["images_dir"],
        help="Folder to recursively collect images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LOCAL_DEFAULTS["limit"],
        help="Max number of images to test (<=0 means all).",
    )

    # Keep defaults aligned with your current InternVL2 settings.
    parser.add_argument("--image-size", type=int, default=LOCAL_DEFAULTS["image_size"])
    parser.add_argument("--patch-size", type=int, default=LOCAL_DEFAULTS["patch_size"])
    parser.add_argument(
        "--downsample-ratio",
        type=float,
        default=LOCAL_DEFAULTS["downsample_ratio"],
    )
    parser.add_argument(
        "--min-dynamic-patch",
        type=int,
        default=LOCAL_DEFAULTS["min_dynamic_patch"],
    )
    parser.add_argument(
        "--max-dynamic-patch",
        type=int,
        default=LOCAL_DEFAULTS["max_dynamic_patch"],
    )
    parser.add_argument("--use-thumbnail", type=str, default=LOCAL_DEFAULTS["use_thumbnail"])
    parser.add_argument("--wrapper-tokens", type=int, default=LOCAL_DEFAULTS["wrapper_tokens"])

    args = parser.parse_args()

    if not args.model:
        raise SystemExit("Missing --model (or set VISION_MODEL_NAME).")

    image_paths = _collect_images(args.image, args.images_dir, args.limit)
    if not image_paths:
        raise SystemExit("No images found. Use --image or --images-dir.")

    use_thumbnail = _bool_from_str(args.use_thumbnail)
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    print("=== Image Token Alignment Test ===")
    print(f"api_base={args.api_base}")
    print(f"model={args.model}")
    print(
        "local_formula="
        f"((image_size//patch_size)^2 * downsample_ratio^2) with dynamic patches + wrapper_tokens"
    )
    print(
        f"params: image_size={args.image_size}, patch_size={args.patch_size}, "
        f"downsample_ratio={args.downsample_ratio}, min/max_patch={args.min_dynamic_patch}/{args.max_dynamic_patch}, "
        f"use_thumbnail={use_thumbnail}, wrapper_tokens={args.wrapper_tokens}"
    )
    print()

    total_abs_diff = 0
    total_local = 0
    total_measured = 0

    for idx, image_path in _iter_rows(image_paths):
        data_url = _encode_image_to_data_url(image_path)
        local_tokens, patches, per_patch_tokens = estimate_image_tokens_local(
            image_path=image_path,
            image_size=args.image_size,
            patch_size=args.patch_size,
            downsample_ratio=args.downsample_ratio,
            min_dynamic_patch=args.min_dynamic_patch,
            max_dynamic_patch=args.max_dynamic_patch,
            use_thumbnail=use_thumbnail,
            wrapper_tokens=args.wrapper_tokens,
        )
        text_pt, image_pt, measured_delta = measure_prompt_tokens_delta(
            client=client,
            model=args.model,
            text_prompt=args.prompt,
            image_data_url=data_url,
            max_tokens=args.max_tokens,
        )

        diff = measured_delta - local_tokens
        total_abs_diff += abs(diff)
        total_local += local_tokens
        total_measured += measured_delta

        print(
            f"[{idx}] {image_path.name}\n"
            f"  text_prompt_tokens={text_pt}, image_prompt_tokens={image_pt}, measured_image_delta={measured_delta}\n"
            f"  local_estimate={local_tokens} (patches={patches}, per_patch={per_patch_tokens})\n"
            f"  diff(measured-local)={diff}\n"
        )

    n = max(1, len(image_paths))
    print("=== Summary ===")
    print(f"images_tested={len(image_paths)}")
    print(f"sum_local_estimate={total_local}")
    print(f"sum_measured_delta={total_measured}")
    print(f"mean_abs_diff={total_abs_diff / n:.2f}")
    print(
        "Tip: if diff is consistently positive/negative, adjust wrapper_tokens or check vLLM mm_processor kwargs."
    )


if __name__ == "__main__":
    main()
