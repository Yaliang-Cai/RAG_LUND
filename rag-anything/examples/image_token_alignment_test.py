#!/usr/bin/env python3
"""
Compare local Qwen-VL image-token estimate vs vLLM prompt-token delta.

Local estimate follows vLLM Qwen-VL logic:
  visual_tokens = (t * h * w) // (merge_size ** 2)
where (t, h, w) comes from AutoImageProcessor(...).image_grid_thw.
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from PIL import Image
from transformers import AutoImageProcessor


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

LOCAL_DEFAULTS = {
    "api_base": "http://127.0.0.1:8001/v1",
    "api_key": "EMPTY",
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "model_path": "/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8",
    "images_dir": "",
    "limit": 20,
    "prompt": "Answer briefly.",
    "max_tokens": 8,
    "wrapper_tokens": 2,
}


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


def estimate_image_tokens_qwen_local(
    image_path: Path,
    image_processor,
    wrapper_tokens: int,
) -> tuple[int, int, int, tuple[int, int, int]]:
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        processed = image_processor(images=image_rgb, return_tensors="np")

    image_grid_thw = processed.get("image_grid_thw")
    if image_grid_thw is None or len(image_grid_thw) <= 0:
        raise RuntimeError(f"image_grid_thw missing for {image_path}")

    t, h, w = [int(v) for v in image_grid_thw[0].tolist()]
    merge_size = max(1, int(getattr(image_processor, "merge_size", 2)))
    visual_tokens = (t * h * w) // (merge_size**2)
    if visual_tokens <= 0:
        raise RuntimeError(f"Invalid visual token estimate for {image_path}")

    total_tokens = visual_tokens + max(0, int(wrapper_tokens))
    return total_tokens, visual_tokens, merge_size, (t, h, w)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local Qwen-VL image token estimate with vLLM prompt token delta."
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
    parser.add_argument(
        "--model-path",
        default=os.getenv(
            "IMAGE_TOKEN_MODEL_NAME_OR_PATH",
            os.getenv("VISION_MODEL_PATH", LOCAL_DEFAULTS["model_path"]),
        ),
        help="Local model path or HF repo id used to load AutoImageProcessor.",
    )
    parser.add_argument("--prompt", default=LOCAL_DEFAULTS["prompt"])
    parser.add_argument("--max-tokens", type=int, default=LOCAL_DEFAULTS["max_tokens"])

    parser.add_argument(
        "--image", action="append", default=[], help="Single image path; can be repeated."
    )
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
    parser.add_argument(
        "--wrapper-tokens",
        type=int,
        default=LOCAL_DEFAULTS["wrapper_tokens"],
        help="Fixed wrapper reserve per image (e.g., vision_start/end).",
    )

    args = parser.parse_args()

    if not args.model:
        raise SystemExit("Missing --model (or set VISION_MODEL_NAME).")
    if not args.model_path:
        raise SystemExit(
            "Missing --model-path (or set IMAGE_TOKEN_MODEL_NAME_OR_PATH / VISION_MODEL_PATH)."
        )

    image_paths = _collect_images(args.image, args.images_dir, args.limit)
    if not image_paths:
        raise SystemExit("No images found. Use --image or --images-dir.")

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    print("=== Qwen Image Token Alignment Test ===")
    print(f"api_base={args.api_base}")
    print(f"model={args.model}")
    print(f"model_path={args.model_path}")
    print(
        "local_formula=(image_grid_thw[0].prod() // merge_size^2) + wrapper_tokens"
    )
    print(f"wrapper_tokens={args.wrapper_tokens}")
    print(
        "processor: "
        f"patch_size={getattr(image_processor, 'patch_size', 'NA')}, "
        f"merge_size={getattr(image_processor, 'merge_size', 'NA')}, "
        f"min_pixels={getattr(image_processor, 'min_pixels', 'NA')}, "
        f"max_pixels={getattr(image_processor, 'max_pixels', 'NA')}"
    )
    print()

    total_abs_diff = 0
    total_local = 0
    total_measured = 0

    for idx, image_path in _iter_rows(image_paths):
        data_url = _encode_image_to_data_url(image_path)
        local_tokens, visual_tokens, merge_size, grid_thw = estimate_image_tokens_qwen_local(
            image_path=image_path,
            image_processor=image_processor,
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
            f"  local_estimate={local_tokens} (visual_tokens={visual_tokens}, merge_size={merge_size}, grid_thw={grid_thw})\n"
            f"  diff(measured-local)={diff}\n"
        )

    n = max(1, len(image_paths))
    print("=== Summary ===")
    print(f"images_tested={len(image_paths)}")
    print(f"sum_local_estimate={total_local}")
    print(f"sum_measured_delta={total_measured}")
    print(f"mean_abs_diff={total_abs_diff / n:.2f}")


if __name__ == "__main__":
    main()
