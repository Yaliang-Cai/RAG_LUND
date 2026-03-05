#!/usr/bin/env python3
"""
Download all frontend static assets for fully offline operation.

Run **once** while the machine has internet access:

    python server/download_static.py

After this, start the server normally:

    uvicorn server.app:app --host 0.0.0.0 --port 9621

The server detects that local assets exist and serves them instead of CDN URLs.
"""

import re
import sys
import urllib.request
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Asset manifest
# ---------------------------------------------------------------------------
ASSETS = [
    # marked.js
    (
        "https://cdn.bootcdn.net/ajax/libs/marked/4.3.0/marked.min.js",
        "marked.min.js",
    ),
    # KaTeX
    (
        "https://cdn.bootcdn.net/ajax/libs/KaTeX/0.16.9/katex.min.css",
        "katex/katex.min.css",
    ),
    (
        "https://cdn.bootcdn.net/ajax/libs/KaTeX/0.16.9/katex.min.js",
        "katex/katex.min.js",
    ),
    (
        "https://cdn.bootcdn.net/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js",
        "katex/auto-render.min.js",
    ),
    # highlight.js
    (
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js",
        "hljs/highlight.min.js",
    ),
    (
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css",
        "hljs/github-dark.min.css",
    ),
    (
        "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css",
        "hljs/github.min.css",
    ),
]

KATEX_FONT_BASE = "https://cdn.bootcdn.net/ajax/libs/KaTeX/0.16.9/fonts/"


def _download(url: str, dest: Path, force: bool = False) -> bool:
    """Download url → dest. Returns True if downloaded, False if skipped."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  skip  {dest.relative_to(STATIC_DIR)}  (already exists)")
        return False
    print(f"  get   {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}", file=sys.stderr)
        return False


def main(force: bool = False) -> None:
    print(f"Target directory: {STATIC_DIR}\n")

    # Step 1 – download JS/CSS
    for url, rel_path in ASSETS:
        _download(url, STATIC_DIR / rel_path, force=force)

    # Step 2 – download KaTeX fonts (parsed from the CSS)
    css_path = STATIC_DIR / "katex" / "katex.min.css"
    if not css_path.exists():
        print("\nERROR: KaTeX CSS not downloaded; cannot resolve font URLs.", file=sys.stderr)
        sys.exit(1)

    font_files = sorted(set(re.findall(r"url\(fonts/([^)]+)\)", css_path.read_text())))
    print(f"\nDownloading {len(font_files)} KaTeX font files …")
    for font_file in font_files:
        _download(
            KATEX_FONT_BASE + font_file,
            STATIC_DIR / "katex" / "fonts" / font_file,
            force=force,
        )

    # Step 3 – verify sentinel files exist
    sentinels = [
        STATIC_DIR / "marked.min.js",
        STATIC_DIR / "katex" / "katex.min.js",
        STATIC_DIR / "hljs" / "highlight.min.js",
    ]
    missing = [s for s in sentinels if not s.exists()]
    if missing:
        print("\nERROR: some sentinel files are missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll assets ready in {STATIC_DIR}")
    print("Restart the server — it will automatically use local files.")


if __name__ == "__main__":
    force = "--force" in sys.argv or "-f" in sys.argv
    if force:
        print("--force: re-downloading all assets even if they exist\n")
    main(force=force)
