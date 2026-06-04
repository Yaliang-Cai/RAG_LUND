// Block-level fuzzy matching for "jump to source" highlighting.
//
// The chunk text from LightRAG and the rendered markdown DOM live in different
// "shapes": the chunk is whitespace-collapsed prose, while the DOM keeps source
// newlines and splits text across inline elements (<strong>, <a>, katex spans).
// Comparing a normalized needle against raw, per-node text fails most of the
// time. Instead we normalize BOTH sides and match at block granularity, using
// each block's full textContent (which stitches inline fragments back together).

const BLOCK_SELECTOR = 'p, li, td, th, h1, h2, h3, h4, h5, h6, blockquote, pre'

const NEEDLE_LENGTH = 60

/** Collapse whitespace and drop markdown punctuation that may survive into rendered text. */
export function normalizeForMatch(s: string): string {
  return s
    .replace(/[`*_~#>[\]()]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Find the block element whose text contains the start of `rawText`.
 * Returns null when the text is empty or no block matches.
 */
export function findHighlightBlock(root: HTMLElement, rawText: string): HTMLElement | null {
  const needle = normalizeForMatch(rawText).slice(0, NEEDLE_LENGTH).trim()
  if (!needle) return null

  const blocks = root.querySelectorAll<HTMLElement>(BLOCK_SELECTOR)
  for (const block of blocks) {
    if (normalizeForMatch(block.textContent ?? '').includes(needle)) return block
  }
  return null
}
