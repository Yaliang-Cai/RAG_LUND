import { describe, it, expect } from 'vitest'
import { findHighlightBlock } from '../textHighlight'

function makeRoot(html: string): HTMLElement {
  const root = document.createElement('div')
  root.innerHTML = html
  return root
}

describe('findHighlightBlock', () => {
  it('matches a paragraph whose rendered text wraps across a newline', () => {
    // Chunk text is whitespace-collapsed; the rendered DOM keeps the source
    // newline. The old per-node indexOf failed on this; block matching must not.
    const root = makeRoot('<p>The quick brown fox\njumps over the lazy dog.</p>')
    const block = findHighlightBlock(root, 'The quick brown fox jumps over the lazy dog.')
    expect(block).not.toBeNull()
    expect(block!.tagName).toBe('P')
  })

  it('matches across inline formatting that splits the text into multiple nodes', () => {
    // <strong> fragments the paragraph into 3 text nodes; no single node holds
    // the whole needle. Block-level textContent stitches them back together.
    const root = makeRoot('<p>The <strong>quick</strong> brown fox jumps.</p>')
    const block = findHighlightBlock(root, 'The quick brown fox jumps.')
    expect(block).not.toBeNull()
    expect(block!.tagName).toBe('P')
  })

  it('picks the correct block among several', () => {
    const root = makeRoot(
      '<p>First unrelated paragraph.</p>' +
        '<p>Second paragraph with the target phrase inside.</p>',
    )
    const block = findHighlightBlock(root, 'target phrase inside')
    expect(block).not.toBeNull()
    expect(block!.textContent).toContain('Second paragraph')
  })

  it('returns null when no block contains the text', () => {
    const root = makeRoot('<p>Nothing relevant here.</p>')
    expect(findHighlightBlock(root, 'completely different content')).toBeNull()
  })

  it('returns null for empty / whitespace-only chunk text', () => {
    const root = makeRoot('<p>Some content.</p>')
    expect(findHighlightBlock(root, '')).toBeNull()
    expect(findHighlightBlock(root, '   \n  ')).toBeNull()
  })
})
