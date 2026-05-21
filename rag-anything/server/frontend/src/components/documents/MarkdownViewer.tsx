import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import 'highlight.js/styles/github-dark.css'
import 'katex/dist/katex.min.css'

interface MarkdownViewerProps {
  content: string
  scrollToText?: string | null
  onScrollComplete?: () => void
}

// Trim leading/trailing whitespace + punctuation that often differ between
// the chunk content and the rendered DOM, then keep the first 80 chars as
// the substring to look for.
function buildNeedle(raw: string): string {
  // Strip markdown formatting characters (#, *, _, `, ~, >, brackets, parens)
  // so the substring search works against rendered DOM text where those are gone.
  const stripped = raw
    .replace(/[`*_~#>[\]()]/g, '')
    .replace(/\s+/g, ' ')
    .replace(/^[\s\p{P}]+|[\s\p{P}]+$/gu, '')
    .trim()
  return stripped.slice(0, 60).trim()
}

function findTextNode(root: HTMLElement, needle: string): { node: Text; offset: number } | null {
  if (!needle) return null
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  let node: Node | null = walker.nextNode()
  while (node) {
    const text = node.nodeValue ?? ''
    const idx = text.indexOf(needle)
    if (idx !== -1) return { node: node as Text, offset: idx }
    node = walker.nextNode()
  }
  return null
}

export function MarkdownViewer({ content, scrollToText, onScrollComplete }: MarkdownViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!scrollToText) return
    // Wait until markdown has actually been rendered before attempting to find
    // the needle. Otherwise the first run (with empty content) fails and we
    // clear pendingChunkText prematurely.
    if (!content) return
    const root = containerRef.current
    if (!root) return

    // Wait one frame for ReactMarkdown to render the new content.
    const raf = requestAnimationFrame(() => {
      const needle = buildNeedle(scrollToText)
      const hit = findTextNode(root, needle)
      if (hit) {
        const range = document.createRange()
        range.setStart(hit.node, hit.offset)
        range.setEnd(hit.node, hit.offset + needle.length)
        const rect = range.getBoundingClientRect()
        const rootRect = root.getBoundingClientRect()
        root.scrollTop += rect.top - rootRect.top - 80

        const mark = document.createElement('mark')
        mark.className = 'bg-yellow-300/40 rounded px-0.5 transition-opacity duration-1000'
        try {
          range.surroundContents(mark)
          setTimeout(() => {
            mark.style.opacity = '0'
            setTimeout(() => {
              const parent = mark.parentNode
              if (parent) {
                while (mark.firstChild) parent.insertBefore(mark.firstChild, mark)
                parent.removeChild(mark)
              }
            }, 1100)
          }, 1500)
        } catch {
          // surroundContents fails if the range crosses element boundaries —
          // accept the scroll-only behavior in that case.
        }
      }
      onScrollComplete?.()
    })

    return () => cancelAnimationFrame(raf)
  }, [scrollToText, content, onScrollComplete])

  return (
    <div
      ref={containerRef}
      className="prose prose-invert prose-sm max-w-none p-4 overflow-y-auto h-full text-foreground"
    >
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
