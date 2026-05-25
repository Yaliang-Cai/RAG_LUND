import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import 'highlight.js/styles/github-dark.css'
import 'katex/dist/katex.min.css'
import { findHighlightBlock } from './textHighlight'

interface MarkdownViewerProps {
  content: string
  scrollToText?: string | null
  onScrollComplete?: () => void
}

export function MarkdownViewer({ content, scrollToText, onScrollComplete }: MarkdownViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!scrollToText) return
    // Wait until markdown has actually been rendered before attempting to match.
    // Otherwise the first run (with empty content) fails and we clear
    // pendingChunkText prematurely.
    if (!content) return
    const root = containerRef.current
    if (!root) return

    // Wait one frame for ReactMarkdown to render the new content.
    const raf = requestAnimationFrame(() => {
      const block = findHighlightBlock(root, scrollToText)
      if (block) {
        block.scrollIntoView({ behavior: 'smooth', block: 'center' })
        // Temporary block-level highlight that fades out.
        block.style.transition = 'background-color 1000ms ease'
        block.style.backgroundColor = 'rgba(253, 224, 71, 0.35)' // tailwind yellow-300 @ 35%
        setTimeout(() => {
          block.style.backgroundColor = ''
        }, 2000)
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
