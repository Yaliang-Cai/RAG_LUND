import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import 'highlight.js/styles/github-dark.css'
import 'katex/dist/katex.min.css'

export function MarkdownViewer({ content }: { content: string }) {
  return (
    <div className="prose prose-invert prose-sm max-w-none p-4 overflow-y-auto h-full text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
