import { Children, isValidElement, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { AgenticTrace } from './AgenticTrace'
import { InlineCitation } from './InlineCitation'
import { ReasoningTrace } from './ReasoningTrace'
import { ReferenceList } from './ReferenceList'
import { cn } from '@/lib/utils'
import type { SourceNode, TraceType, ChunkRef } from '@/types'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sourceNodes?: SourceNode[]
  traceType?: TraceType
  traceMetadata?: Record<string, unknown>
  chunks?: ChunkRef[]
  images?: string[]   // blob URLs for user-attached images (multimodal queries)
  query?: string   // original user question (kept for future eval/review use)
}

interface MessageBubbleProps {
  message: Message
  workspaceId?: string
}

const EMPTY_PLACEHOLDER =
  'No relevant information was found in the knowledge base for this query.'

/**
 * Build a map for both citation key variants the LLM may emit:
 *   - by chunk.id        (the new format, e.g. "DC1" — see
 *     _INLINE_CITATION_INSTRUCTION in services/local_rag.py)
 *   - by reference_id    (numeric fallback)
 *   - by 1-based index   (very old answers)
 */
function buildChunkMap(chunks?: ChunkRef[]): Map<string, ChunkRef> {
  const map = new Map<string, ChunkRef>()
  if (!chunks) return map
  chunks.forEach((c, idx) => {
    const keys: string[] = []
    if (c.id != null) keys.push(String(c.id))
    if (c.reference_id != null) keys.push(String(c.reference_id))
    keys.push(String(idx + 1))
    for (const k of keys) if (!map.has(k)) map.set(k, c)
  })
  return map
}

// Citation tokens: pure digits ([1], [12]) or upper-case prefix + digits ([DC1], [E12]).
// Avoids false matches on free-text brackets like [note] or [TODO].
const CITATION_RE = /\[([A-Z]{0,4}\d{1,4})\]/g

/** Replace [N] tokens in a string with InlineCitation components. */
function injectCitations(text: string, chunkMap: Map<string, ChunkRef>): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  CITATION_RE.lastIndex = 0
  while ((m = CITATION_RE.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const refId = m[1]
    out.push(
      <InlineCitation
        key={`cite-${m.index}-${refId}`}
        refId={refId}
        chunk={chunkMap.get(refId)}
      />,
    )
    last = CITATION_RE.lastIndex
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

/** Recursively walk ReactNode children, replacing [N] in string leaves. */
function processChildren(
  children: ReactNode,
  chunkMap: Map<string, ChunkRef>,
): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child === 'string') return injectCitations(child, chunkMap)
    if (isValidElement(child)) {
      const el = child as React.ReactElement<{ children?: ReactNode }>
      if (el.props?.children !== undefined) {
        const next = processChildren(el.props.children, chunkMap)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return { ...el, props: { ...(el.props as any), children: next } }
      }
    }
    return child
  })
}

export function MessageBubble({ message, workspaceId: _workspaceId }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const chunkMap = buildChunkMap(message.chunks)
  const hasChunks = chunkMap.size > 0
  const content = !isUser && !message.content.trim() ? EMPTY_PLACEHOLDER : message.content

  // Markdown component overrides — only wrap text-bearing inline parents.
  // Code blocks and links are left alone so [N] inside code stays literal.
  const mdComponents = hasChunks
    ? {
        p: ({ children, ...rest }: { children?: ReactNode }) => (
          <p {...rest}>{processChildren(children, chunkMap)}</p>
        ),
        li: ({ children, ...rest }: { children?: ReactNode }) => (
          <li {...rest}>{processChildren(children, chunkMap)}</li>
        ),
        td: ({ children, ...rest }: { children?: ReactNode }) => (
          <td {...rest}>{processChildren(children, chunkMap)}</td>
        ),
      }
    : undefined

  return (
    <div className={cn('flex flex-col gap-1', isUser ? 'items-end' : 'items-start')}>
      {isUser && message.images && message.images.length > 0 && (
        <div className="flex flex-wrap gap-2 max-w-[80%] justify-end">
          {message.images.map((src, i) => (
            <img
              key={i}
              src={src}
              alt={`attached ${i + 1}`}
              className="h-24 max-w-[160px] rounded-lg border border-border object-cover"
            />
          ))}
        </div>
      )}
      <div
        className={cn(
          'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-secondary text-secondary-foreground'
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <ReactMarkdown
            rehypePlugins={[rehypeHighlight]}
            components={mdComponents}
          >
            {content}
          </ReactMarkdown>
        )}
      </div>
      {!isUser && message.reasoning && <ReasoningTrace text={message.reasoning} />}
      {!isUser && <ReferenceList messageId={message.id} chunks={message.chunks} />}
      {!isUser && message.traceType === 'agentic' && (
        <AgenticTrace
          traceType={message.traceType}
          metadata={message.traceMetadata ?? {}}
        />
      )}
    </div>
  )
}
