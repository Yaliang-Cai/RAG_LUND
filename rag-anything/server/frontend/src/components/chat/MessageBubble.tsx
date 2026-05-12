import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { AgenticTrace } from './AgenticTrace'
import { EvalBadge } from './EvalBadge'
import { CitationChip } from './CitationChip'
import { cn } from '@/lib/utils'
import type { SourceNode, TraceType } from '@/types'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sourceNodes?: SourceNode[]
  traceType?: TraceType
  traceMetadata?: Record<string, unknown>
  query?: string   // original user question (set on assistant messages for EvalBadge)
}

interface MessageBubbleProps {
  message: Message
  workspaceId?: string
}

export function MessageBubble({ message, workspaceId }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  return (
    <div className={cn('flex flex-col gap-1', isUser ? 'items-end' : 'items-start')}>
      <div
        className={cn(
          'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-secondary text-secondary-foreground'
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{message.content}</ReactMarkdown>
        )}
      </div>
      {!isUser && message.traceType && (
        <AgenticTrace
          traceType={message.traceType}
          metadata={message.traceMetadata ?? {}}
        />
      )}
      {!isUser && message.query && message.content && workspaceId && (
        <EvalBadge
          workspaceId={workspaceId}
          query={message.query}
          answer={message.content}
        />
      )}
      {!isUser && message.sourceNodes && message.sourceNodes.length > 0 && (
        <div className="flex flex-wrap gap-1 max-w-[80%]">
          {message.sourceNodes.map((n, i) => <CitationChip key={i} node={n} />)}
        </div>
      )}
    </div>
  )
}
