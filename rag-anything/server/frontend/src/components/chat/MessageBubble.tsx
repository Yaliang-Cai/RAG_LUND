import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { ReasoningTrace } from './ReasoningTrace'
import { CitationChip } from './CitationChip'
import { cn } from '@/lib/utils'
import type { SourceNode } from '@/types'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sourceNodes?: SourceNode[]
}

export function MessageBubble({ message }: { message: Message }) {
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
      {!isUser && message.reasoning && <ReasoningTrace text={message.reasoning} />}
      {!isUser && message.sourceNodes && message.sourceNodes.length > 0 && (
        <div className="flex flex-wrap gap-1 max-w-[80%]">
          {message.sourceNodes.map((n, i) => <CitationChip key={i} node={n} />)}
        </div>
      )}
    </div>
  )
}
