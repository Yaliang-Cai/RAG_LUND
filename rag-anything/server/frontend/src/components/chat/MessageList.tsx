import { useEffect, useRef, useState } from 'react'
import { MessageBubble } from './MessageBubble'
import type { Message } from './MessageBubble'

interface MessageListProps {
  messages: Message[]
  streamingAnswer: string
  streamingReasoning: string
  isStreaming: boolean
  workspaceId?: string
}

export function MessageList({
  messages,
  streamingAnswer,
  streamingReasoning,
  isStreaming,
  workspaceId,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (autoScroll) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingAnswer, autoScroll])

  function handleScroll() {
    const el = containerRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60
    setAutoScroll(atBottom)
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4"
      onScroll={handleScroll}
    >
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} workspaceId={workspaceId} />
      ))}
      {isStreaming && (
        <MessageBubble
          message={{
            id: '__streaming__',
            role: 'assistant',
            content: streamingAnswer || '_Thinking…_',
            reasoning: streamingReasoning,
          }}
        />
      )}
      <div ref={bottomRef} />
    </div>
  )
}
