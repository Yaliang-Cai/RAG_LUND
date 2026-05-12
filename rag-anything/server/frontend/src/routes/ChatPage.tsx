import { useState, useCallback, useRef } from 'react'
import { useStreamQuery } from '@/hooks/useStreamQuery'
import { useAppStore } from '@/store'
import { MessageList } from '@/components/chat/MessageList'
import { ChatInput } from '@/components/chat/ChatInput'
import { Button } from '@/components/ui/button'
import { SquarePen } from 'lucide-react'
import type { Message } from '@/components/chat/MessageBubble'
import type { SourceNode, TraceType, QueryParams } from '@/types'

let msgId = 0

function modeToTraceType(mode: string): TraceType {
  if (mode === 'agentic') return 'agentic'
  if (mode === 'ppr') return 'ppr'
  if (mode === 'auto') return 'auto'
  return null
}

export default function ChatPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { send, answer, reasoning, status, sourceNodes, metadata } = useStreamQuery()
  const [messages, setMessages] = useState<Message[]>([])
  const latestRef = useRef<{
    answer: string
    reasoning: string
    sourceNodes: SourceNode[]
    metadata: Record<string, unknown>
  }>({ answer: '', reasoning: '', sourceNodes: [], metadata: {} })

  latestRef.current = { answer, reasoning, sourceNodes, metadata }

  const handleSend = useCallback(async (query: string, mode: string, profile: string) => {
    const userMsg: Message = { id: String(++msgId), role: 'user', content: query }
    setMessages((prev) => [...prev, userMsg])

    await send({
      workspace_id: workspaceId,
      query,
      mode: mode as QueryParams['mode'],
      profile: mode === 'auto' && profile ? profile : undefined,
    })

    const { answer: a, reasoning: r, sourceNodes: sn, metadata: m } = latestRef.current
    const traceType = modeToTraceType(mode)
    setMessages((prev) => [
      ...prev,
      {
        id: String(++msgId),
        role: 'assistant',
        content: a,
        reasoning: r,
        sourceNodes: sn,
        traceType,
        traceMetadata: m,
      },
    ])
  }, [workspaceId, send])

  const isStreaming = status === 'streaming'

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <span className="text-xs text-muted-foreground">workspace: {workspaceId}</span>
        <Button variant="ghost" size="icon" onClick={() => setMessages([])} title="New conversation">
          <SquarePen className="h-4 w-4" />
        </Button>
      </div>
      <MessageList
        messages={messages}
        streamingAnswer={answer}
        streamingReasoning={reasoning}
        isStreaming={isStreaming}
      />
      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </div>
  )
}
