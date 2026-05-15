import { useCallback, useRef, useState } from 'react'
import { useStreamQuery } from '@/hooks/useStreamQuery'
import { useAppStore } from '@/store'
import { MessageList } from '@/components/chat/MessageList'
import { ChatInput } from '@/components/chat/ChatInput'
import { Button } from '@/components/ui/button'
import { SquarePen } from 'lucide-react'
import { postMultimodalQuery } from '@/api/query'
import type { Message } from '@/components/chat/MessageBubble'
import type { TraceType, QueryParams, ChunkRef } from '@/types'

let msgId = 0

const MAX_HISTORY_TURNS = 5

function modeToTraceType(mode: string): TraceType {
  // Trace panels are only meaningful for agentic mode; other modes hide it.
  return mode === 'agentic' ? 'agentic' : null
}

function extractChunks(metadata: Record<string, unknown>): ChunkRef[] {
  const data = (metadata?.data ?? {}) as Record<string, unknown>
  const raw = data?.chunks
  return Array.isArray(raw) ? (raw as ChunkRef[]) : []
}

function buildHistory(messages: Message[]): { role: string; content: string }[] {
  return messages
    .slice(-MAX_HISTORY_TURNS * 2)
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .map((m) => ({ role: m.role, content: m.content }))
}

export default function ChatPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const messages = useAppStore((s) => s.chatMessages)
  const setMessages = useAppStore((s) => s.setChatMessages)
  const clearMessages = useAppStore((s) => s.clearChatMessages)

  const { send, answer, reasoning, status } = useStreamQuery()

  // Non-streaming (multimodal) busy flag — separate from streaming status.
  const [multimodalBusy, setMultimodalBusy] = useState(false)

  const messagesRef = useRef<Message[]>([])
  messagesRef.current = messages

  const handleSend = useCallback(async (
    query: string,
    mode: string,
    profile: string,
    images: File[],
  ) => {
    const userMsg: Message = { id: String(++msgId), role: 'user', content: query }
    setMessages((prev) => [...prev, userMsg])

    // ── Branch A: multimodal (images attached) → non-streaming ──
    if (images.length > 0) {
      setMultimodalBusy(true)
      try {
        const result = await postMultimodalQuery({
          workspace_id: workspaceId,
          query,
          images,
          mode,
        })
        setMessages((prev) => [
          ...prev,
          {
            id: String(++msgId),
            role: 'assistant',
            content: result.answer,
            reasoning: '',
            sourceNodes: [],
            traceType: null,
            traceMetadata: { image_count: result.image_count },
            query,
          },
        ])
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: String(++msgId),
            role: 'assistant',
            content: `Multimodal query failed: ${(err as Error).message}`,
            reasoning: '',
            sourceNodes: [],
            traceType: null,
            traceMetadata: {},
            query,
          },
        ])
      } finally {
        setMultimodalBusy(false)
      }
      return
    }

    // ── Branch B: text-only → streaming ──
    const history = buildHistory(messagesRef.current)

    // vlm_enhanced is intentionally NOT sent from the frontend:
    // backend default (DEFAULT_VLM_ENHANCED=True) lets RAGAnything auto-route,
    // and aquery_vlm_enhanced already falls back to text when chunks have no images.
    // send() returns a final snapshot accumulated inside the SSE loop, so we
    // don't depend on React having flushed setState calls before this awaits.
    const snap = await send({
      workspace_id: workspaceId,
      query,
      mode: mode as QueryParams['mode'],
      profile: mode === 'auto' && profile ? profile : undefined,
      conversation_history: history.length > 0 ? history : undefined,
    })

    const traceType = modeToTraceType(mode)
    setMessages((prev) => [
      ...prev,
      {
        id: String(++msgId),
        role: 'assistant',
        content: snap.answer,
        reasoning: snap.reasoning,
        sourceNodes: snap.sourceNodes,
        traceType,
        traceMetadata: snap.metadata,
        chunks: extractChunks(snap.metadata),
        query,
      },
    ])
  }, [workspaceId, send, setMessages])

  const isStreaming = status === 'streaming'
  const isBusy = isStreaming || multimodalBusy

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <span className="text-xs text-muted-foreground">workspace: {workspaceId}</span>
        <Button variant="ghost" size="icon" onClick={clearMessages} title="New conversation">
          <SquarePen className="h-4 w-4" />
        </Button>
      </div>
      <MessageList
        messages={messages}
        streamingAnswer={answer}
        streamingReasoning={reasoning}
        isStreaming={isStreaming}
        workspaceId={workspaceId}
      />
      <ChatInput onSend={handleSend} disabled={isBusy} />
    </div>
  )
}
