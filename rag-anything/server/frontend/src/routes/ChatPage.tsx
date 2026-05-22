import { useCallback, useRef, useState } from 'react'
import { useStreamQuery } from '@/hooks/useStreamQuery'
import { useAppStore } from '@/store'
import { useQuerySettings } from '@/store/querySettings'
import { MessageList } from '@/components/chat/MessageList'
import { ChatInput } from '@/components/chat/ChatInput'
import { Button } from '@/components/ui/button'
import { SquarePen } from 'lucide-react'
import { postMultimodalQuery } from '@/api/query'
import type { Message } from '@/components/chat/MessageBubble'
import type { TraceType, QueryParams, ChunkRef } from '@/types'
import type { ModeKey } from '@/config/modePresets'

let msgId = 0

const MAX_HISTORY_TURNS = 5

function modeToTraceType(mode: ModeKey): TraceType {
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
    images: File[],
  ) => {
    const qs = useQuerySettings.getState()
    const mode = qs.mode
    const imagePreviews = images.length > 0 ? images.map((f) => URL.createObjectURL(f)) : undefined
    const userMsg: Message = {
      id: String(++msgId),
      role: 'user',
      content: query,
      ...(imagePreviews ? { images: imagePreviews } : {}),
    }
    setMessages((prev) => [...prev, userMsg])

    // ── Branch A: multimodal (images attached) → non-streaming ──
    if (images.length > 0) {
      setMultimodalBusy(true)
      try {
        const result = await postMultimodalQuery({
          workspace_id: workspaceId,
          query,
          images,
          mode: qs.mode === 'naive' ? 'naive' : 'mix',
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

    // Backend `mode`:
    //   naive  → 'naive'
    //   lightrag/multihop → 'auto' with profile locked
    //   agentic → 'agentic' (streaming branch in stream_query)
    // For agentic with explicit profile (not Auto), still send 'auto' so
    // the router locks the profile instead of going through the agent graph.
    let backendMode: QueryParams['mode']
    let profile: string | undefined
    if (mode === 'naive') {
      backendMode = 'naive'
    } else if (mode === 'agentic') {
      if (qs.agenticProfile && qs.agenticProfile !== 'auto') {
        backendMode = 'auto'
        profile = qs.agenticProfile
      } else {
        backendMode = 'agentic'
      }
    } else {
      // lightrag, multihop
      backendMode = 'auto'
      profile = mode === 'lightrag' ? 'semantic' : 'multihop'
    }

    const snap = await send({
      workspace_id: workspaceId,
      query,
      mode: backendMode,
      profile,
      top_k: qs.top_k,
      chunk_top_k: qs.chunk_top_k,
      enable_rerank: qs.enable_rerank,
      qdrant_retrieval_mode: qs.qdrant_retrieval_mode,
      rerank_candidate_cap: qs.rerank_candidate_cap,
      ppr_damping: qs.ppr_damping,
      ppr_top_k: qs.ppr_top_k,
      recognition_top_k: qs.recognition_top_k,
      linking_top_k: qs.linking_top_k,
      ppr_qa_top_k: qs.ppr_qa_top_k,
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
