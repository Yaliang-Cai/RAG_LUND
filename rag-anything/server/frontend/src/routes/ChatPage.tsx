import { useCallback, useRef, useState } from 'react'
import { useQuery } from '@/hooks/useQuery'
import { useAppStore } from '@/store'
import { useQuerySettings } from '@/store/querySettings'
import { MessageList } from '@/components/chat/MessageList'
import { ChatInput } from '@/components/chat/ChatInput'
import { Button } from '@/components/ui/button'
import { SquarePen, CircleStop } from 'lucide-react'
import { postMultimodalQuery } from '@/api/query'
import { postAgentChat, cancelAgent, AgentBusyError } from '@/api/agent'
import type { Message } from '@/components/chat/MessageBubble'
import type { TraceType, QueryParams, ChunkRef, AgentChatResponse } from '@/types'
import { MODE_PRESETS, type ModeKey } from '@/config/modePresets'

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

function newSessionId(): string {
  try {
    return crypto.randomUUID()
  } catch {
    return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  }
}

// Turn a v3 agent response into the assistant bubble text. The backend already
// appends unverifiable / partial-evidence disclosures to `answer`; here we only
// handle the no-answer cases (refusal / cancellation).
function agentContent(resp: AgentChatResponse): string {
  if (resp.cancelled) return '_(This answer was cancelled.)_'
  if (resp.answer && resp.answer.trim() !== '') return resp.answer
  const refusal = resp.refusal ?? {}
  const reason = String((refusal as { reason?: string }).reason ?? 'unanswered')
  const missing = (refusal as { missing_facts?: unknown[] }).missing_facts ?? []
  const attempts = (refusal as { attempts?: unknown[] }).attempts ?? []
  const lines = [`**Unable to answer** (${reason})`]
  if (Array.isArray(missing) && missing.length > 0) {
    lines.push('', 'Missing key facts:', ...missing.map((m) => `- ${String(m)}`))
  }
  if (Array.isArray(attempts) && attempts.length > 0) {
    lines.push('', `Retrieval attempts: ${attempts.map((a) => String(a)).join(' → ')}`)
  }
  return lines.join('\n')
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

  const { send, answer, status } = useQuery()

  // Multimodal busy flag — separate from text-query status so the input is
  // disabled during multimodal uploads even though they don't go through
  // useQuery().
  const [multimodalBusy, setMultimodalBusy] = useState(false)

  // v3 agent loop runs through /agent/chat (not useQuery). It needs its own
  // busy flag, a stable per-conversation session id, and an abort controller.
  const [agentBusy, setAgentBusy] = useState(false)
  const sessionIdRef = useRef<string>(newSessionId())
  const agentAbortRef = useRef<AbortController | null>(null)

  const messagesRef = useRef<Message[]>([])
  messagesRef.current = messages

  const newConversation = useCallback(() => {
    sessionIdRef.current = newSessionId()  // 新会话 → 新 session 记忆
    clearMessages()
  }, [clearMessages])

  const stopAgent = useCallback(() => {
    cancelAgent(workspaceId, sessionIdRef.current)
    agentAbortRef.current?.abort()
  }, [workspaceId])

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

    // ── Branch C: v3 agent loop → POST /agent/chat ──
    // The agent self-manages retrieval/budget and keeps its own session memory
    // server-side (coreference + cross-turn cache), so we send no history/knobs.
    if (MODE_PRESETS[mode].usesAgentEndpoint) {
      setAgentBusy(true)
      const controller = new AbortController()
      agentAbortRef.current = controller
      try {
        const resp = await postAgentChat(
          { workspace_id: workspaceId, session_id: sessionIdRef.current, query },
          controller.signal,
        )
        setMessages((prev) => [
          ...prev,
          {
            id: String(++msgId),
            role: 'assistant',
            content: agentContent(resp),
            reasoning: '',
            sourceNodes: [],
            traceType: 'agentv3',
            traceMetadata: { agent_v3: resp },
            chunks: (resp.chunks ?? []) as ChunkRef[],
            query,
          },
        ])
      } catch (err) {
        if ((err as Error).name === 'AbortError') return
        const busy = err instanceof AgentBusyError
        setMessages((prev) => [
          ...prev,
          {
            id: String(++msgId),
            role: 'assistant',
            content: busy
              ? `_This session already has a query running. Wait for it to finish or press Stop, then retry._`
              : `Agent query failed: ${(err as Error).message}`,
            reasoning: '',
            sourceNodes: [],
            traceType: null,
            traceMetadata: {},
            query,
          },
        ])
      } finally {
        setAgentBusy(false)
        agentAbortRef.current = null
      }
      return
    }

    // ── Branch B: text-only → POST /query (non-streaming) ──
    const history = buildHistory(messagesRef.current)

    // Backend `mode`:
    //   naive  → 'naive'
    //   lightrag/multihop → 'auto' with profile locked
    //   agentic → 'agentic' (LangGraph agent route on the server)
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
      min_rerank_score: qs.min_rerank_score,
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
        sourceNodes: snap.sourceNodes,
        traceType,
        traceMetadata: snap.metadata,
        chunks: extractChunks(snap.metadata),
        query,
      },
    ])
  }, [workspaceId, send, setMessages])

  // MessageList still calls it "streaming" externally — it really just means
  // "a request is in flight; show the placeholder bubble". The actual
  // transport is non-streaming now.
  const isPending = status === 'pending'
  const isBusy = isPending || multimodalBusy || agentBusy

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <span className="text-xs text-muted-foreground">workspace: {workspaceId}</span>
        <div className="flex items-center gap-1">
          {agentBusy && (
            <Button variant="ghost" size="icon" onClick={stopAgent} title="Stop agent">
              <CircleStop className="h-4 w-4 text-red-500" />
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={newConversation} title="New conversation">
            <SquarePen className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <MessageList
        messages={messages}
        streamingAnswer={answer}
        streamingReasoning=""
        isStreaming={isPending || agentBusy}
        workspaceId={workspaceId}
      />
      <ChatInput onSend={handleSend} disabled={isBusy} />
    </div>
  )
}
