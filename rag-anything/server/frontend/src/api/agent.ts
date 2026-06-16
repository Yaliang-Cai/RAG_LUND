// Client for the v3 agent loop endpoints (server/agent_routes.py).
//   POST /agent/chat                         — run one agent turn for a session
//   POST /agent/chat/stream                  — same, streamed as SSE (phases/tokens/done)
//   POST /agent/sessions/{id}/cancel         — interrupt the in-flight turn
import type { AgentChatResponse, AgentStreamEvent } from '@/types'

export interface AgentChatParams {
  workspace_id: string
  session_id: string
  query: string
  max_seconds?: number
}

/** Run one agent turn. Throws on non-2xx, except 409 which is surfaced typed. */
export async function postAgentChat(
  params: AgentChatParams,
  signal?: AbortSignal
): Promise<AgentChatResponse> {
  const response = await fetch('/agent/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal,
  })
  if (response.status === 409) {
    const body = await response.json().catch(() => ({}))
    const detail = (body as { detail?: { hint?: string } }).detail
    throw new AgentBusyError(detail?.hint ?? 'session busy')
  }
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error((err as { detail?: string }).detail ?? 'Agent chat failed')
  }
  return (await response.json()) as AgentChatResponse
}

/**
 * Run one agent turn over SSE, yielding live events:
 *   { type:'phase', ... }  — the agent's thinking chain, in real time
 *   { type:'token', text }  — answer text, streamed
 *   { type:'done', ...AgentChatResponse }  — final result (chunks/citations/metrics)
 *   { type:'error', message }
 * Throws AgentBusyError on 409.
 */
export async function* streamAgentChat(
  params: AgentChatParams,
  signal?: AbortSignal
): AsyncGenerator<AgentStreamEvent> {
  const response = await fetch('/agent/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal,
  })
  if (response.status === 409) {
    const body = await response.json().catch(() => ({}))
    const detail = (body as { detail?: { hint?: string } }).detail
    throw new AgentBusyError(detail?.hint ?? 'session busy')
  }
  if (!response.ok || !response.body) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error((err as { detail?: string }).detail ?? 'Agent chat failed')
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let sep: number
    // SSE frames are separated by a blank line.
    while ((sep = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      const line = frame.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      const json = line.slice(5).trim()
      if (!json) continue
      try {
        yield JSON.parse(json) as AgentStreamEvent
      } catch {
        /* ignore malformed frame */
      }
    }
  }
}

/** Signal the agent loop to stop at the next step boundary / mid-IO. */
export async function cancelAgent(workspaceId: string, sessionId: string): Promise<void> {
  await fetch(
    `/agent/sessions/${encodeURIComponent(sessionId)}/cancel?workspace_id=${encodeURIComponent(workspaceId)}`,
    { method: 'POST' }
  ).catch(() => {
    /* best-effort — the loop also checks the cancel flag each step */
  })
}

/** Raised when the session already has a turn running (HTTP 409). */
export class AgentBusyError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'AgentBusyError'
  }
}
