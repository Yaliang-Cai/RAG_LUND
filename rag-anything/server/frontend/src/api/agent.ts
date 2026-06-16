// Client for the v3 agent loop endpoints (server/agent_routes.py).
//   POST /agent/chat                         — run one agent turn for a session
//   POST /agent/sessions/{id}/cancel         — interrupt the in-flight turn
import type { AgentChatResponse } from '@/types'

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
