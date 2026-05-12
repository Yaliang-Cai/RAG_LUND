import type { QueryParams } from '@/types'

export async function openQueryStream(params: QueryParams): Promise<Response> {
  const body: Record<string, unknown> = {
    workspace_id: params.workspace_id,
    query: params.query,
    mode: params.mode ?? 'hybrid',
    top_k: params.top_k ?? 10,
    chunk_top_k: params.chunk_top_k ?? 10,
    enable_rerank: params.enable_rerank ?? true,
    return_graph: params.return_graph ?? false,
  }
  if (params.mode === 'auto' && params.profile) {
    body.profile = params.profile
  }

  const response = await fetch('/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error((err as { detail?: string }).detail ?? 'Query failed')
  }
  return response
}
