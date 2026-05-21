import type { QueryParams } from '@/types'

export interface MultimodalQueryResult {
  answer: string
  workspace_id: string
  image_count: number
}

export async function postMultimodalQuery(args: {
  workspace_id: string
  query: string
  images: File[]
  mode?: string
  top_k?: number
  chunk_top_k?: number
  enable_rerank?: boolean
}): Promise<MultimodalQueryResult> {
  const form = new FormData()
  form.append('workspace_id', args.workspace_id)
  form.append('query', args.query)
  if (args.mode) form.append('mode', args.mode)
  if (args.top_k != null) form.append('top_k', String(args.top_k))
  if (args.chunk_top_k != null) form.append('chunk_top_k', String(args.chunk_top_k))
  if (args.enable_rerank != null) form.append('enable_rerank', String(args.enable_rerank))
  for (const img of args.images) form.append('images', img, img.name)

  const response = await fetch('/query/multimodal', { method: 'POST', body: form })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error((err as { detail?: string }).detail ?? 'Multimodal query failed')
  }
  return (await response.json()) as MultimodalQueryResult
}

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
  if (params.qdrant_retrieval_mode) body.qdrant_retrieval_mode = params.qdrant_retrieval_mode
  if (params.rerank_candidate_cap != null) body.rerank_candidate_cap = params.rerank_candidate_cap
  if (params.ppr_damping != null) body.ppr_damping = params.ppr_damping
  if (params.ppr_top_k != null) body.ppr_top_k = params.ppr_top_k
  if (params.ppr_synonym_weight_mode) body.ppr_synonym_weight_mode = params.ppr_synonym_weight_mode
  if (params.recognition_top_k != null) body.recognition_top_k = params.recognition_top_k
  if (params.conversation_history && params.conversation_history.length > 0) {
    body.conversation_history = params.conversation_history
  }
  if (params.vlm_enhanced) {
    body.vlm_enhanced = true
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
