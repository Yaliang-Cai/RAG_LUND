import type { QueryParams, SourceNode } from '@/types'

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

export interface QueryResponse {
  answer: string
  data: Record<string, unknown>
  metadata: Record<string, unknown>
  source_nodes: SourceNode[]
  graph: Record<string, unknown> | null
}

/**
 * Non-streaming POST /query. The endpoint goes through LightRAG's
 * ``aquery_llm(stream=False)`` path so identical queries hit
 * ``llm_response_cache`` on the second call (LightRAG explicitly skips the
 * cache write for streaming responses, which is why we no longer use SSE).
 */
export async function postQuery(params: QueryParams): Promise<QueryResponse> {
  const body: Record<string, unknown> = {
    workspace_id: params.workspace_id,
    query: params.query,
    mode: params.mode ?? 'hybrid',
    // Fallbacks mirror raganything/constants.py defaults; ChatPage always
    // sends explicit values from the mode preset, so these only apply to
    // direct callers.
    top_k: params.top_k ?? 10,
    chunk_top_k: params.chunk_top_k ?? 5,
    enable_rerank: params.enable_rerank ?? true,
    return_graph: params.return_graph ?? false,
  }
  if (params.mode === 'auto' && params.profile) {
    body.profile = params.profile
  }
  if (params.qdrant_retrieval_mode) body.qdrant_retrieval_mode = params.qdrant_retrieval_mode
  if (params.min_rerank_score != null) body.min_rerank_score = params.min_rerank_score
  if (params.rerank_candidate_cap != null) body.rerank_candidate_cap = params.rerank_candidate_cap
  if (params.ppr_damping != null) body.ppr_damping = params.ppr_damping
  if (params.ppr_top_k != null) body.ppr_top_k = params.ppr_top_k
  if (params.recognition_top_k != null) body.recognition_top_k = params.recognition_top_k
  if (params.linking_top_k != null) body.linking_top_k = params.linking_top_k
  if (params.ppr_qa_top_k != null) body.ppr_qa_top_k = params.ppr_qa_top_k
  if (params.conversation_history && params.conversation_history.length > 0) {
    body.conversation_history = params.conversation_history
  }
  if (params.vlm_enhanced) {
    body.vlm_enhanced = true
  }

  const response = await fetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error((err as { detail?: string }).detail ?? 'Query failed')
  }
  return (await response.json()) as QueryResponse
}
