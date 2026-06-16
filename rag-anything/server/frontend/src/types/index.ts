export interface Workspace {
  workspace_id: string
  name: string
  frozen: boolean
  document_count: number
  created_at: string
}

export interface DocumentRecord {
  doc_id: string
  filename: string
  file_hash: string
  status: 'pending' | 'parsing' | 'indexing' | 'done' | 'failed' | 'deleting' | 'deleted'
  created_at: string
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'crashed' | 'cancelled'

export interface Job {
  job_id: string
  workspace_id: string
  doc_id: string
  doc_ids?: string[]
  filename: string
  filenames?: string[]
  status: JobStatus
  progress: number
  progress_detail?: Record<string, unknown>
  error: string | null
  created_at: string
  updated_at: string
}

export interface GraphNode {
  id: string
  label: string
  type: string
  description: string
}

export interface GraphEdge {
  source: string
  target: string
  label: string
  weight: number
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface SourceNode {
  doc_id: string
  filename: string
  page_num: number | null
  excerpt: string
}

/**
 * Single chunk surfaced in the SSE meta event (data.chunks[]).
 * Backend field names vary slightly across modes; the loose typing here
 * matches what LightRAG actually emits.
 */
export interface ChunkRef {
  id?: string                  // e.g. "DC1", "DC2" — what the LLM cites
  reference_id?: string | number
  file_path?: string
  filename?: string
  page_idx?: number | null
  page_num?: number | null
  content?: string
  text?: string
  excerpt?: string
}

export interface StreamMetaEvent {
  type: 'meta'
  data: Record<string, unknown>
  metadata: Record<string, unknown>
}

export interface StreamChunkEvent {
  type: 'chunk'
  text: string
}

export interface StreamDoneEvent {
  type: 'done'
  graph: GraphData | null
  source_nodes: SourceNode[]
}

export interface StreamErrorEvent {
  type: 'error'
  text: string
}

export type StreamEvent = StreamMetaEvent | StreamChunkEvent | StreamDoneEvent | StreamErrorEvent

export type TraceType = 'agentic' | 'agentv3' | null

/** One planner decision in the v3 agent loop trace. */
export interface AgentDecision {
  thought: string
  action: string
  params: Record<string, unknown>
  budget: Record<string, unknown>
  fallback: boolean
}

/** Response shape of POST /agent/chat (raganything/agent/loop.py AgentResult). */
export interface AgentChatResponse {
  answer: string | null
  grounded: boolean
  refusal: Record<string, unknown> | null
  ledger: Record<string, unknown>
  trace: Record<string, unknown>
  cancelled: boolean
}

export interface QueryParams {
  workspace_id: string
  query: string
  mode?: 'naive' | 'local' | 'global' | 'hybrid' | 'ppr' | 'auto' | 'agentic'
  profile?: string
  top_k?: number
  chunk_top_k?: number
  enable_rerank?: boolean
  min_rerank_score?: number
  qdrant_retrieval_mode?: 'hybrid' | 'bm25' | 'dense'
  rerank_candidate_cap?: number
  ppr_damping?: number
  ppr_top_k?: number
  recognition_top_k?: number
  linking_top_k?: number
  ppr_qa_top_k?: number
  return_graph?: boolean
  conversation_history?: { role: string; content: string }[]
  vlm_enhanced?: boolean
}

export interface FileRecord {
  filename: string
  doc_id?: string
}

export interface AuditEntry {
  id: number
  workspace_id: string
  action: string
  doc_id: string | null
  detail: string | null
  details?: Record<string, unknown>
  timestamp?: string
  created_at: string
}
