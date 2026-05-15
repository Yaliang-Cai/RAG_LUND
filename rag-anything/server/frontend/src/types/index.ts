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
  status: 'pending' | 'processing' | 'processed' | 'failed'
  created_at: string
}

export type JobStatus = 'running' | 'done' | 'failed' | 'cancelled'

export interface Job {
  job_id: string
  workspace_id: string
  doc_id: string
  filename: string
  status: JobStatus
  progress: number
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

export type TraceType = 'agentic' | null

export interface QueryParams {
  workspace_id: string
  query: string
  mode?: 'naive' | 'local' | 'global' | 'hybrid' | 'ppr' | 'auto' | 'agentic'
  profile?: string
  top_k?: number
  chunk_top_k?: number
  enable_rerank?: boolean
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
  created_at: string
}
