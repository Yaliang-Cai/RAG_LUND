// Mode → preset parameter map. Single source of truth for the chat UI.
// Values mirror backend defaults in raganything/constants.py; the mode-specific
// overrides below match the design spec at
// docs/superpowers/specs/2026-05-21-agentic-modes-ui-refactor-design.md.

export type ModeKey = 'naive' | 'lightrag' | 'multihop' | 'agentic' | 'agentv3'

export type QdrantRetrievalMode = 'hybrid' | 'bm25' | 'dense'
export type AgenticProfile = 'auto' | 'semantic' | 'multihop' | 'full'

export interface ModeConfig {
  // Sent to backend `mode` field
  backendMode: 'naive' | 'auto'
  // Sent to backend `profile` field — undefined means "let backend decide"
  profile?: 'semantic' | 'multihop'

  // Core retrieval knobs
  top_k: number          // KG entity recall; ignored when topKVisible=false
  chunk_top_k: number    // final chunk window
  enable_rerank: boolean
  min_rerank_score: number  // post-rerank score filter; chunks below are dropped
  qdrant_retrieval_mode: QdrantRetrievalMode

  // Naive mode only — pool size before rerank (= 4 * chunk_top_k by default)
  rerank_candidate_cap?: number

  // Multi-hop only
  ppr_damping?: number
  ppr_top_k?: number
  recognition_top_k?: number   // 0 = disabled, >0 = enabled (treated as flag)
  linking_top_k?: number       // HippoRAG2 link_top_k: PPR seed entity cap
  ppr_qa_top_k?: number        // HippoRAG2 qa_top_k: chunks fed to LLM

  // Agentic only — UI dropdown for picking the router profile
  agenticProfile?: AgenticProfile

  // When true the chat calls POST /agent/chat (v3 agent loop) instead of /query.
  // The agent self-manages top_k / rerank / budget, so retrieval knobs are hidden.
  usesAgentEndpoint?: boolean

  // UI affordances
  topKVisible: boolean   // Naive hides top_k
}

export const DEFAULT_MODE: ModeKey = 'agentic'

export const MODE_LABELS: Record<ModeKey, string> = {
  naive: 'Naive',
  lightrag: 'LightRAG',
  multihop: 'Multi-hop',
  agentic: 'Agentic',
  agentv3: 'Agent v3',
}

export const MODE_PRESETS: Record<ModeKey, ModeConfig> = {
  naive: {
    backendMode: 'naive',
    top_k: 10,
    chunk_top_k: 5,
    enable_rerank: true,
    min_rerank_score: 0.3,
    qdrant_retrieval_mode: 'hybrid',
    rerank_candidate_cap: 20, // = 4 * chunk_top_k
    topKVisible: false,
  },
  lightrag: {
    backendMode: 'auto',
    profile: 'semantic',
    top_k: 10,
    chunk_top_k: 5,
    enable_rerank: true,
    min_rerank_score: 0.3,
    qdrant_retrieval_mode: 'hybrid',
    topKVisible: true,
  },
  multihop: {
    backendMode: 'auto',
    profile: 'multihop',
    top_k: 10,
    chunk_top_k: 5,
    enable_rerank: false,
    min_rerank_score: 0.3,
    qdrant_retrieval_mode: 'hybrid',
    ppr_damping: 0.5,
    ppr_top_k: 50,
    recognition_top_k: 20,    // > 0 enables Recognition Memory
    linking_top_k: 5,
    ppr_qa_top_k: 5,
    topKVisible: true,
  },
  agentic: {
    backendMode: 'auto',
    top_k: 10,
    chunk_top_k: 5,
    enable_rerank: true,
    min_rerank_score: 0.3,
    qdrant_retrieval_mode: 'hybrid',
    agenticProfile: 'auto',
    topKVisible: true,
  },
  // v3 真 agent loop：参数由后端 agent 自管，前端不暴露检索旋钮。
  agentv3: {
    backendMode: 'auto',
    top_k: 10,
    chunk_top_k: 5,
    enable_rerank: true,
    min_rerank_score: 0.3,
    qdrant_retrieval_mode: 'hybrid',
    usesAgentEndpoint: true,
    topKVisible: false,
  },
}
