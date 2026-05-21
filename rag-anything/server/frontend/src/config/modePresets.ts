// Mode → preset parameter map. Single source of truth for the chat UI.
// Values mirror backend defaults in raganything/constants.py; the mode-specific
// overrides below match the design spec at
// docs/superpowers/specs/2026-05-21-agentic-modes-ui-refactor-design.md.

export type ModeKey = 'naive' | 'lightrag' | 'multihop' | 'agentic'

export type QdrantRetrievalMode = 'hybrid' | 'bm25' | 'dense'
export type AgenticProfile = 'auto' | 'semantic' | 'multihop' | 'full'
export type PprSynonymWeightMode = 'raw' | 'plus_one'

export interface ModeConfig {
  // Sent to backend `mode` field
  backendMode: 'naive' | 'auto'
  // Sent to backend `profile` field — undefined means "let backend decide"
  profile?: 'semantic' | 'multihop'

  // Core retrieval knobs
  top_k: number          // KG entity recall; ignored when topKVisible=false
  chunk_top_k: number    // final chunk window
  enable_rerank: boolean
  qdrant_retrieval_mode: QdrantRetrievalMode

  // Naive mode only — pool size before rerank (= 4 * chunk_top_k by default)
  rerank_candidate_cap?: number

  // Multi-hop only
  ppr_damping?: number
  ppr_top_k?: number
  ppr_synonym_weight_mode?: PprSynonymWeightMode
  recognition_top_k?: number

  // Agentic only — UI dropdown for picking the router profile
  agenticProfile?: AgenticProfile

  // UI affordances
  topKVisible: boolean   // Naive hides top_k
}

export const DEFAULT_MODE: ModeKey = 'agentic'

export const MODE_LABELS: Record<ModeKey, string> = {
  naive: 'Naive',
  lightrag: 'LightRAG',
  multihop: 'Multi-hop',
  agentic: 'Agentic',
}

export const MODE_PRESETS: Record<ModeKey, ModeConfig> = {
  naive: {
    backendMode: 'naive',
    top_k: 10,
    chunk_top_k: 5,
    enable_rerank: true,
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
    qdrant_retrieval_mode: 'hybrid',
    topKVisible: true,
  },
  multihop: {
    backendMode: 'auto',
    profile: 'multihop',
    top_k: 10,
    chunk_top_k: 5,
    enable_rerank: false,
    qdrant_retrieval_mode: 'hybrid',
    ppr_damping: 0.5,
    ppr_top_k: 50,
    ppr_synonym_weight_mode: 'plus_one',
    recognition_top_k: 20,
    topKVisible: true,
  },
  agentic: {
    backendMode: 'auto',
    top_k: 10,
    chunk_top_k: 5,
    enable_rerank: true,
    qdrant_retrieval_mode: 'hybrid',
    agenticProfile: 'auto',
    topKVisible: true,
  },
}
