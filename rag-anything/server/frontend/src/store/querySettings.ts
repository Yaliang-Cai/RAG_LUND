import { create } from 'zustand'
import {
  DEFAULT_MODE,
  MODE_PRESETS,
  type ModeKey,
  type ModeConfig,
  type QdrantRetrievalMode,
  type AgenticProfile,
  type PprSynonymWeightMode,
} from '@/config/modePresets'

export interface QuerySettingsState {
  mode: ModeKey

  // Effective current values. Initialized from preset, mutated by user edits,
  // discarded and reset on mode switch (per design spec §6.4).
  top_k: number
  chunk_top_k: number
  enable_rerank: boolean
  qdrant_retrieval_mode: QdrantRetrievalMode
  rerank_candidate_cap?: number
  ppr_damping?: number
  ppr_top_k?: number
  ppr_synonym_weight_mode?: PprSynonymWeightMode
  recognition_top_k?: number
  agenticProfile?: AgenticProfile

  setMode: (m: ModeKey) => void
  setField: <K extends keyof QuerySettingsState>(k: K, v: QuerySettingsState[K]) => void
  resetField: (k: keyof ModeConfig) => void
}

function snapshotFromPreset(mode: ModeKey): Omit<QuerySettingsState, 'mode' | 'setMode' | 'setField' | 'resetField'> {
  const p = MODE_PRESETS[mode]
  return {
    top_k: p.top_k,
    chunk_top_k: p.chunk_top_k,
    enable_rerank: p.enable_rerank,
    qdrant_retrieval_mode: p.qdrant_retrieval_mode,
    rerank_candidate_cap: p.rerank_candidate_cap,
    ppr_damping: p.ppr_damping,
    ppr_top_k: p.ppr_top_k,
    ppr_synonym_weight_mode: p.ppr_synonym_weight_mode,
    recognition_top_k: p.recognition_top_k,
    agenticProfile: p.agenticProfile,
  }
}

export const useQuerySettings = create<QuerySettingsState>((set, get) => ({
  mode: DEFAULT_MODE,
  ...snapshotFromPreset(DEFAULT_MODE),

  setMode: (m) => set({ mode: m, ...snapshotFromPreset(m) }),

  setField: (k, v) => set({ [k]: v } as Partial<QuerySettingsState>),

  resetField: (k) => {
    const p = MODE_PRESETS[get().mode] as unknown as Record<string, unknown>
    if (k in p) set({ [k]: p[k] } as Partial<QuerySettingsState>)
  },
}))
