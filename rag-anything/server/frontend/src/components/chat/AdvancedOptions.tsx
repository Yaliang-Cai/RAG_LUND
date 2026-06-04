import { RotateCcw } from 'lucide-react'
import { useQuerySettings } from '@/store/querySettings'
import { MODE_PRESETS, type AgenticProfile, type QdrantRetrievalMode } from '@/config/modePresets'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'

function Row({
  label,
  tooltip,
  onReset,
  children,
}: {
  label: string
  tooltip?: string
  onReset?: () => void
  children: React.ReactNode
}) {
  return (
    <div className="flex items-center gap-2">
      <label className="flex-1 text-xs text-muted-foreground" title={tooltip}>
        {label}
      </label>
      <div className="flex items-center gap-1">
        {children}
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            title="Reset to mode default"
            className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-accent"
          >
            <RotateCcw className="h-3 w-3" />
          </button>
        )}
      </div>
    </div>
  )
}

export function AdvancedOptions() {
  const s = useQuerySettings()

  return (
    <div className="flex flex-col gap-2">
      <Row
        label="Qdrant retrieval mode"
        tooltip="Vector search strategy: hybrid (dense + sparse), bm25 (sparse only), dense (vectors only)"
        onReset={() => s.resetField('qdrant_retrieval_mode')}
      >
        <Select
          value={s.qdrant_retrieval_mode}
          onValueChange={(v) => v && s.setField('qdrant_retrieval_mode', v as QdrantRetrievalMode)}
        >
          <SelectTrigger className="h-7 w-28 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="hybrid" className="text-xs">hybrid</SelectItem>
            <SelectItem value="bm25" className="text-xs">bm25</SelectItem>
            <SelectItem value="dense" className="text-xs">dense</SelectItem>
          </SelectContent>
        </Select>
      </Row>

      {s.mode === 'agentic' && (
        <Row
          label="Profile"
          tooltip="Auto = LLM classifier picks profile; otherwise locks the router to a specific profile."
          onReset={() => s.resetField('agenticProfile')}
        >
          <Select
            value={s.agenticProfile ?? 'auto'}
            onValueChange={(v) => v && s.setField('agenticProfile', v as AgenticProfile)}
          >
            <SelectTrigger className="h-7 w-28 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto" className="text-xs">Auto (LLM)</SelectItem>
              <SelectItem value="semantic" className="text-xs">semantic</SelectItem>
              <SelectItem value="multihop" className="text-xs">multihop</SelectItem>
              <SelectItem value="full" className="text-xs">full</SelectItem>
            </SelectContent>
          </Select>
        </Row>
      )}

      {s.mode === 'multihop' && (
        <>
          <Row
            label="PPR damping"
            tooltip="PageRank damping factor (0,1). Lower = explore farther from seed entities."
            onReset={() => s.resetField('ppr_damping')}
          >
            <Input
              type="number"
              min={0.05}
              max={0.95}
              step={0.05}
              value={s.ppr_damping ?? 0.5}
              onChange={(e) => s.setField('ppr_damping', Math.min(0.95, Math.max(0.05, Number(e.target.value) || 0.5)))}
              className="h-7 w-20 text-xs"
            />
          </Row>
          <Row
            label="PPR retrieval top_k"
            tooltip="Pool size returned by the PPR walk before truncation to qa_top_k."
            onReset={() => s.resetField('ppr_top_k')}
          >
            <Input
              type="number"
              min={1}
              value={s.ppr_top_k ?? 50}
              onChange={(e) => s.setField('ppr_top_k', Math.max(1, Number(e.target.value) || 50))}
              className="h-7 w-20 text-xs"
            />
          </Row>
          <Row
            label="Recognition Memory"
            tooltip="When enabled, an LLM step prunes hybrid-retrieved entities/relations into a focused PPR seed set (HippoRAG2 recognition memory). Disable to skip the LLM call and use direct score merge."
            onReset={() => s.resetField('recognition_top_k')}
          >
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={(s.recognition_top_k ?? 0) > 0}
                onChange={(e) =>
                  s.setField('recognition_top_k', e.target.checked
                    ? (MODE_PRESETS.multihop.recognition_top_k ?? 20)
                    : 0)
                }
                className="cursor-pointer"
              />
              <span className="text-xs text-muted-foreground">
                {(s.recognition_top_k ?? 0) > 0 ? 'Enabled' : 'Disabled'}
              </span>
            </label>
          </Row>
          <Row
            label="Linking top_k"
            tooltip="HippoRAG2 link_top_k: maximum seed entities passed to PPR after recognition. Lower = tighter focus, higher = broader exploration."
            onReset={() => s.resetField('linking_top_k')}
          >
            <Input
              type="number"
              min={1}
              value={s.linking_top_k ?? 5}
              onChange={(e) => s.setField('linking_top_k', Math.max(1, Number(e.target.value) || 5))}
              className="h-7 w-20 text-xs"
            />
          </Row>
          <Row
            label="PPR QA top_k"
            tooltip="HippoRAG2 qa_top_k: number of PPR-ranked chunks fed to the LLM as context. Replaces chunk_top_k for the multi-hop path."
            onReset={() => {
              s.resetField('ppr_qa_top_k')
              s.resetField('chunk_top_k')
            }}
          >
            <Input
              type="number"
              min={1}
              value={s.ppr_qa_top_k ?? 5}
              onChange={(e) => {
                const v = Math.max(1, Number(e.target.value) || 5)
                s.setField('ppr_qa_top_k', v)
                // Mirror to chunk_top_k so the router's final [:chunk_top_k]
                // truncation does not silently shrink the LLM context.
                s.setField('chunk_top_k', v)
              }}
              className="h-7 w-20 text-xs"
            />
          </Row>
        </>
      )}

      {s.mode === 'lightrag' && (
        <p className="text-xs text-muted-foreground italic">No mode-specific advanced options.</p>
      )}
    </div>
  )
}
