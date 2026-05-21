import { useState } from 'react'
import { ChevronDown, RotateCcw, Settings2 } from 'lucide-react'
import { useQuerySettings } from '@/store/querySettings'
import { MODE_LABELS, MODE_PRESETS, type ModeKey } from '@/config/modePresets'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'
import { AdvancedOptions } from './AdvancedOptions'

const MODES: ModeKey[] = ['naive', 'lightrag', 'multihop', 'agentic']

function ResetButton({ onClick, title }: { onClick: () => void; title: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
    >
      <RotateCcw className="h-3 w-3" />
    </button>
  )
}

export function QuerySettings() {
  const s = useQuerySettings()
  const preset = MODE_PRESETS[s.mode]
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const hasAdvanced = s.mode === 'agentic' || s.mode === 'multihop' || s.mode === 'lightrag'

  return (
    <div className="border-t border-border px-3 pt-2 pb-1 flex flex-col gap-1.5 shrink-0">
      <div className="flex items-center gap-2 flex-wrap text-xs">
        <span className="text-muted-foreground">Mode</span>
        <Select value={s.mode} onValueChange={(v) => v && s.setMode(v as ModeKey)}>
          <SelectTrigger className="h-7 w-28 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODES.map((m) => (
              <SelectItem key={m} value={m} className="text-xs">{MODE_LABELS[m]}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        {preset.topKVisible && (
          <>
            <span className="text-muted-foreground ml-2">top_k</span>
            <Input
              type="number"
              min={1}
              value={s.top_k}
              onChange={(e) => s.setField('top_k', Math.max(1, Number(e.target.value) || 1))}
              className="h-7 w-16 text-xs"
            />
            <ResetButton onClick={() => s.resetField('top_k')} title="Reset top_k" />
          </>
        )}

        <span className="text-muted-foreground ml-2">chunk_top_k</span>
        <Input
          type="number"
          min={1}
          value={s.chunk_top_k}
          onChange={(e) => s.setField('chunk_top_k', Math.max(1, Number(e.target.value) || 1))}
          className="h-7 w-16 text-xs"
        />
        <ResetButton onClick={() => s.resetField('chunk_top_k')} title="Reset chunk_top_k" />

        <label className="flex items-center gap-1.5 ml-2 cursor-pointer">
          <input
            type="checkbox"
            checked={s.enable_rerank}
            onChange={(e) => s.setField('enable_rerank', e.target.checked)}
            className="cursor-pointer"
          />
          <span className="text-muted-foreground">Rerank</span>
        </label>
        <ResetButton onClick={() => s.resetField('enable_rerank')} title="Reset rerank" />

        {hasAdvanced && (
          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen} className="ml-auto">
            <CollapsibleTrigger
              className={cn(
                'inline-flex items-center gap-1 h-7 px-2 rounded-md border border-border text-xs',
                'text-muted-foreground hover:text-foreground hover:bg-accent transition-colors',
              )}
            >
              <Settings2 className="h-3 w-3" />
              Advanced
              <ChevronDown className={cn('h-3 w-3 transition-transform', advancedOpen && 'rotate-180')} />
            </CollapsibleTrigger>
            <CollapsibleContent className="absolute right-3 z-10 mt-1 w-[320px] rounded-md border border-border bg-background p-3 shadow-md">
              <AdvancedOptions />
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  )
}
