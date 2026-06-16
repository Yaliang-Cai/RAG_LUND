import { useState } from 'react'
import {
  Brain,
  Compass,
  Search,
  Layers,
  ClipboardCheck,
  PenLine,
  Anchor,
  Loader2,
  ChevronDown,
  ChevronRight,
  type LucideIcon,
} from 'lucide-react'
import { cn } from '@/lib/utils'

export interface PhaseStep {
  phase: string
  detail: string
  items?: string[]
}

// Phase → human label + icon. Keeps the UI English-only.
const PHASE_META: Record<string, { label: string; icon: LucideIcon }> = {
  plan: { label: 'Plan', icon: Brain },
  decide: { label: 'Decide', icon: Compass },
  expand: { label: 'Expand query', icon: Layers },
  seed: { label: 'PPR seed', icon: Anchor },
  retrieve: { label: 'Retrieve', icon: Search },
  grade: { label: 'Grade evidence', icon: ClipboardCheck },
  generate: { label: 'Generate', icon: PenLine },
}

function PhaseRow({ step, active }: { step: PhaseStep; active: boolean }) {
  const meta = PHASE_META[step.phase] ?? { label: step.phase, icon: Compass }
  const Icon = active ? Loader2 : meta.icon
  return (
    <div className="space-y-0.5">
      <div className="flex items-start gap-2 text-[11px] leading-snug">
        <Icon className={cn('mt-0.5 h-3 w-3 shrink-0', active ? 'animate-spin text-primary' : 'text-muted-foreground')} />
        <span className="font-medium text-foreground">{meta.label}</span>
        {step.detail && <span className="text-muted-foreground truncate">— {step.detail}</span>}
      </div>
      {step.items && step.items.length > 0 && (
        <ul className="ml-5 space-y-0.5">
          {step.items.map((it, i) => (
            <li key={i} className="text-[10px] text-muted-foreground leading-snug">
              <span className="text-primary/70">›</span> <span className="break-words">{it}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/**
 * The agent's thinking chain. While `live`, it stays expanded and spins the latest
 * step; once the turn is done it collapses to a compact, reviewable summary.
 */
export function ThinkingTrace({ steps, live }: { steps: PhaseStep[]; live?: boolean }) {
  const [open, setOpen] = useState(false)
  if (steps.length === 0) return null

  if (live) {
    return (
      <div className="rounded-lg border border-border bg-secondary/50 px-3 py-2 space-y-1 max-w-[80%]">
        {steps.map((s, i) => (
          <PhaseRow key={i} step={s} active={i === steps.length - 1} />
        ))}
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-border bg-secondary/50 text-xs overflow-hidden max-w-[80%]">
      <button
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span>Thinking ({steps.length} steps)</span>
      </button>
      {open && (
        <div className="px-3 pb-2 pt-1 space-y-1">
          {steps.map((s, i) => (
            <PhaseRow key={i} step={s} active={false} />
          ))}
        </div>
      )}
    </div>
  )
}
