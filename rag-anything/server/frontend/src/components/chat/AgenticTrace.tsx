import { useState } from 'react'
import type { ReactNode } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { TraceType } from '@/types'

interface AgenticTraceProps {
  traceType: TraceType
  metadata: Record<string, unknown>
}

interface KV { label: string; value: unknown; green?: boolean; red?: boolean }

function TraceGrid({ items }: { items: KV[] }) {
  return (
    <div className="grid grid-cols-3 gap-x-4 gap-y-1 px-3 pb-2 pt-1 text-[11px]">
      {items.map(({ label, value, green, red }) => (
        <div key={label} className="flex gap-1">
          <span className="text-muted-foreground">{label}</span>
          <span className={cn(
            'font-medium',
            green && 'text-green-500',
            red && 'text-red-500',
            !green && !red && 'text-foreground'
          )}>
            {String(value ?? '—')}
          </span>
        </div>
      ))}
    </div>
  )
}

function Pill({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cn(
      'rounded-full border px-2 py-0.5 text-[10px] text-muted-foreground border-border',
      className
    )}>
      {children}
    </span>
  )
}

function PillRow({ children }: { children: ReactNode }) {
  return <div className="flex items-center gap-1.5 px-3 pt-2 pb-1 flex-wrap">{children}</div>
}

function AgenticPanel({ metadata }: { metadata: Record<string, unknown> }) {
  const t = (metadata.agentic_trace ?? {}) as Record<string, unknown>
  if (!('confidence' in t) && !('profile' in t)) return null
  const grounded = Boolean(t.grounded)
  return (
    <>
      <PillRow>
        <Pill className={grounded ? 'text-green-500 border-green-800' : 'text-red-500 border-red-800'}>
          {grounded ? '✓ grounded' : '✗ not grounded'}
        </Pill>
        <Pill>conf {String(t.confidence ?? '?')}</Pill>
        <Pill>{String(t.profile ?? '?')}</Pill>
      </PillRow>
      <TraceGrid items={[
        { label: 'profile', value: t.profile },
        { label: 'retrieve ×', value: t.retrieve_cycles_used },
        { label: 'check ×', value: t.check_cycles_used },
        { label: 'cache hit', value: String(t.router_cache_hit ?? false) },
        { label: 'confidence', value: t.confidence },
        { label: 'grounded', value: String(grounded), green: grounded, red: !grounded },
      ]} />
    </>
  )
}

export function AgenticTrace({ traceType, metadata }: AgenticTraceProps) {
  const [open, setOpen] = useState(false)

  if (traceType !== 'agentic') return null

  return (
    <div className="mt-1 rounded-lg border border-border bg-secondary/50 text-xs overflow-hidden max-w-[80%]">
      <button
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span>Agentic trace</span>
      </button>
      {open && <AgenticPanel metadata={metadata} />}
    </div>
  )
}
