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

function AutoPanel({ metadata }: { metadata: Record<string, unknown> }) {
  const rt = (metadata.routing_trace ?? {}) as Record<string, unknown>
  if (!('profile' in rt) && !('paths_activated' in rt)) return null
  const paths = Array.isArray(rt.paths_activated)
    ? rt.paths_activated.join(', ')
    : String(rt.paths_activated ?? '—')
  return (
    <>
      <PillRow>
        <Pill>{String(rt.profile ?? '?')}</Pill>
        <Pill>conf {String(rt.confidence ?? '?')}</Pill>
        <Pill>paths: {paths}</Pill>
      </PillRow>
      <TraceGrid items={[
        { label: 'profile', value: rt.profile },
        { label: 'paths', value: paths },
        { label: 'after rrf', value: rt.chunks_after_rrf },
        { label: 'after rerank', value: rt.chunks_after_rerank },
        { label: 'final chunks', value: rt.chunks_after_threshold },
      ]} />
    </>
  )
}

function PprPanel({ metadata }: { metadata: Record<string, unknown> }) {
  const d = (metadata.data ?? metadata) as Record<string, unknown>
  const chunks = Array.isArray(d.chunks) ? d.chunks.length : 0
  const entities = Array.isArray(d.entities) ? d.entities.length : 0
  const relations = Array.isArray(d.relations) ? d.relations.length : 0
  if (chunks === 0 && entities === 0) return null
  return (
    <>
      <PillRow>
        <Pill>chunks {chunks}</Pill>
        <Pill>entities {entities}</Pill>
        <Pill>relations {relations}</Pill>
      </PillRow>
      <TraceGrid items={[
        { label: 'chunks', value: chunks },
        { label: 'entities', value: entities },
        { label: 'relations', value: relations },
      ]} />
    </>
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

const LABELS: Record<NonNullable<TraceType>, string> = {
  agentic: 'Agentic trace',
  auto: 'Routing trace',
  ppr: 'PPR trace',
}

export function AgenticTrace({ traceType, metadata }: AgenticTraceProps) {
  const [open, setOpen] = useState(true)

  if (!traceType) return null

  const label = LABELS[traceType]

  return (
    <div className="mt-1 rounded-lg border border-border bg-secondary/50 text-xs overflow-hidden max-w-[80%]">
      <button
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span>{label}</span>
      </button>
      {open && (
        <>
          {traceType === 'agentic' && <AgenticPanel metadata={metadata} />}
          {traceType === 'auto' && <AutoPanel metadata={metadata} />}
          {traceType === 'ppr' && <PprPanel metadata={metadata} />}
        </>
      )}
    </div>
  )
}
