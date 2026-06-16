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

function AgentV3Panel({ metadata }: { metadata: Record<string, unknown> }) {
  const resp = (metadata.agent_v3 ?? {}) as Record<string, unknown>
  const trace = (resp.trace ?? {}) as Record<string, unknown>
  const ledger = (resp.ledger ?? {}) as Record<string, unknown>
  if (!('terminal_reason' in trace) && !('answer' in resp)) return null

  const grounded = Boolean(resp.grounded)
  const usedFallback = Boolean(trace.used_fallback)
  const decisions = Array.isArray(trace.agent_decisions)
    ? (trace.agent_decisions as Record<string, unknown>[])
    : []
  const reclass = Array.isArray(trace.reclassify_events)
    ? (trace.reclassify_events as Record<string, unknown>[])
    : []
  const retrievals = Array.isArray(trace.retrieval_steps)
    ? (trace.retrieval_steps as unknown[])
    : []
  const facts = Array.isArray(ledger.facts) ? (ledger.facts as Record<string, unknown>[]) : []
  const unverifiable = facts.filter((f) => f.status === 'unverifiable').length

  return (
    <>
      <PillRow>
        <Pill className={grounded ? 'text-green-500 border-green-800' : 'text-red-500 border-red-800'}>
          {grounded ? '✓ grounded' : '✗ not grounded'}
        </Pill>
        <Pill>{String(trace.terminal_reason ?? '?')}</Pill>
        <Pill>{String(trace.profile ?? '?')}</Pill>
        {usedFallback && <Pill className="text-amber-500 border-amber-800">fallback</Pill>}
        {Boolean(resp.cancelled) && <Pill className="text-amber-500 border-amber-800">cancelled</Pill>}
      </PillRow>
      <TraceGrid items={[
        { label: 'coverage', value: ledger.coverage },
        { label: 'decisions', value: decisions.length },
        { label: 'retrievals', value: retrievals.length },
        { label: 'reclassify', value: reclass.length },
        { label: 'unverifiable', value: unverifiable, red: unverifiable > 0 },
        { label: 'grounded', value: String(grounded), green: grounded, red: !grounded },
      ]} />
      {decisions.length > 0 && (
        <div className="px-3 pb-2 pt-1 space-y-1">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">decision chain</div>
          {decisions.map((d, i) => (
            <div key={i} className="text-[11px] leading-snug">
              <span className="font-medium text-foreground">{i + 1}. {String(d.action ?? '?')}</span>
              {d.fallback ? <span className="text-amber-500"> (fallback)</span> : null}
              {d.thought ? <span className="text-muted-foreground"> — {String(d.thought)}</span> : null}
            </div>
          ))}
        </div>
      )}
      {Boolean(resp.refusal) && (
        <div className="px-3 pb-2 text-[11px] text-red-400">
          refusal: {String((resp.refusal as { reason?: string }).reason ?? 'unanswered')}
        </div>
      )}
    </>
  )
}

export function AgenticTrace({ traceType, metadata }: AgenticTraceProps) {
  const [open, setOpen] = useState(false)

  if (traceType !== 'agentic' && traceType !== 'agentv3') return null
  const label = traceType === 'agentv3' ? 'Agent v3 trace' : 'Agentic trace'

  return (
    <div className="mt-1 rounded-lg border border-border bg-secondary/50 text-xs overflow-hidden max-w-[80%]">
      <button
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span>{label}</span>
      </button>
      {open && (traceType === 'agentv3'
        ? <AgentV3Panel metadata={metadata} />
        : <AgenticPanel metadata={metadata} />)}
    </div>
  )
}
