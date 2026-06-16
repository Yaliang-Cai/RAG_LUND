import { cn } from '@/lib/utils'
import type { AgentMetrics } from '@/types'

function pct(v?: number): number | null {
  if (typeof v !== 'number' || !Number.isFinite(v)) return null
  return Math.round(v * 100)
}

function barColor(p: number): string {
  if (p >= 80) return 'bg-green-500'
  if (p >= 50) return 'bg-amber-500'
  return 'bg-red-500'
}

/**
 * Compact answer-quality panel shown under an agent answer: faithfulness bar plus
 * coverage / citation counts. Mirrors the credibility readouts of professional RAG
 * systems (RAGAS-style faithfulness). English-only labels.
 */
export function ConfidencePanel({ metrics }: { metrics?: AgentMetrics }) {
  if (!metrics || Object.keys(metrics).length === 0) return null
  const faith = pct(metrics.faithfulness)
  const cov = pct(metrics.coverage)
  const grounded = Boolean(metrics.grounded)

  return (
    <div className="mt-1 rounded-lg border border-border bg-secondary/50 px-3 py-2 text-xs max-w-[80%] space-y-1.5">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'rounded-full border px-2 py-0.5 text-[10px]',
            grounded ? 'text-green-500 border-green-800' : 'text-amber-500 border-amber-800'
          )}
        >
          {grounded ? '✓ grounded' : '~ partial support'}
        </span>
        {typeof metrics.claims_supported === 'number' && typeof metrics.claims_total === 'number' && (
          <span className="text-muted-foreground">
            {metrics.claims_supported}/{metrics.claims_total} claims cited
          </span>
        )}
        {typeof metrics.citations === 'number' && (
          <span className="text-muted-foreground">· {metrics.citations} sources</span>
        )}
      </div>
      {faith !== null && (
        <div className="flex items-center gap-2">
          <span className="w-20 shrink-0 text-muted-foreground">Faithfulness</span>
          <div className="h-1.5 flex-1 rounded-full bg-border overflow-hidden">
            <div className={cn('h-full rounded-full transition-all', barColor(faith))} style={{ width: `${faith}%` }} />
          </div>
          <span className="w-8 shrink-0 text-right font-medium text-foreground">{faith}%</span>
        </div>
      )}
      {cov !== null && (
        <div className="flex items-center gap-2">
          <span className="w-20 shrink-0 text-muted-foreground">Coverage</span>
          <div className="h-1.5 flex-1 rounded-full bg-border overflow-hidden">
            <div className={cn('h-full rounded-full transition-all', barColor(cov))} style={{ width: `${cov}%` }} />
          </div>
          <span className="w-8 shrink-0 text-right font-medium text-foreground">{cov}%</span>
        </div>
      )}
    </div>
  )
}
