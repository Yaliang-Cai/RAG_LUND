import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { evaluateAnswer } from '@/api/evaluate'
import type { EvalResult } from '@/api/evaluate'

interface EvalBadgeProps {
  workspaceId: string
  query: string
  answer: string
}

function scoreColor(score: number) {
  if (score >= 0.85) return 'text-green-600 border-green-300 bg-green-50'
  if (score >= 0.65) return 'text-amber-600 border-amber-300 bg-amber-50'
  return 'text-red-600 border-red-300 bg-red-50'
}

export function EvalBadge({ workspaceId, query, answer }: EvalBadgeProps) {
  const [state, setState] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
  const [result, setResult] = useState<EvalResult | null>(null)
  const [open, setOpen] = useState(true)

  async function runEval() {
    setState('loading')
    try {
      const r = await evaluateAnswer(workspaceId, query, answer)
      setResult(r)
      setState('done')
    } catch {
      setState('error')
    }
  }

  if (state === 'idle') {
    return (
      <button
        onClick={runEval}
        className="mt-1 inline-flex items-center h-6 px-2 rounded border border-border
                   text-[10px] text-muted-foreground hover:text-foreground transition-colors"
      >
        Evaluate
      </button>
    )
  }

  if (state === 'loading') {
    return (
      <span className="mt-1 inline-flex items-center gap-1 h-6 px-2 rounded border border-border text-[10px] text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        Evaluating…
      </span>
    )
  }

  if (state === 'error') {
    return (
      <span className="mt-1 inline-flex items-center h-6 px-2 rounded border border-red-200
                       text-[10px] text-red-500">
        Eval failed
      </span>
    )
  }

  // done
  const score = result!.score
  const gap = result!.gap

  return (
    <div className="mt-1 flex flex-col gap-0.5">
      <button
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'inline-flex items-center h-6 px-2 rounded border text-[10px] font-medium w-fit transition-colors',
          scoreColor(score)
        )}
      >
        {score >= 0.85 ? '●' : score >= 0.65 ? '◐' : '○'} {score.toFixed(2)}
      </button>
      {open && (
        <p className="text-[10px] text-muted-foreground pl-1">
          gap: {gap || '—'}
        </p>
      )}
    </div>
  )
}
