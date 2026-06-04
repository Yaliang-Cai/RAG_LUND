import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible'
import { ChevronDown } from 'lucide-react'

export function ReasoningTrace({ text }: { text: string }) {
  if (!text) return null
  return (
    <Collapsible className="mt-1">
      <CollapsibleTrigger className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors px-2">
        Thinking <ChevronDown className="h-3 w-3" />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <pre className="mt-1 text-[11px] font-mono text-muted-foreground whitespace-pre-wrap bg-muted/30 rounded p-2 max-h-40 overflow-y-auto">
          {text}
        </pre>
      </CollapsibleContent>
    </Collapsible>
  )
}
