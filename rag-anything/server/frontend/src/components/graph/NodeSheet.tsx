import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import type { GraphNode } from '@/types'

export function NodeSheet({ node, onClose }: { node: GraphNode | null; onClose: () => void }) {
  return (
    <Sheet open={!!node} onOpenChange={(open: boolean) => !open && onClose()}>
      <SheetContent side="right" className="w-80">
        {node && (
          <>
            <SheetHeader>
              <SheetTitle className="text-base">{node.label}</SheetTitle>
            </SheetHeader>
            <div className="mt-4 flex flex-col gap-3 text-sm">
              <div>
                <span className="text-xs text-muted-foreground uppercase tracking-wide">Type</span>
                <p className="mt-0.5">{node.type || '—'}</p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground uppercase tracking-wide">Description</span>
                <p className="mt-0.5 text-muted-foreground text-xs leading-relaxed">
                  {node.description || '—'}
                </p>
              </div>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
