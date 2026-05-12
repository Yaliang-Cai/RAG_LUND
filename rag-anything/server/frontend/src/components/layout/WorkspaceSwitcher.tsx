import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ChevronDown } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useWorkspaces } from '@/hooks/useWorkspaces'
import { useAppStore } from '@/store'
import { cn } from '@/lib/utils'

export function WorkspaceSwitcher() {
  const { data: workspaces = [] } = useWorkspaces()
  const { workspaceId, setWorkspace } = useAppStore()
  const qc = useQueryClient()

  function switchWorkspace(id: string) {
    setWorkspace(id)
    qc.invalidateQueries()
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        className={cn(
          'inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md border',
          'border-border bg-background hover:bg-muted text-foreground transition-colors'
        )}
      >
        ws: {workspaceId} <ChevronDown className="h-3 w-3" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {workspaces.map((ws) => (
          <DropdownMenuItem key={ws.workspace_id} onClick={() => switchWorkspace(ws.workspace_id)}>
            {ws.workspace_id}
            {ws.frozen && <span className="ml-2 text-xs text-muted-foreground">🔒</span>}
          </DropdownMenuItem>
        ))}
        {workspaces.length === 0 && (
          <DropdownMenuItem onClick={() => switchWorkspace('default')}>
            default
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
