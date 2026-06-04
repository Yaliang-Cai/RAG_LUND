import { useState } from 'react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { ChevronDown, Plus } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useWorkspaces } from '@/hooks/useWorkspaces'
import { useAppStore } from '@/store'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

// Backend constraint: alphanumeric + _ - only (see _validate_workspace_id)
const VALID_WS = /^[A-Za-z0-9_-]+$/

export function WorkspaceSwitcher() {
  const { data: workspaces = [] } = useWorkspaces()
  const { workspaceId, setWorkspace } = useAppStore()
  const qc = useQueryClient()
  const [newName, setNewName] = useState('')

  function switchWorkspace(id: string) {
    setWorkspace(id)
    qc.invalidateQueries()
  }

  function createWorkspace() {
    const name = newName.trim()
    if (!name) return
    if (!VALID_WS.test(name)) {
      toast.error('Use letters, numbers, _ or - only')
      return
    }
    if (workspaces.some((w) => w.workspace_id === name)) {
      toast.info(`Switched to existing workspace "${name}"`)
    } else {
      toast.success(`Switched to "${name}" — upload a file to materialize it`)
    }
    setNewName('')
    switchWorkspace(name)
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
      <DropdownMenuContent align="end" className="min-w-[220px]">
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
        <DropdownMenuSeparator />
        <div
          className="flex items-center gap-1 px-2 py-1.5"
          onKeyDown={(e) => e.stopPropagation()}
        >
          <Input
            placeholder="New workspace"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                createWorkspace()
              }
            }}
            className="h-7 text-xs"
          />
          <button
            type="button"
            onClick={createWorkspace}
            disabled={!newName.trim()}
            className={cn(
              'shrink-0 inline-flex items-center justify-center h-7 w-7 rounded-md border border-border',
              'text-muted-foreground hover:bg-muted hover:text-foreground transition-colors',
              !newName.trim() && 'opacity-50 pointer-events-none'
            )}
            aria-label="Create workspace"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
