import {
  AlertDialog, AlertDialogAction, AlertDialogCancel,
  AlertDialogContent, AlertDialogDescription, AlertDialogFooter,
  AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Switch } from '@/components/ui/switch'
import { deleteDocument } from '@/api/files'
import { freezeWorkspace, unfreezeWorkspace } from '@/api/workspace'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '@/store'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import type { FileRecord, Workspace } from '@/types'

interface FileListProps {
  files: FileRecord[]
  workspace: Workspace | undefined
  selectedFile: string | null
  onSelect: (filename: string) => void
}

export function FileList({ files, workspace, selectedFile, onSelect }: FileListProps) {
  const qc = useQueryClient()
  const workspaceId = useAppStore((s) => s.workspaceId)
  const frozen = workspace?.frozen ?? false

  async function handleDelete(docId: string, filename: string) {
    try {
      await deleteDocument(workspaceId, docId)
      qc.invalidateQueries({ queryKey: ['files', workspaceId] })
      toast.success(`Deleted ${filename}`)
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  async function handleFreezeToggle(checked: boolean) {
    try {
      if (checked) await freezeWorkspace(workspaceId)
      else await unfreezeWorkspace(workspaceId)
      qc.invalidateQueries({ queryKey: ['workspaces'] })
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  return (
    <div className="flex flex-col h-full border-r border-border overflow-hidden">
      <div className="p-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch checked={frozen} onCheckedChange={handleFreezeToggle} id="freeze-switch" />
          <label htmlFor="freeze-switch" className="cursor-pointer select-none">
            {frozen ? 'Frozen 🔒' : 'Freeze'}
          </label>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {files.length === 0 && (
          <p className="text-xs text-muted-foreground p-3">No files yet</p>
        )}
        {files.map((f) => (
          <div
            key={f.filename}
            className={cn(
              'flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-accent/50 group text-sm',
              selectedFile === f.filename && 'bg-accent'
            )}
            onClick={() => onSelect(f.filename)}
          >
            <span className="truncate flex-1 text-foreground">📄 {f.filename}</span>
            {f.doc_id && !frozen && (
              <AlertDialog>
                <AlertDialogTrigger
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-destructive/20 transition-opacity shrink-0"
                  onClick={(e: React.MouseEvent) => e.stopPropagation()}
                >
                  🗑
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete document?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This removes {f.filename} from the knowledge graph. Cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={() => handleDelete(f.doc_id!, f.filename)}>
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
