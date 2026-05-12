import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { cancelJob, retryWorkspace } from '@/api/jobs'
import { useQueryClient } from '@tanstack/react-query'
import { useAppStore } from '@/store'
import { toast } from 'sonner'
import type { Job } from '@/types'

const STATUS_COLORS: Record<string, string> = {
  running: 'bg-primary text-primary-foreground',
  done: 'bg-green-600 text-white',
  failed: 'bg-destructive text-destructive-foreground',
  cancelled: 'bg-muted text-muted-foreground',
}

export function JobCard({ job }: { job: Job }) {
  const qc = useQueryClient()
  const workspaceId = useAppStore((s) => s.workspaceId)

  async function handleCancel() {
    try {
      await cancelJob(job.job_id)
      qc.invalidateQueries({ queryKey: ['jobs', workspaceId] })
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  async function handleRetry() {
    try {
      await retryWorkspace(workspaceId)
      qc.invalidateQueries({ queryKey: ['jobs', workspaceId] })
      toast.success('Retry job created')
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  return (
    <div className="border border-border rounded-lg p-4 flex flex-col gap-2 bg-card">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium truncate">{job.filename}</span>
        <Badge className={STATUS_COLORS[job.status] + ' text-xs shrink-0'}>
          {job.status}
        </Badge>
      </div>
      {job.status === 'running' && (
        <>
          <Progress value={job.progress} className="h-1.5" />
          <Button variant="outline" size="sm" className="self-start" onClick={handleCancel}>
            Cancel
          </Button>
        </>
      )}
      {job.status === 'failed' && (
        <div className="flex items-center gap-2">
          {job.error && (
            <span className="text-xs text-muted-foreground truncate flex-1">{job.error}</span>
          )}
          <Button variant="outline" size="sm" onClick={handleRetry}>Retry</Button>
        </div>
      )}
      {(job.status === 'done' || job.status === 'cancelled') && (
        <span className="text-xs text-muted-foreground">
          {new Date(job.updated_at).toLocaleString()}
        </span>
      )}
    </div>
  )
}
