import { useEffect } from 'react'
import { Outlet, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { TopNav } from './TopNav'
import { useJobs } from '@/hooks/useJobs'
import { useAppStore } from '@/store'

export function AppShell() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { lastSeenJobStatuses, setLastSeenJobStatuses } = useAppStore()
  const navigate = useNavigate()
  const { data: jobs = [] } = useJobs(workspaceId)

  useEffect(() => {
    if (jobs.length === 0) return

    const currentStatuses: Record<string, string> = {}
    for (const job of jobs) {
      currentStatuses[job.job_id] = job.status
      const prev = lastSeenJobStatuses[job.job_id]

      if (prev === 'running' && job.status === 'failed') {
        toast.error(`Ingest failed: ${job.filename}`, {
          action: { label: 'View Jobs', onClick: () => navigate('/jobs') },
        })
      } else if (prev === 'running' && job.status === 'done') {
        toast.success(`Ingest complete: ${job.filename}`)
      }
    }
    setLastSeenJobStatuses(currentStatuses)
  }, [jobs]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col h-screen bg-background">
      <TopNav />
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
