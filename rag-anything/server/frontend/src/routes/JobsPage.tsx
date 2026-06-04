import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible'
import { ChevronDown } from 'lucide-react'
import { useJobs } from '@/hooks/useJobs'
import { useAppStore } from '@/store'
import { JobList } from '@/components/jobs/JobList'

export default function JobsPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { data: jobs = [], isLoading } = useJobs(workspaceId)

  if (isLoading) {
    return <div className="p-6 text-sm text-muted-foreground">Loading...</div>
  }

  return (
    <div className="h-full overflow-y-auto p-6 max-w-2xl mx-auto flex flex-col gap-6">
      <h1 className="text-lg font-semibold text-foreground">Ingest Jobs</h1>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Queued</h2>
        <JobList jobs={jobs} filter="queued" />
      </section>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Running</h2>
        <JobList jobs={jobs} filter="running" />
      </section>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Completed</h2>
        <JobList jobs={jobs} filter="done" />
      </section>

      <section>
        <h2 className="text-sm font-medium text-muted-foreground mb-2">Failed</h2>
        <JobList jobs={jobs} filter="failed" />
        <JobList jobs={jobs} filter="crashed" />
      </section>

      <Collapsible>
        <CollapsibleTrigger
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          Audit log <ChevronDown className="h-3 w-3" />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <p className="text-xs text-muted-foreground mt-2">Audit log coming in v2.</p>
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}
