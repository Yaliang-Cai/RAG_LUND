import { JobCard } from './JobCard'
import type { Job, JobStatus } from '@/types'

export function JobList({ jobs, filter }: { jobs: Job[]; filter?: JobStatus }) {
  const filtered = filter ? jobs.filter((j) => j.status === filter) : jobs
  if (filtered.length === 0) {
    return <p className="text-sm text-muted-foreground py-4 text-center">No jobs</p>
  }
  return (
    <div className="flex flex-col gap-2">
      {filtered.map((job) => <JobCard key={job.job_id} job={job} />)}
    </div>
  )
}
