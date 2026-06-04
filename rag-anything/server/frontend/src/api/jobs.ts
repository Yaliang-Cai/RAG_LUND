import client from './client'
import type { Job, JobStatus } from '@/types'

type RawJob = {
  job_id?: string
  workspace_id?: string
  doc_id?: string
  doc_ids?: string[]
  filename?: string
  filenames?: string[]
  status?: JobStatus | string
  progress?: number | Record<string, unknown>
  progress_detail?: Record<string, unknown>
  error?: string | null
  created_at?: string
  updated_at?: string
  started_at?: string
  finished_at?: string | null
}

function progressPercent(progress: RawJob['progress'], status?: string): number {
  if (typeof progress === 'number') return Math.max(0, Math.min(100, progress))
  if (progress && typeof progress === 'object') {
    for (const key of ['percent', 'percentage', 'value']) {
      const value = progress[key]
      if (typeof value === 'number') return Math.max(0, Math.min(100, value))
    }
    const total = Number(progress.total ?? 0)
    if (total > 0) {
      const indexed = Number(progress.indexed ?? 0)
      const parsed = Number(progress.parsed ?? 0)
      return Math.max(0, Math.min(100, Math.round((Math.max(indexed, parsed) / total) * 100)))
    }
  }
  return status === 'done' ? 100 : 0
}

function normalizeJob(job: RawJob): Job {
  const docIds = job.doc_ids ?? (job.doc_id ? [job.doc_id] : [])
  const filename =
    job.filename ??
    job.filenames?.[0] ??
    (docIds.length > 1 ? `${docIds.length} documents` : docIds[0] ?? '')

  return {
    ...job,
    job_id: job.job_id ?? '',
    workspace_id: job.workspace_id ?? '',
    doc_id: job.doc_id ?? docIds[0] ?? '',
    doc_ids: docIds,
    filename,
    filenames: job.filenames ?? (filename ? [filename] : []),
    status: (job.status ?? 'queued') as JobStatus,
    progress: progressPercent(job.progress ?? job.progress_detail, job.status),
    progress_detail:
      job.progress_detail ?? (typeof job.progress === 'object' ? job.progress : undefined),
    error: job.error ?? null,
    created_at: job.created_at ?? job.started_at ?? '',
    updated_at: job.updated_at ?? job.finished_at ?? job.started_at ?? '',
  }
}

export async function getJobs(workspaceId?: string): Promise<Job[]> {
  const { data } = await client.get<{ jobs: RawJob[] }>('/jobs', {
    params: workspaceId ? { workspace_id: workspaceId } : undefined,
  })
  return data.jobs.map(normalizeJob)
}

export async function getJob(jobId: string): Promise<Job> {
  const { data } = await client.get<RawJob>(`/jobs/${jobId}`)
  return normalizeJob(data)
}

export async function cancelJob(jobId: string): Promise<void> {
  await client.delete(`/jobs/${jobId}`)
}

export async function retryWorkspace(workspaceId: string): Promise<{ job_id: string }> {
  const { data } = await client.post<{ job_id: string }>(`/retry/${workspaceId}`)
  return data
}
