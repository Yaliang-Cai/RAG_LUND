import client from './client'
import type { Job } from '@/types'

export async function getJobs(workspaceId?: string): Promise<Job[]> {
  const { data } = await client.get<{ jobs: Job[] }>('/jobs', {
    params: workspaceId ? { workspace_id: workspaceId } : undefined,
  })
  return data.jobs
}

export async function getJob(jobId: string): Promise<Job> {
  const { data } = await client.get<Job>(`/jobs/${jobId}`)
  return data
}

export async function cancelJob(jobId: string): Promise<void> {
  await client.delete(`/jobs/${jobId}`)
}

export async function retryWorkspace(workspaceId: string): Promise<{ job_id: string }> {
  const { data } = await client.post<{ job_id: string }>(`/retry/${workspaceId}`)
  return data
}
