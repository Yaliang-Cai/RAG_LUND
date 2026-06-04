import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement } from 'react'
import { useJobs } from '@/hooks/useJobs'
import * as jobsApi from '@/api/jobs'
import type { Job } from '@/types'

vi.mock('@/api/jobs')

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return createElement(QueryClientProvider, { client: qc }, children)
}

describe('useJobs', () => {
  beforeEach(() => vi.clearAllMocks())

  it('returns jobs from API', async () => {
    const mockJobs: Job[] = [
      {
        job_id: 'j1', status: 'done', filename: 'a.pdf', progress: 100,
        workspace_id: 'ws1', doc_id: 'd1', error: null,
        created_at: '', updated_at: '',
      },
    ]
    vi.mocked(jobsApi.getJobs).mockResolvedValue(mockJobs)
    const { result } = renderHook(() => useJobs('ws1'), { wrapper })
    await waitFor(() => expect(result.current.data).toEqual(mockJobs))
  })

  it('has running job in data when API returns one', async () => {
    const runningJobs: Job[] = [{
      job_id: 'j2', status: 'running', progress: 50, filename: 'b.pdf',
      workspace_id: 'ws1', doc_id: 'd2', error: null,
      created_at: '', updated_at: '',
    }]
    vi.mocked(jobsApi.getJobs).mockResolvedValue(runningJobs)
    const { result } = renderHook(() => useJobs('ws1'), { wrapper })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.some((j) => j.status === 'running')).toBe(true)
  })
})
