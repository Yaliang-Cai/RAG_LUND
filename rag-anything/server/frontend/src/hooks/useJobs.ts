import { useQuery } from '@tanstack/react-query'
import { getJobs } from '@/api/jobs'
import type { Job } from '@/types'

export function useJobs(workspaceId: string) {
  return useQuery<Job[]>({
    queryKey: ['jobs', workspaceId],
    queryFn: () => getJobs(workspaceId),
    enabled: !!workspaceId,
    refetchInterval: (query) => {
      const data = query.state.data
      return data?.some((j) => j.status === 'running') ? 2000 : false
    },
  })
}
