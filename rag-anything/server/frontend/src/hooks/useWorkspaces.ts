import { useQuery } from '@tanstack/react-query'
import { getWorkspaces } from '@/api/workspace'

export function useWorkspaces() {
  return useQuery({
    queryKey: ['workspaces'],
    queryFn: getWorkspaces,
    staleTime: 30_000,
  })
}
