import { useQuery } from '@tanstack/react-query'
import { getOverview, getSubgraph } from '@/api/graph'

export function useGraphOverview(workspaceId: string) {
  return useQuery({
    queryKey: ['graph', 'overview', workspaceId],
    queryFn: () => getOverview(workspaceId),
    enabled: !!workspaceId,
    staleTime: 60_000,
  })
}

export function useSubgraph(workspaceId: string, seed: string | null, depth = 2) {
  return useQuery({
    queryKey: ['graph', 'subgraph', workspaceId, seed, depth],
    queryFn: () => getSubgraph(workspaceId, seed!, depth),
    enabled: !!workspaceId && !!seed,
  })
}
