import { useQuery } from '@tanstack/react-query'
import { getFiles } from '@/api/files'

export function useFiles(workspaceId: string) {
  return useQuery({
    queryKey: ['files', workspaceId],
    queryFn: () => getFiles(workspaceId),
    enabled: !!workspaceId,
  })
}
