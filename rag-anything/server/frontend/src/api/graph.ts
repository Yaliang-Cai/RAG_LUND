import client from './client'
import type { GraphData } from '@/types'

export async function getOverview(workspaceId: string, maxNodes = 100): Promise<GraphData> {
  const { data } = await client.get<GraphData>(`/graph/${workspaceId}/overview`, {
    params: { max_nodes: maxNodes },
  })
  return data
}

export async function getSubgraph(
  workspaceId: string,
  seed: string,
  depth = 2,
  maxNodes = 50
): Promise<GraphData> {
  const { data } = await client.get<GraphData>(`/graph/${workspaceId}/subgraph`, {
    params: { seed, depth, max_nodes: maxNodes },
  })
  return data
}

// Backend `/graph/{ws}/search` returns matching entity names as plain strings
// (LightRAG `search_labels` -> list[str]), not node objects.
export async function searchGraph(
  workspaceId: string,
  query: string
): Promise<{ results: string[] }> {
  const { data } = await client.get<{ results: string[] }>(`/graph/${workspaceId}/search`, {
    params: { q: query, limit: 20 },
  })
  return data
}
