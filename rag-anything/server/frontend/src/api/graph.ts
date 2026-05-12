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

export async function searchGraph(
  workspaceId: string,
  query: string
): Promise<{ nodes: Array<{ id: string; label: string; type: string }> }> {
  const { data } = await client.get(`/graph/${workspaceId}/search`, {
    params: { query, limit: 20 },
  })
  return data
}
