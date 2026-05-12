import client from './client'
import type { Workspace, AuditEntry } from '@/types'

export async function getWorkspaces(): Promise<Workspace[]> {
  const { data } = await client.get<{ workspaces: Workspace[] }>('/workspaces')
  return data.workspaces
}

export async function deleteWorkspace(id: string): Promise<void> {
  await client.delete(`/workspace/${id}`)
}

export async function freezeWorkspace(id: string): Promise<void> {
  await client.post(`/workspace/${id}/freeze`)
}

export async function unfreezeWorkspace(id: string): Promise<void> {
  await client.post(`/workspace/${id}/unfreeze`)
}

export async function getAuditLog(id: string): Promise<AuditEntry[]> {
  const { data } = await client.get<{ entries: AuditEntry[] }>(`/workspace/${id}/audit`)
  return data.entries
}
