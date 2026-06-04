import client from './client'
import type { Workspace, AuditEntry } from '@/types'

type WorkspaceResponse = Workspace & {
  uploaded_files?: string[]
}

export async function getWorkspaces(): Promise<Workspace[]> {
  const { data } = await client.get<{ workspaces: WorkspaceResponse[] }>('/workspaces')
  return data.workspaces.map((ws) => ({
    ...ws,
    name: ws.name ?? ws.workspace_id,
    frozen: Boolean(ws.frozen),
    document_count: Number(ws.document_count ?? ws.uploaded_files?.length ?? 0),
    created_at: ws.created_at ?? '',
  }))
}

export async function deleteWorkspace(id: string): Promise<void> {
  await client.delete(`/workspace/${id}`)
}

export async function freezeWorkspace(id: string): Promise<void> {
  await client.patch(`/workspace/${id}/freeze`)
}

export async function unfreezeWorkspace(id: string): Promise<void> {
  await client.patch(`/workspace/${id}/unfreeze`)
}

export async function getAuditLog(id: string): Promise<AuditEntry[]> {
  const { data } = await client.get<{ entries?: AuditEntry[]; audit?: AuditEntry[] }>(
    `/workspace/${id}/audit`
  )
  return data.entries ?? data.audit ?? []
}
