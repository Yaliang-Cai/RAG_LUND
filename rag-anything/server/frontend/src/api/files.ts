import client from './client'
import type { FileRecord } from '@/types'

export async function getFiles(workspaceId: string): Promise<FileRecord[]> {
  const { data } = await client.get<{ files: string[] }>(`/files/${workspaceId}`)
  return data.files.map((filename) => ({ filename }))
}

export async function getFileContent(workspaceId: string, filename: string): Promise<string> {
  const { data } = await client.get<{ content: string }>(`/content/${workspaceId}`, {
    params: { filename },
  })
  return data.content
}

export async function uploadFile(workspaceId: string, file: File): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('file', file)
  form.append('workspace_id', workspaceId)
  const { data } = await client.post<{ job_id: string }>('/ingest', form)
  return data
}

export async function deleteDocument(workspaceId: string, docId: string): Promise<void> {
  await client.delete(`/workspace/${workspaceId}/document/${docId}`)
}

export function getUploadUrl(workspaceId: string, filename: string): string {
  return `/uploads/${workspaceId}/${filename}`
}
