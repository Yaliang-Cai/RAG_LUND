import client from './client'
import type { FileRecord } from '@/types'

interface UploadEntry {
  name: string
  size: number
  modified: number
}

export async function getFiles(workspaceId: string): Promise<FileRecord[]> {
  // List original uploaded files (PDF/DOCX/…) rather than parsed .md outputs
  const { data } = await client.get<{ files: UploadEntry[] }>(`/uploads/${workspaceId}`)
  return data.files.map((f) => ({ filename: f.name }))
}

export async function getFileContent(workspaceId: string, filename: string): Promise<string> {
  // Backend /content endpoint requires the .md filename. Map any original
  // extension (e.g. paper.pdf) → paper.md to locate the parsed markdown.
  const dot = filename.lastIndexOf('.')
  const stem = dot > 0 ? filename.slice(0, dot) : filename
  const mdName = `${stem}.md`
  const { data } = await client.get<{ content: string }>(`/content/${workspaceId}`, {
    params: { filename: mdName },
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

export interface BatchUploadResult {
  job_id: string | null
  workspace_id: string
  status: 'queued' | 'duplicate'
  doc_ids?: string[]
  duplicate_doc_ids?: string[]
}

export async function uploadFiles(workspaceId: string, files: File[]): Promise<BatchUploadResult> {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  form.append('workspace_id', workspaceId)
  const { data } = await client.post<BatchUploadResult>('/ingest/batch', form)
  return data
}

export async function deleteDocument(workspaceId: string, docId: string): Promise<void> {
  await client.delete(`/workspace/${workspaceId}/document/${docId}`)
}

export function getUploadUrl(workspaceId: string, filename: string): string {
  return `/uploads/${workspaceId}/${filename}`
}
