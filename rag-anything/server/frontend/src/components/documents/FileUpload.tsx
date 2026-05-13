import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadFile, uploadFiles } from '@/api/files'
import { useAppStore } from '@/store'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

const SUPPORTED = new Set([
  'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
  'txt', 'md', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp',
])

// HTMLInputElement extension for folder selection
type DirInputAttrs = { webkitdirectory?: string; directory?: string }

export function FileUpload({ disabled }: { disabled?: boolean }) {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dirInputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)

  function filterSupported(list: File[]): File[] {
    const ok: File[] = []
    const rejected: string[] = []
    for (const f of list) {
      const ext = f.name.split('.').pop()?.toLowerCase() ?? ''
      if (SUPPORTED.has(ext)) ok.push(f)
      else rejected.push(f.name)
    }
    if (rejected.length) {
      toast.error(`Skipped ${rejected.length} unsupported file(s)`)
    }
    return ok
  }

  async function uploadMany(rawFiles: File[]) {
    const files = filterSupported(rawFiles)
    if (files.length === 0) return
    setUploading(true)
    try {
      if (files.length === 1) {
        await uploadFile(workspaceId, files[0])
        toast.success('Ingest job created')
      } else {
        const res = await uploadFiles(workspaceId, files)
        if (res.status === 'duplicate') {
          toast.info(`All ${files.length} files already ingested`)
        } else {
          const dup = res.duplicate_doc_ids?.length ?? 0
          const queued = res.doc_ids?.length ?? files.length
          toast.success(
            dup > 0
              ? `Queued ${queued} file(s) (${dup} duplicate skipped)`
              : `Queued ${queued} file(s)`
          )
        }
      }
      navigate('/jobs')
    } catch (err) {
      toast.error((err as Error).message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="p-3 border-t border-border shrink-0 space-y-2">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files) uploadMany(Array.from(e.target.files))
          e.target.value = ''
        }}
      />
      <input
        ref={dirInputRef}
        type="file"
        multiple
        className="hidden"
        {...({ webkitdirectory: '', directory: '' } as DirInputAttrs)}
        onChange={(e) => {
          if (e.target.files) uploadMany(Array.from(e.target.files))
          e.target.value = ''
        }}
      />
      <div
        className={cn(
          'border border-dashed rounded-lg p-3 text-center cursor-pointer text-xs',
          'text-muted-foreground transition-colors hover:border-primary hover:text-foreground',
          dragging && 'border-primary bg-primary/5',
          (disabled || uploading) && 'opacity-50 pointer-events-none'
        )}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          const list = Array.from(e.dataTransfer.files)
          if (list.length) uploadMany(list)
        }}
      >
        {uploading ? 'Uploading...' : '+ Upload file(s)'}
      </div>
      <button
        type="button"
        disabled={disabled || uploading}
        onClick={() => dirInputRef.current?.click()}
        className={cn(
          'w-full text-xs py-1.5 rounded-md border border-border text-muted-foreground',
          'hover:bg-muted hover:text-foreground transition-colors',
          (disabled || uploading) && 'opacity-50 pointer-events-none'
        )}
      >
        + Upload folder
      </button>
    </div>
  )
}
