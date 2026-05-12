import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadFile } from '@/api/files'
import { useAppStore } from '@/store'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'

const SUPPORTED = new Set([
  'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
  'txt', 'md', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp',
])

export function FileUpload({ disabled }: { disabled?: boolean }) {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)

  function validate(file: File): boolean {
    const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
    if (!SUPPORTED.has(ext)) {
      toast.error(`Unsupported file type: .${ext}`)
      return false
    }
    return true
  }

  async function upload(file: File) {
    if (!validate(file)) return
    setUploading(true)
    try {
      await uploadFile(workspaceId, file)
      toast.success('Ingest job created')
      navigate('/jobs')
    } catch (err) {
      toast.error((err as Error).message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="p-3 border-t border-border shrink-0">
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
      />
      <div
        className={cn(
          'border border-dashed rounded-lg p-3 text-center cursor-pointer text-xs',
          'text-muted-foreground transition-colors hover:border-primary hover:text-foreground',
          dragging && 'border-primary bg-primary/5',
          (disabled || uploading) && 'opacity-50 pointer-events-none'
        )}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          const file = e.dataTransfer.files[0]
          if (file) upload(file)
        }}
      >
        {uploading ? 'Uploading...' : '+ Upload file'}
      </div>
    </div>
  )
}
