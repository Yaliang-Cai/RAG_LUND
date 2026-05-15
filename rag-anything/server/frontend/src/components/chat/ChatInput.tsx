import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Send, Eye, Image as ImageIcon, X } from 'lucide-react'
import { cn } from '@/lib/utils'

const MODES = ['naive', 'local', 'global', 'hybrid', 'ppr', 'auto', 'agentic'] as const
const PROFILES = ['precise', 'local', 'multihop', 'descriptive', 'full'] as const

const MAX_IMAGES = 4
const ACCEPT_IMAGE = 'image/jpeg,image/png,image/gif,image/bmp,image/webp,image/tiff'

interface ChatInputProps {
  onSend: (
    query: string,
    mode: string,
    profile: string,
    vlmEnabled: boolean,
    images: File[],
  ) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [profile, setProfile] = useState('')
  const [vlmEnabled, setVlmEnabled] = useState(false)
  const [images, setImages] = useState<File[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q, mode, profile, vlmEnabled, images)
    setValue('')
    setImages([])
    textareaRef.current?.focus()
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function handleFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? [])
    const remaining = MAX_IMAGES - images.length
    setImages((prev) => [...prev, ...files.slice(0, remaining)])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  function removeImage(idx: number) {
    setImages((prev) => prev.filter((_, i) => i !== idx))
  }

  const isAuto = mode === 'auto'
  const hasImages = images.length > 0

  return (
    <div className="border-t border-border p-3 flex flex-col gap-2 shrink-0">
      {hasImages && (
        <div className="flex flex-wrap gap-2">
          {images.map((file, idx) => (
            <div
              key={`${file.name}-${idx}`}
              className="relative h-16 w-16 rounded-md overflow-hidden border border-border bg-secondary"
            >
              <img
                src={URL.createObjectURL(file)}
                alt={file.name}
                className="h-full w-full object-cover"
              />
              <button
                type="button"
                onClick={() => removeImage(idx)}
                className="absolute top-0 right-0 bg-background/80 rounded-bl-md p-0.5 hover:bg-background"
                title="Remove"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2 items-start">
        <textarea
          ref={textareaRef}
          className={cn(
            'flex-1 bg-secondary rounded-xl px-3 py-2 text-sm resize-none outline-none',
            'min-h-[40px] max-h-[160px] placeholder:text-muted-foreground text-foreground'
          )}
          placeholder={
            hasImages
              ? 'Ask a question about the attached image(s)…'
              : 'Ask about your documents... (Enter to send, Shift+Enter for newline)'
          }
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          rows={1}
        />
        <Button size="icon" onClick={submit} disabled={disabled || !value.trim()}>
          <Send className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPT_IMAGE}
          multiple
          className="hidden"
          onChange={handleFiles}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || images.length >= MAX_IMAGES}
          className={cn(
            'inline-flex items-center gap-1 h-7 px-2 rounded-md border text-xs transition-colors',
            'border-border text-muted-foreground hover:text-foreground disabled:opacity-50',
          )}
          title={`Attach image (max ${MAX_IMAGES})`}
        >
          <ImageIcon className="h-3 w-3" />
          Image{images.length > 0 ? ` (${images.length})` : ''}
        </button>

        <span className="text-xs text-muted-foreground">Mode:</span>
        <Select value={mode} onValueChange={(v) => { if (v != null) { setMode(v); if (v !== 'auto') setProfile('') } }}>
          <SelectTrigger className="h-7 w-28 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODES.map((m) => (
              <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        {isAuto && (
          <>
            <span className="text-xs text-muted-foreground">Profile:</span>
            <Select value={profile} onValueChange={(v) => { if (v != null) setProfile(v) }}>
              <SelectTrigger className="h-7 w-32 text-xs">
                <SelectValue placeholder="— auto detect —" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="" className="text-xs text-muted-foreground">— auto detect —</SelectItem>
                {PROFILES.map((p) => (
                  <SelectItem key={p} value={p} className="text-xs">{p}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </>
        )}

        <button
          type="button"
          onClick={() => setVlmEnabled((v) => !v)}
          className={cn(
            'inline-flex items-center gap-1 h-7 px-2 rounded-md border text-xs transition-colors',
            vlmEnabled
              ? 'bg-primary/10 border-primary text-primary'
              : 'border-border text-muted-foreground hover:text-foreground'
          )}
          title="VLM enhanced: use vision model to reason over images in retrieved documents"
        >
          <Eye className="h-3 w-3" />
          VLM
        </button>
      </div>
    </div>
  )
}
