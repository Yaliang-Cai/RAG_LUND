import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Send, Eye } from 'lucide-react'
import { cn } from '@/lib/utils'

const MODES = ['naive', 'local', 'global', 'hybrid', 'ppr', 'auto', 'agentic'] as const
const PROFILES = ['precise', 'local', 'multihop', 'descriptive', 'full'] as const

interface ChatInputProps {
  onSend: (query: string, mode: string, profile: string, vlmEnabled: boolean) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [profile, setProfile] = useState('')
  const [vlmEnabled, setVlmEnabled] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q, mode, profile, vlmEnabled)
    setValue('')
    textareaRef.current?.focus()
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const isAuto = mode === 'auto'

  return (
    <div className="border-t border-border p-3 flex flex-col gap-2 shrink-0">
      <div className="flex gap-2 items-start">
        <textarea
          ref={textareaRef}
          className={cn(
            'flex-1 bg-secondary rounded-xl px-3 py-2 text-sm resize-none outline-none',
            'min-h-[40px] max-h-[160px] placeholder:text-muted-foreground text-foreground'
          )}
          placeholder="Ask about your documents... (Enter to send, Shift+Enter for newline)"
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
