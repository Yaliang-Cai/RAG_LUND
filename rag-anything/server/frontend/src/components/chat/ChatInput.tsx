import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Send } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSend: (query: string, mode: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [mode, setMode] = useState('hybrid')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q, mode)
    setValue('')
    textareaRef.current?.focus()
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

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
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Mode:</span>
        <Select value={mode} onValueChange={(v) => { if (v !== null) setMode(v) }}>
          <SelectTrigger className="h-7 w-28 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {['naive', 'local', 'global', 'hybrid'].map((m) => (
              <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
