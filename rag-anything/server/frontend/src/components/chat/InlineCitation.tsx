import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '@/store'
import { cn } from '@/lib/utils'
import type { ChunkRef } from '@/types'

interface InlineCitationProps {
  refId: string
  chunk?: ChunkRef
}

function pickFilename(c?: ChunkRef): string | null {
  if (!c) return null
  const raw = c.filename ?? c.file_path ?? ''
  if (!raw) return null
  return String(raw).replace(/\\/g, '/').split('/').pop() ?? null
}

function pickPage(c?: ChunkRef): number | null {
  if (!c) return null
  const p = c.page_num ?? c.page_idx
  if (p == null) return null
  const n = typeof p === 'number' ? p : Number(p)
  return Number.isFinite(n) ? n : null
}

function pickExcerpt(c?: ChunkRef): string {
  if (!c) return ''
  return String(c.content ?? c.text ?? c.excerpt ?? '').slice(0, 240)
}

export function InlineCitation({ refId, chunk }: InlineCitationProps) {
  const navigate = useNavigate()
  const { setSelectedFile, setPendingPageNum } = useAppStore()
  const [open, setOpen] = useState(false)
  const wrapperRef = useRef<HTMLSpanElement>(null)

  // Close on outside click
  useEffect(() => {
    if (!open) return
    function onDocClick(e: MouseEvent) {
      if (!wrapperRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [open])

  const filename = pickFilename(chunk)
  const page = pickPage(chunk)
  const excerpt = pickExcerpt(chunk)
  const hasInfo = Boolean(filename || excerpt)

  function jumpToSource(e: React.MouseEvent) {
    e.stopPropagation()
    if (!filename) return
    setSelectedFile(filename)
    if (page != null) setPendingPageNum(page)
    setOpen(false)
    navigate('/documents')
  }

  return (
    <span ref={wrapperRef} className="relative inline-block align-baseline">
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          if (hasInfo) setOpen((o) => !o)
          else if (filename) jumpToSource(e)
        }}
        onMouseEnter={() => hasInfo && setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        className={cn(
          'mx-0.5 inline-flex items-center justify-center rounded',
          'px-1 text-[10px] font-medium leading-none align-super',
          'border border-border bg-secondary text-muted-foreground',
          'hover:bg-accent hover:text-foreground transition-colors',
          hasInfo ? 'cursor-pointer' : 'cursor-default'
        )}
        title={hasInfo ? undefined : `[${refId}]`}
      >
        [{refId}]
      </button>
      {open && hasInfo && (
        <span
          className={cn(
            'absolute z-50 left-0 top-full mt-1 w-72 max-w-[80vw]',
            'rounded-md border border-border bg-popover text-popover-foreground shadow-md',
            'p-2 text-xs whitespace-normal'
          )}
          onMouseEnter={() => setOpen(true)}
          onMouseLeave={() => setOpen(false)}
        >
          {filename && (
            <div className="flex items-center justify-between gap-2 mb-1">
              <span className="font-medium truncate" title={filename}>
                {filename}
                {page != null && (
                  <span className="ml-1 text-muted-foreground">p.{page}</span>
                )}
              </span>
              <button
                type="button"
                onClick={jumpToSource}
                className="shrink-0 text-[10px] text-primary hover:underline"
              >
                Open →
              </button>
            </div>
          )}
          {excerpt && (
            <p className="text-muted-foreground leading-snug line-clamp-5">
              {excerpt}
            </p>
          )}
        </span>
      )}
    </span>
  )
}
