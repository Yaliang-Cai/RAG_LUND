import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { useAppStore } from '@/store'
import { cn } from '@/lib/utils'
import type { ChunkRef } from '@/types'

interface ReferenceListProps {
  chunks?: ChunkRef[]
}

function basename(p: string): string {
  return p.replace(/\\/g, '/').split('/').filter(Boolean).pop() ?? p
}

function chunkLabel(c: ChunkRef, idx: number): string {
  if (c.id != null) return String(c.id)
  if (c.reference_id != null) return String(c.reference_id)
  return String(idx + 1)
}

function chunkFilename(c: ChunkRef): string {
  return basename(c.file_path ?? c.filename ?? 'unknown')
}

function chunkExcerpt(c: ChunkRef): string {
  const raw = (c.content ?? c.text ?? c.excerpt ?? '').replace(/\s+/g, ' ').trim()
  return raw.length > 120 ? raw.slice(0, 120) + '…' : raw
}

function chunkPage(c: ChunkRef): number | null {
  const p = c.page_idx ?? c.page_num
  return typeof p === 'number' ? p : null
}

export function ReferenceList({ chunks }: ReferenceListProps) {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  const { setSelectedFile, setPendingPageNum, setPendingChunkText } = useAppStore()

  if (!chunks || chunks.length === 0) return null

  function jumpTo(c: ChunkRef) {
    const filename = chunkFilename(c)
    setSelectedFile(filename)
    const page = chunkPage(c)
    if (page !== null) {
      setPendingPageNum(page)
      setPendingChunkText(null)
    } else {
      setPendingChunkText(c.content ?? c.text ?? '')
      setPendingPageNum(null)
    }
    navigate('/documents')
  }

  return (
    <div className="mt-1 rounded-lg border border-border bg-secondary/50 text-xs overflow-hidden max-w-[80%]">
      <button
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span>引用 ({chunks.length})</span>
      </button>
      {open && (
        <ul className="divide-y divide-border">
          {chunks.map((c, i) => {
            const page = chunkPage(c)
            return (
              <li
                key={c.id ?? c.reference_id ?? i}
                role="button"
                tabIndex={0}
                onClick={() => jumpTo(c)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    jumpTo(c)
                  }
                }}
                className={cn(
                  'cursor-pointer px-3 py-2 hover:bg-accent/60 transition-colors',
                )}
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] text-muted-foreground shrink-0">
                    [{chunkLabel(c, i)}]
                  </span>
                  <span className="truncate font-medium">{chunkFilename(c)}</span>
                  {page !== null && (
                    <span className="text-muted-foreground shrink-0">· 第 {page} 页</span>
                  )}
                </div>
                <p className="mt-1 text-muted-foreground line-clamp-2">{chunkExcerpt(c)}</p>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
