import { useState, useCallback, useRef } from 'react'
import { openQueryStream } from '@/api/query'
import type { QueryParams, SourceNode } from '@/types'

export type StreamStatus = 'idle' | 'streaming' | 'done' | 'error'

export function useStreamQuery() {
  const [answer, setAnswer] = useState('')
  const [reasoning, setReasoning] = useState('')
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [sourceNodes, setSourceNodes] = useState<SourceNode[]>([])
  const [metadata, setMetadata] = useState<Record<string, unknown>>({})
  const abortRef = useRef<AbortController | null>(null)

  const send = useCallback(async (params: QueryParams) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('streaming')
    setAnswer('')
    setReasoning('')
    setSourceNodes([])
    setMetadata({})

    try {
      const response = await openQueryStream(params)
      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            if (event.type === 'meta') {
              setMetadata((event.metadata as Record<string, unknown>) ?? {})
            } else if (event.type === 'chunk') {
              setAnswer((a) => a + (event.text as string))
            } else if (event.type === 'reasoning') {
              setReasoning((r) => r + (event.text as string))
            } else if (event.type === 'done') {
              setSourceNodes((event.source_nodes as SourceNode[]) ?? [])
              setStatus('done')
            } else if (event.type === 'error') {
              setStatus('error')
            }
          } catch {
            // malformed SSE line — skip
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setStatus('error')
    }
  }, [])

  const abort = useCallback(() => {
    abortRef.current?.abort()
    setStatus('idle')
  }, [])

  return { send, abort, answer, reasoning, status, sourceNodes, metadata }
}
