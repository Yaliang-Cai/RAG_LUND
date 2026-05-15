import { useState, useCallback, useRef } from 'react'
import { openQueryStream } from '@/api/query'
import type { QueryParams, SourceNode } from '@/types'

export type StreamStatus = 'idle' | 'streaming' | 'done' | 'error'

const EMPTY_FALLBACK =
  'No relevant information was found in the knowledge base for this query.'

export interface StreamFinalSnapshot {
  answer: string
  reasoning: string
  metadata: Record<string, unknown>
  sourceNodes: SourceNode[]
  status: StreamStatus
}

export function useStreamQuery() {
  const [answer, setAnswer] = useState('')
  const [reasoning, setReasoning] = useState('')
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [sourceNodes, setSourceNodes] = useState<SourceNode[]>([])
  const [metadata, setMetadata] = useState<Record<string, unknown>>({})
  const abortRef = useRef<AbortController | null>(null)

  /**
   * Run a streaming query and return the final snapshot when the SSE stream
   * closes. The snapshot is accumulated synchronously inside this callback,
   * so callers don't depend on React having flushed all setState calls
   * before `await send(...)` resolves (which it usually hasn't).
   */
  const send = useCallback(async (params: QueryParams): Promise<StreamFinalSnapshot> => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('streaming')
    setAnswer('')
    setReasoning('')
    setSourceNodes([])
    setMetadata({})

    // Local accumulators — single source of truth for the returned snapshot.
    let finalAnswer = ''
    let finalReasoning = ''
    let finalMetadata: Record<string, unknown> = {}
    let finalSourceNodes: SourceNode[] = []
    let finalStatus: StreamStatus = 'streaming'

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
              finalMetadata = {
                ...((event.metadata as Record<string, unknown>) ?? {}),
                data: (event.data as Record<string, unknown>) ?? {},
              }
              setMetadata(finalMetadata)
            } else if (event.type === 'chunk') {
              const text = event.text as string
              finalAnswer += text
              setAnswer((a) => a + text)
            } else if (event.type === 'reasoning') {
              const text = event.text as string
              finalReasoning += text
              setReasoning((r) => r + text)
            } else if (event.type === 'done') {
              finalSourceNodes = (event.source_nodes as SourceNode[]) ?? []
              setSourceNodes(finalSourceNodes)
              if (finalAnswer.trim() === '') {
                finalAnswer = EMPTY_FALLBACK
                setAnswer(EMPTY_FALLBACK)
              }
              finalStatus = 'done'
              setStatus('done')
            } else if (event.type === 'error') {
              finalStatus = 'error'
              setStatus('error')
            }
          } catch {
            // malformed SSE line — skip
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        finalStatus = 'error'
        setStatus('error')
      }
    }

    return {
      answer: finalAnswer,
      reasoning: finalReasoning,
      metadata: finalMetadata,
      sourceNodes: finalSourceNodes,
      status: finalStatus,
    }
  }, [])

  const abort = useCallback(() => {
    abortRef.current?.abort()
    setStatus('idle')
  }, [])

  return { send, abort, answer, reasoning, status, sourceNodes, metadata }
}
