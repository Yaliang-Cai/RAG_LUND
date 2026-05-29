import { useState, useCallback, useRef } from 'react'
import { postQuery } from '@/api/query'
import type { QueryParams, SourceNode } from '@/types'

export type QueryStatus = 'idle' | 'pending' | 'done' | 'error'

const EMPTY_FALLBACK =
  'No relevant information was found in the knowledge base for this query.'

export interface QuerySnapshot {
  answer: string
  metadata: Record<string, unknown>
  sourceNodes: SourceNode[]
  status: QueryStatus
}

/**
 * Drives a non-streaming POST /query call.
 *
 * Single round-trip — ``status`` transitions ``idle → pending → done``.
 * Going through the non-streaming path is what lets LightRAG's
 * ``llm_response_cache`` write its result, so identical queries hit the
 * cache on the next call (see api/query.ts).
 */
export function useQuery() {
  const [answer, setAnswer] = useState('')
  const [status, setStatus] = useState<QueryStatus>('idle')
  const [sourceNodes, setSourceNodes] = useState<SourceNode[]>([])
  const [metadata, setMetadata] = useState<Record<string, unknown>>({})
  const abortRef = useRef<AbortController | null>(null)

  const send = useCallback(async (params: QueryParams): Promise<QuerySnapshot> => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('pending')
    setAnswer('')
    setSourceNodes([])
    setMetadata({})

    let finalAnswer = ''
    let finalMetadata: Record<string, unknown> = {}
    let finalSourceNodes: SourceNode[] = []
    let finalStatus: QueryStatus = 'pending'

    try {
      const resp = await postQuery(params)
      finalAnswer = resp.answer && resp.answer.trim() !== '' ? resp.answer : EMPTY_FALLBACK
      finalMetadata = {
        ...(resp.metadata ?? {}),
        data: resp.data ?? {},
      }
      finalSourceNodes = resp.source_nodes ?? []
      finalStatus = 'done'

      setAnswer(finalAnswer)
      setMetadata(finalMetadata)
      setSourceNodes(finalSourceNodes)
      setStatus('done')
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        finalStatus = 'error'
        setStatus('error')
      }
    }

    return {
      answer: finalAnswer,
      metadata: finalMetadata,
      sourceNodes: finalSourceNodes,
      status: finalStatus,
    }
  }, [])

  const abort = useCallback(() => {
    abortRef.current?.abort()
    setStatus('idle')
  }, [])

  return { send, abort, answer, status, sourceNodes, metadata }
}
