import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useStreamQuery } from '@/hooks/useStreamQuery'
import * as queryApi from '@/api/query'

vi.mock('@/api/query')

function makeStream(lines: string[]): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      for (const line of lines) controller.enqueue(encoder.encode(line + '\n'))
      controller.close()
    },
  })
  return new Response(stream, { status: 200 })
}

describe('useStreamQuery', () => {
  beforeEach(() => vi.clearAllMocks())

  it('starts idle', () => {
    const { result } = renderHook(() => useStreamQuery())
    expect(result.current.status).toBe('idle')
    expect(result.current.answer).toBe('')
  })

  it('accumulates chunks and transitions to done', async () => {
    vi.mocked(queryApi.openQueryStream).mockResolvedValue(
      makeStream([
        'data: {"type":"meta","data":{},"metadata":{}}',
        'data: {"type":"chunk","text":"Hello"}',
        'data: {"type":"chunk","text":" world"}',
        'data: {"type":"done","graph":null,"source_nodes":[]}',
      ])
    )

    const { result } = renderHook(() => useStreamQuery())
    await act(async () => {
      await result.current.send({ workspace_id: 'ws1', query: 'test' })
    })

    expect(result.current.answer).toBe('Hello world')
    expect(result.current.status).toBe('done')
    expect(result.current.sourceNodes).toEqual([])
  })
})
