import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useQuery } from '@/hooks/useQuery'
import * as queryApi from '@/api/query'

vi.mock('@/api/query')

describe('useQuery (non-streaming postQuery)', () => {
  // status uses the non-streaming vocabulary now: idle → pending → done
  beforeEach(() => vi.clearAllMocks())

  it('starts idle', () => {
    const { result } = renderHook(() => useQuery())
    expect(result.current.status).toBe('idle')
    expect(result.current.answer).toBe('')
  })

  it('captures answer + source_nodes from a single response', async () => {
    vi.mocked(queryApi.postQuery).mockResolvedValue({
      answer: 'Hello world',
      data: {},
      metadata: {},
      source_nodes: [],
      graph: null,
    })

    const { result } = renderHook(() => useQuery())
    await act(async () => {
      await result.current.send({ workspace_id: 'ws1', query: 'test' })
    })

    expect(result.current.answer).toBe('Hello world')
    expect(result.current.status).toBe('done')
    expect(result.current.sourceNodes).toEqual([])
  })

  it('folds backend data into metadata.data for chunk extraction', async () => {
    const fakeTrace = {
      confidence: 0.91,
      grounded: true,
      profile: 'precise',
      retrieve_cycles_used: 2,
      check_cycles_used: 1,
    }
    vi.mocked(queryApi.postQuery).mockResolvedValue({
      answer: '72.3% accuracy',
      data: {},
      metadata: { agentic_trace: fakeTrace },
      source_nodes: [],
      graph: null,
    })

    const { result } = renderHook(() => useQuery())
    await act(async () => {
      await result.current.send({ workspace_id: 'ws1', query: 'test', mode: 'agentic' })
    })

    expect(result.current.answer).toBe('72.3% accuracy')
    expect(result.current.metadata).toEqual({ agentic_trace: fakeTrace, data: {} })
    expect(result.current.status).toBe('done')
  })
})
