import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { GraphSearch } from '../GraphSearch'
import { searchGraph } from '@/api/graph'

vi.mock('@/api/graph', () => ({ searchGraph: vi.fn() }))
vi.mock('sonner', () => ({ toast: { info: vi.fn(), error: vi.fn() } }))

const searchGraphMock = vi.mocked(searchGraph)

function setup() {
  const onResult = vi.fn()
  render(<GraphSearch workspaceId="ws1" onResult={onResult} />)
  const input = screen.getByPlaceholderText('Search nodes...')
  fireEvent.change(input, { target: { value: 'alice' } })
  fireEvent.keyDown(input, { key: 'Enter' })
  return { onResult }
}

describe('GraphSearch', () => {
  beforeEach(() => {
    searchGraphMock.mockReset()
  })

  // Regression for "can't access property length, i.nodes is undefined":
  // backend returns { results: string[] }, not { nodes: [...] }.
  it('reads the real backend shape and highlights the first entity name', async () => {
    searchGraphMock.mockResolvedValue({ results: ['Alice', 'Acme'] })
    const { onResult } = setup()
    await waitFor(() => expect(onResult).toHaveBeenCalledWith('Alice'))
  })

  it('does not call onResult when there are no matches', async () => {
    searchGraphMock.mockResolvedValue({ results: [] })
    const { onResult } = setup()
    await waitFor(() => expect(searchGraphMock).toHaveBeenCalled())
    expect(onResult).not.toHaveBeenCalled()
  })
})
