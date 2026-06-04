import { describe, expect, it, vi, beforeEach } from 'vitest'
import client from '../client'
import { getSubgraph } from '../graph'

vi.mock('../client', () => ({
  default: {
    get: vi.fn(),
  },
}))

const getMock = vi.mocked(client.get)

describe('graph api', () => {
  beforeEach(() => {
    getMock.mockReset()
  })

  it('uses backend subgraph query parameter names', async () => {
    getMock.mockResolvedValue({ data: { nodes: [], edges: [] } })

    await getSubgraph('ws1', 'Alice', 3, 25)

    expect(getMock).toHaveBeenCalledWith('/graph/ws1/subgraph', {
      params: { label: 'Alice', max_depth: 3, max_nodes: 25 },
    })
  })
})
