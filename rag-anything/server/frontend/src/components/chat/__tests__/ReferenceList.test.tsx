import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { ReferenceList } from '../ReferenceList'
import { useAppStore } from '@/store'
import type { ChunkRef } from '@/types'

const navigateMock = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => navigateMock }
})

function renderList(chunks?: ChunkRef[], messageId = 'm1') {
  return render(
    <MemoryRouter>
      <ReferenceList messageId={messageId} chunks={chunks} />
    </MemoryRouter>,
  )
}

describe('ReferenceList', () => {
  beforeEach(() => {
    navigateMock.mockReset()
    useAppStore.setState({
      selectedFileId: null,
      pendingPageNum: null,
      pendingChunkText: null,
      openReferences: {},
    })
  })

  it('renders nothing for empty chunks', () => {
    const { container } = renderList([])
    expect(container.firstChild).toBeNull()
  })

  it('renders nothing for undefined chunks', () => {
    const { container } = renderList(undefined)
    expect(container.firstChild).toBeNull()
  })

  it('renders collapsed by default and toggles open', () => {
    renderList([{ id: 'DC1', file_path: '/a/b/foo.pdf', content: 'hello world' }])
    expect(screen.getByText(/References \(1\)/)).toBeTruthy()
    expect(screen.queryByText('foo.pdf')).toBeNull()
    fireEvent.click(screen.getByText(/References \(1\)/))
    expect(screen.getByText('foo.pdf')).toBeTruthy()
  })

  it('shows page suffix only for chunks with page_idx (1-based display)', () => {
    renderList([
      { id: 'DC1', file_path: 'a.pdf', content: 'x', page_idx: 4 },
      { id: 'DC2', file_path: 'a.pdf', content: 'y' },
    ])
    fireEvent.click(screen.getByText(/References \(2\)/))
    // page_idx 4 (0-based) → "p.5" (1-based)
    expect(screen.getByText(/p\.5/)).toBeTruthy()
    expect(screen.queryAllByText(/p\./).length).toBe(1)
  })

  it('open state persists across remount via store', () => {
    const { unmount } = renderList(
      [{ id: 'DC1', file_path: 'a.pdf', content: 'hello' }],
      'msg-x',
    )
    fireEvent.click(screen.getByText(/References \(1\)/))
    expect(screen.getByText('a.pdf')).toBeTruthy()
    unmount()
    // Remount with same id — should still be open
    renderList([{ id: 'DC1', file_path: 'a.pdf', content: 'hello' }], 'msg-x')
    expect(screen.getByText('a.pdf')).toBeTruthy()
  })

  it('pdf chunk with page_idx → sets pendingPageNum (1-based), not pendingChunkText', () => {
    renderList([{ id: 'DC1', file_path: 'a.pdf', content: 'mm', page_idx: 7 }])
    fireEvent.click(screen.getByText(/References \(1\)/))
    fireEvent.click(screen.getByText('a.pdf'))
    const s = useAppStore.getState()
    expect(s.selectedFileId).toBe('a.pdf')
    // page_idx 7 (0-based) → page 8 (1-based)
    expect(s.pendingPageNum).toBe(8)
    expect(s.pendingChunkText).toBeNull()
    expect(navigateMock).toHaveBeenCalledWith('/documents')
  })

  it('pdf chunk without page → sets pendingChunkText, not pendingPageNum', () => {
    renderList([{ id: 'DC1', file_path: 'a.pdf', content: 'long passage of text' }])
    fireEvent.click(screen.getByText(/References \(1\)/))
    fireEvent.click(screen.getByText('a.pdf'))
    const s = useAppStore.getState()
    expect(s.selectedFileId).toBe('a.pdf')
    expect(s.pendingChunkText).toBe('long passage of text')
    expect(s.pendingPageNum).toBeNull()
    expect(navigateMock).toHaveBeenCalledWith('/documents')
  })

  it('non-pdf chunk (md) always sets pendingChunkText, even when page_idx is present', () => {
    renderList([{ id: 'DC1', file_path: 'notes.md', content: 'highlight me', page_idx: 0 }])
    fireEvent.click(screen.getByText(/References \(1\)/))
    fireEvent.click(screen.getByText('notes.md'))
    const s = useAppStore.getState()
    expect(s.selectedFileId).toBe('notes.md')
    expect(s.pendingChunkText).toBe('highlight me')
    expect(s.pendingPageNum).toBeNull()
  })

  it('truncates excerpt to 120 chars', () => {
    const long = 'a'.repeat(200)
    renderList([{ id: 'DC1', file_path: 'a.pdf', content: long }])
    fireEvent.click(screen.getByText(/References \(1\)/))
    const para = screen.getByText(/a+…$/)
    expect(para.textContent!.length).toBeLessThanOrEqual(121)
    expect(para.textContent!.endsWith('…')).toBe(true)
  })
})
