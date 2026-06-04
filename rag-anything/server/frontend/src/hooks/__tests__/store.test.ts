import { describe, it, expect, beforeEach } from 'vitest'
import { useAppStore } from '@/store'
import { act } from '@testing-library/react'

describe('AppStore', () => {
  beforeEach(() => {
    useAppStore.setState({
      workspaceId: 'default',
      theme: 'dark',
      selectedFileId: null,
      pendingPageNum: null,
      pendingChunkText: null,
      openReferences: {},
      lastSeenJobStatuses: {},
    })
  })

  it('sets workspaceId', () => {
    act(() => useAppStore.getState().setWorkspace('ws2'))
    expect(useAppStore.getState().workspaceId).toBe('ws2')
  })

  it('toggles theme', () => {
    act(() => useAppStore.getState().toggleTheme())
    expect(useAppStore.getState().theme).toBe('light')
    act(() => useAppStore.getState().toggleTheme())
    expect(useAppStore.getState().theme).toBe('dark')
  })

  it('sets pendingPageNum and clears it', () => {
    act(() => useAppStore.getState().setPendingPageNum(5))
    expect(useAppStore.getState().pendingPageNum).toBe(5)
    act(() => useAppStore.getState().setPendingPageNum(null))
    expect(useAppStore.getState().pendingPageNum).toBeNull()
  })

  it('sets pendingChunkText and clears it', () => {
    act(() => useAppStore.getState().setPendingChunkText('hello world'))
    expect(useAppStore.getState().pendingChunkText).toBe('hello world')
    act(() => useAppStore.getState().setPendingChunkText(null))
    expect(useAppStore.getState().pendingChunkText).toBeNull()
  })

  it('toggles openReferences per message id', () => {
    act(() => useAppStore.getState().toggleOpenReference('m1'))
    expect(useAppStore.getState().openReferences['m1']).toBe(true)
    act(() => useAppStore.getState().toggleOpenReference('m1'))
    expect(useAppStore.getState().openReferences['m1']).toBe(false)
    act(() => useAppStore.getState().toggleOpenReference('m2'))
    expect(useAppStore.getState().openReferences['m2']).toBe(true)
    expect(useAppStore.getState().openReferences['m1']).toBe(false)
  })
})
