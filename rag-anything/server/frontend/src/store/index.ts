import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Message } from '@/components/chat/MessageBubble'

interface AppStore {
  workspaceId: string
  setWorkspace: (id: string) => void
  theme: 'dark' | 'light'
  toggleTheme: () => void
  selectedFileId: string | null
  setSelectedFile: (id: string | null) => void
  pendingPageNum: number | null
  setPendingPageNum: (n: number | null) => void
  pendingChunkText: string | null
  setPendingChunkText: (t: string | null) => void
  // Per-message ReferenceList open state — in-memory only, survives navigation
  openReferences: Record<string, boolean>
  toggleOpenReference: (id: string) => void
  lastSeenJobStatuses: Record<string, string>
  setLastSeenJobStatuses: (statuses: Record<string, string>) => void
  // Chat history — persists across navigation (not persisted to localStorage)
  chatMessages: Message[]
  setChatMessages: (msgs: Message[] | ((prev: Message[]) => Message[])) => void
  clearChatMessages: () => void
}

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      workspaceId: 'default',
      setWorkspace: (id) => set({ workspaceId: id }),
      theme: 'light',
      toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
      selectedFileId: null,
      setSelectedFile: (id) => set({ selectedFileId: id }),
      pendingPageNum: null,
      setPendingPageNum: (n) => set({ pendingPageNum: n }),
      pendingChunkText: null,
      setPendingChunkText: (t) => set({ pendingChunkText: t }),
      openReferences: {},
      toggleOpenReference: (id) =>
        set((s) => ({
          openReferences: { ...s.openReferences, [id]: !s.openReferences[id] },
        })),
      lastSeenJobStatuses: {},
      setLastSeenJobStatuses: (statuses) => set({ lastSeenJobStatuses: statuses }),
      chatMessages: [],
      setChatMessages: (msgs) =>
        set((s) => ({
          chatMessages: typeof msgs === 'function' ? msgs(s.chatMessages) : msgs,
        })),
      clearChatMessages: () => set({ chatMessages: [] }),
    }),
    {
      name: 'raganything-store',
      partialize: (s) => ({
        workspaceId: s.workspaceId,
        theme: s.theme,
        lastSeenJobStatuses: s.lastSeenJobStatuses,
        // chatMessages intentionally NOT persisted (cleared on page reload)
      }),
    }
  )
)
