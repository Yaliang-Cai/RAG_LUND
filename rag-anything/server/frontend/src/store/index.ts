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
  // Chat history — in-memory only (NOT persisted). Reloading the page or
  // restarting the server starts a fresh conversation, so stale turns never
  // leak into a new session's context.
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
      // Bump when the persisted shape changes. v1 drops any chatMessages array left
      // in localStorage by older builds so upgraded clients don't rehydrate stale
      // chat history once (workspace/theme/job statuses are preserved by migrate).
      version: 1,
      migrate: (persisted) => {
        if (persisted && typeof persisted === 'object') {
          const { chatMessages: _drop, ...rest } = persisted as Record<string, unknown>
          return rest as never
        }
        return persisted as never
      },
      // chatMessages is intentionally NOT persisted: a reload / server restart
      // should begin a clean conversation (fresh session memory), not replay
      // old turns that would pollute the new session's context.
      partialize: (s) => ({
        workspaceId: s.workspaceId,
        theme: s.theme,
        lastSeenJobStatuses: s.lastSeenJobStatuses,
      }),
    }
  )
)
