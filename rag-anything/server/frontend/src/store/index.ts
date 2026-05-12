import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AppStore {
  workspaceId: string
  setWorkspace: (id: string) => void
  theme: 'dark' | 'light'
  toggleTheme: () => void
  selectedFileId: string | null
  setSelectedFile: (id: string | null) => void
  pendingPageNum: number | null
  setPendingPageNum: (n: number | null) => void
  lastSeenJobStatuses: Record<string, string>
  setLastSeenJobStatuses: (statuses: Record<string, string>) => void
}

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      workspaceId: 'default',
      setWorkspace: (id) => set({ workspaceId: id }),
      theme: 'dark',
      toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
      selectedFileId: null,
      setSelectedFile: (id) => set({ selectedFileId: id }),
      pendingPageNum: null,
      setPendingPageNum: (n) => set({ pendingPageNum: n }),
      lastSeenJobStatuses: {},
      setLastSeenJobStatuses: (statuses) => set({ lastSeenJobStatuses: statuses }),
    }),
    {
      name: 'raganything-store',
      partialize: (s) => ({
        workspaceId: s.workspaceId,
        theme: s.theme,
        lastSeenJobStatuses: s.lastSeenJobStatuses,
      }),
    }
  )
)
