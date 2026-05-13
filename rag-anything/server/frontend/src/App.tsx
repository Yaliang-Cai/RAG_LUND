import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { AppShell } from '@/components/layout/AppShell'
import { useAppStore } from '@/store'

const ChatPage      = lazy(() => import('@/routes/ChatPage'))
const DocumentsPage = lazy(() => import('@/routes/DocumentsPage'))
const GraphPage     = lazy(() => import('@/routes/GraphPage'))
const JobsPage      = lazy(() => import('@/routes/JobsPage'))

const qc = new QueryClient()

export default function App() {
  const theme = useAppStore((s) => s.theme)
  useEffect(() => {
    if (theme === 'light') {
      document.documentElement.classList.add('light')
    } else {
      document.documentElement.classList.remove('light')
    }
  }, [theme])

  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Toaster richColors position="top-right" />
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Navigate to="/chat" replace />} />
            <Route path="/chat"      element={<Suspense fallback={null}><ChatPage /></Suspense>} />
            <Route path="/documents" element={<Suspense fallback={null}><DocumentsPage /></Suspense>} />
            <Route path="/graph"     element={<Suspense fallback={null}><GraphPage /></Suspense>} />
            <Route path="/jobs"      element={<Suspense fallback={null}><JobsPage /></Suspense>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
