import { NavLink } from 'react-router-dom'
import { Badge } from '@/components/ui/badge'
import { WorkspaceSwitcher } from './WorkspaceSwitcher'
import { ThemeToggle } from './ThemeToggle'
import { useJobs } from '@/hooks/useJobs'
import { useAppStore } from '@/store'
import { cn } from '@/lib/utils'

const NAV_ITEMS = [
  { to: '/chat', label: 'Chat' },
  { to: '/documents', label: 'Documents' },
  { to: '/graph', label: 'Graph' },
  { to: '/jobs', label: 'Jobs' },
]

export function TopNav() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { data: jobs = [] } = useJobs(workspaceId)
  const runningCount = jobs.filter((j) => j.status === 'running').length

  return (
    <header className="h-12 border-b border-border flex items-center px-4 gap-6 shrink-0 bg-background">
      <span className="text-sm font-semibold text-foreground">RAGAnything</span>
      <nav className="flex items-center gap-1">
        {NAV_ITEMS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-1 px-3 py-1.5 text-sm rounded-md transition-colors',
                isActive
                  ? 'text-primary border-b-2 border-primary font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              )
            }
          >
            {label}
            {label === 'Jobs' && runningCount > 0 && (
              <Badge variant="default" className="h-4 min-w-4 px-1 text-[10px]">
                {runningCount}
              </Badge>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="ml-auto flex items-center gap-2">
        <WorkspaceSwitcher />
        <ThemeToggle />
      </div>
    </header>
  )
}
