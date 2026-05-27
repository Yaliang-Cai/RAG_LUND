import { useState } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Search } from 'lucide-react'
import { searchGraph } from '@/api/graph'
import { toast } from 'sonner'

interface GraphSearchProps {
  workspaceId: string
  // Receives the matched entity name, which equals the graph node's label.
  onResult: (label: string) => void
}

export function GraphSearch({ workspaceId, onResult }: GraphSearchProps) {
  const [query, setQuery] = useState('')

  async function handleSearch() {
    if (!query.trim()) return
    try {
      const result = await searchGraph(workspaceId, query)
      const names = result.results ?? []
      if (names.length > 0) onResult(names[0])
      else toast.info('No matching nodes found')
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  return (
    <div className="flex gap-2">
      <Input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        placeholder="Search nodes..."
        className="h-8 text-xs"
      />
      <Button variant="outline" size="icon-sm" onClick={handleSearch}>
        <Search className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}
