import { useState } from 'react'
import { useGraphOverview } from '@/hooks/useGraph'
import { useAppStore } from '@/store'
import { ForceGraph } from '@/components/graph/ForceGraph'
import { GraphSearch } from '@/components/graph/GraphSearch'
import { NodeSheet } from '@/components/graph/NodeSheet'
import { Button } from '@/components/ui/button'
import { RefreshCw } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import type { GraphNode } from '@/types'

export default function GraphPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const { data, isLoading } = useGraphOverview(workspaceId)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [highlightId, setHighlightId] = useState<string | null>(null)
  const qc = useQueryClient()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
        Loading graph...
      </div>
    )
  }

  if (!data || data.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
        No graph data. Ingest documents first.
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border shrink-0">
        <GraphSearch workspaceId={workspaceId} onResult={setHighlightId} />
        <Button
          variant="ghost"
          size="icon-sm"
          title="Refresh"
          onClick={() => qc.invalidateQueries({ queryKey: ['graph', 'overview', workspaceId] })}
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
        <span className="text-xs text-muted-foreground ml-auto">
          {data.nodes.length} nodes · {data.edges.length} edges
        </span>
      </div>
      <div className="flex-1 relative">
        <ForceGraph
          data={data}
          onNodeClick={setSelectedNode}
          highlightNodeId={highlightId}
        />
      </div>
      <NodeSheet node={selectedNode} onClose={() => setSelectedNode(null)} />
    </div>
  )
}
