import { useCallback } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { GraphData, GraphNode } from '@/types'

const TYPE_COLORS: Record<string, string> = {
  concept: '#6366f1',
  person: '#22c55e',
  organization: '#f59e0b',
  location: '#06b6d4',
  other: '#64748b',
}

function nodeColor(node: GraphNode): string {
  return TYPE_COLORS[(node.type ?? '').toLowerCase()] ?? TYPE_COLORS.other
}

interface ForceGraphProps {
  data: GraphData
  onNodeClick?: (node: GraphNode) => void
  // Entity name to highlight. Matches on label (not id) so search works for
  // both Neo4j (id == name) and Postgres/AGE (id == numeric) backends.
  highlightLabel?: string | null
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyNode = any

export function ForceGraph({ data, onNodeClick, highlightLabel }: ForceGraphProps) {
  const handleNodeClick = useCallback((node: AnyNode) => {
    onNodeClick?.(node as GraphNode)
  }, [onNodeClick])

  const paintNode = useCallback((node: AnyNode, ctx: CanvasRenderingContext2D) => {
    const gn = node as GraphNode & { x: number; y: number }
    const color = nodeColor(gn)
    const isHighlighted = !!highlightLabel && gn.label === highlightLabel
    const radius = isHighlighted ? 8 : 5
    ctx.beginPath()
    ctx.arc(gn.x, gn.y, radius, 0, 2 * Math.PI)
    ctx.fillStyle = color
    ctx.fill()
    if (isHighlighted) {
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
    ctx.font = '3px sans-serif'
    ctx.fillStyle = '#94a3b8'
    ctx.textAlign = 'center'
    ctx.fillText(gn.label ?? gn.id, gn.x, gn.y + 8)
  }, [highlightLabel])

  const graphData = {
    nodes: data.nodes,
    links: data.edges.map((e) => ({ source: e.source, target: e.target, label: e.label })),
  }

  return (
    <ForceGraph2D
      graphData={graphData as AnyNode}
      nodeCanvasObject={paintNode}
      nodeCanvasObjectMode={() => 'replace'}
      linkLabel="label"
      linkColor={() => '#334155'}
      onNodeClick={handleNodeClick}
      backgroundColor="transparent"
    />
  )
}
