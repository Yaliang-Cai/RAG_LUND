import { useNavigate } from 'react-router-dom'
import { useAppStore } from '@/store'
import { Badge } from '@/components/ui/badge'
import type { SourceNode } from '@/types'

export function CitationChip({ node }: { node: SourceNode }) {
  const navigate = useNavigate()
  const { setSelectedFile, setPendingPageNum } = useAppStore()

  function handleClick() {
    setSelectedFile(node.filename)
    if (node.page_num !== null) setPendingPageNum(node.page_num)
    navigate('/documents')
  }

  return (
    <Badge
      variant="outline"
      className="cursor-pointer hover:bg-accent text-xs gap-1"
      onClick={handleClick}
    >
      {node.filename}
      {node.page_num !== null && (
        <span className="text-muted-foreground">p.{node.page_num}</span>
      )}
    </Badge>
  )
}
