import { useState, useEffect } from 'react'
import { useFiles } from '@/hooks/useFiles'
import { useWorkspaces } from '@/hooks/useWorkspaces'
import { useAppStore } from '@/store'
import { getFileContent, getUploadUrl } from '@/api/files'
import { FileList } from '@/components/documents/FileList'
import { FileUpload } from '@/components/documents/FileUpload'
import { MarkdownViewer } from '@/components/documents/MarkdownViewer'
import { PdfViewer } from '@/components/documents/PdfViewer'
import { Button } from '@/components/ui/button'

export default function DocumentsPage() {
  const workspaceId = useAppStore((s) => s.workspaceId)
  const {
    pendingPageNum,
    setPendingPageNum,
    pendingChunkText,
    setPendingChunkText,
    selectedFileId,
    setSelectedFile,
  } = useAppStore()
  const { data: files = [] } = useFiles(workspaceId)
  const { data: workspaces = [] } = useWorkspaces()
  const workspace = workspaces.find((w) => w.workspace_id === workspaceId)
  const frozen = workspace?.frozen ?? false

  const [tab, setTab] = useState<'markdown' | 'pdf'>('markdown')
  const [mdContent, setMdContent] = useState('')

  const selectedFilename = selectedFileId ?? files[0]?.filename ?? null

  useEffect(() => {
    if (!selectedFilename) return
    getFileContent(workspaceId, selectedFilename)
      .then(setMdContent)
      .catch(() => setMdContent('Failed to load content.'))
  }, [selectedFilename, workspaceId])

  // Citation jump: switch to PDF tab when pendingPageNum is set
  useEffect(() => {
    if (pendingPageNum !== null) setTab('pdf')
  }, [pendingPageNum])

  // Chunk jump for text chunks: force markdown tab when pendingChunkText is set
  useEffect(() => {
    if (pendingChunkText !== null) setTab('markdown')
  }, [pendingChunkText])

  const isPdf = selectedFilename?.toLowerCase().endsWith('.pdf') ?? false

  return (
    <div className="flex h-full">
      <div className="w-64 shrink-0 flex flex-col">
        <FileList
          files={files}
          workspace={workspace}
          selectedFile={selectedFilename}
          onSelect={(f) => { setSelectedFile(f); setTab('markdown') }}
        />
        <FileUpload disabled={frozen} />
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        {selectedFilename && (
          <div className="flex items-center gap-2 px-4 py-2 border-b border-border shrink-0">
            <Button
              variant={tab === 'markdown' ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => setTab('markdown')}
            >
              Markdown
            </Button>
            {isPdf && (
              <Button
                variant={tab === 'pdf' ? 'secondary' : 'ghost'}
                size="sm"
                onClick={() => setTab('pdf')}
              >
                PDF
              </Button>
            )}
          </div>
        )}
        <div className="flex-1 min-h-0">
          {!selectedFilename && (
            <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
              Select a file to preview
            </div>
          )}
          {selectedFilename && tab === 'markdown' && (
            <MarkdownViewer
              content={mdContent}
              scrollToText={pendingChunkText}
              onScrollComplete={() => setPendingChunkText(null)}
            />
          )}
          {selectedFilename && tab === 'pdf' && isPdf && (
            <PdfViewer
              url={getUploadUrl(workspaceId, selectedFilename)}
              initialPage={pendingPageNum}
              onPageSet={() => setPendingPageNum(null)}
            />
          )}
        </div>
      </div>
    </div>
  )
}
