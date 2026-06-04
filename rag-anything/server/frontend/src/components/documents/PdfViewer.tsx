import { useState, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { Button } from '@/components/ui/button'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

interface PdfViewerProps {
  url: string
  /** When set to a number, the viewer jumps to that page and calls onPageSet
   *  to acknowledge. Pass null to mean "no pending jump". */
  initialPage?: number | null
  onPageSet?: () => void
}

export function PdfViewer({ url, initialPage, onPageSet }: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [currentPage, setCurrentPage] = useState<number>(
    typeof initialPage === 'number' && initialPage > 0 ? initialPage : 1,
  )
  const [scale, setScale] = useState(1.0)

  useEffect(() => {
    if (typeof initialPage === 'number' && initialPage > 0) {
      setCurrentPage(initialPage)
      onPageSet?.()
    }
  }, [initialPage]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 p-2 border-b border-border shrink-0">
        <Button
          variant="ghost" size="icon"
          onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
          disabled={currentPage <= 1}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="text-xs text-muted-foreground">
          {currentPage} / {numPages || '?'}
        </span>
        <Button
          variant="ghost" size="icon"
          onClick={() => setCurrentPage((p) => Math.min(numPages, p + 1))}
          disabled={currentPage >= numPages}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        <div className="ml-auto flex gap-1 items-center">
          <Button variant="ghost" size="icon" onClick={() => setScale((s) => Math.max(0.5, s - 0.25))}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs text-muted-foreground">{Math.round(scale * 100)}%</span>
          <Button variant="ghost" size="icon" onClick={() => setScale((s) => Math.min(2.5, s + 0.25))}>
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <div className="flex-1 overflow-auto flex justify-center bg-muted/20 p-4">
        <Document
          file={url}
          onLoadSuccess={({ numPages: n }) => setNumPages(n)}
          loading={<div className="text-sm text-muted-foreground mt-8">Loading PDF...</div>}
          error={<div className="text-sm text-destructive mt-8">Failed to load PDF</div>}
        >
          <Page pageNumber={currentPage} scale={scale} />
        </Document>
      </div>
    </div>
  )
}
