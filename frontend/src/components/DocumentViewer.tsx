import { useEffect, useRef, useState } from 'react'
import { Viewer, Worker } from '@react-pdf-viewer/core'
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout'
import { searchPlugin } from '@react-pdf-viewer/search'

import '@react-pdf-viewer/core/lib/styles/index.css'
import '@react-pdf-viewer/default-layout/lib/styles/index.css'
import '@react-pdf-viewer/search/lib/styles/index.css'

interface MatchedChunk {
  chunk_id: number
  text: string
  score: number
  section?: string
}

interface DocumentViewerProps {
  fileUrl: string
  fileType: 'pdf' | 'htm' | 'html'
  matchedChunks?: MatchedChunk[]
}

// Compact match navigation component
function MatchNavigation({
  matchedChunks,
  currentIndex,
  onJump
}: {
  matchedChunks: MatchedChunk[]
  currentIndex: number
  onJump: (index: number) => void
}) {
  if (!matchedChunks || matchedChunks.length === 0) return null

  return (
    <div className="flex-shrink-0 px-3 py-2 bg-[var(--color-accent-light)] border-b border-[var(--color-accent)] flex items-center gap-2 overflow-x-auto">
      <span className="text-xs font-medium text-[var(--color-accent)] whitespace-nowrap">
        {matchedChunks.length} match{matchedChunks.length !== 1 ? 'es' : ''}:
      </span>
      <div className="flex gap-1">
        {matchedChunks.map((chunk, idx) => (
          <button
            key={chunk.chunk_id}
            onClick={() => onJump(idx)}
            className={`px-2 py-1 text-xs font-medium transition-colors whitespace-nowrap ${
              currentIndex === idx
                ? 'bg-[var(--color-accent)] text-white'
                : 'bg-white text-[var(--color-accent)] hover:bg-[var(--color-accent)] hover:text-white border border-[var(--color-accent)]'
            }`}
            title={`Match ${idx + 1}: ${(chunk.score * 100).toFixed(0)}% relevance`}
          >
            #{idx + 1}
          </button>
        ))}
      </div>
    </div>
  )
}

// PDF Viewer with search highlighting
function PdfViewer({ fileUrl, matchedChunks }: { fileUrl: string; matchedChunks?: MatchedChunk[] }) {
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0)

  const searchPluginInstance = searchPlugin({
    keyword: matchedChunks && matchedChunks.length > 0
      ? matchedChunks[0].text.slice(0, 50)
      : '',
  })

  const { highlight } = searchPluginInstance

  const defaultLayoutPluginInstance = defaultLayoutPlugin({
    sidebarTabs: () => [], // Hide sidebar for more space
    toolbarPlugin: {
      searchPlugin: {
        keyword: matchedChunks?.[0]?.text.slice(0, 50) || '',
      },
    },
  })

  const handleJumpToMatch = (index: number) => {
    setCurrentMatchIndex(index)
    if (matchedChunks && matchedChunks[index]) {
      highlight(matchedChunks[index].text.slice(0, 50))
    }
  }

  return (
    <div className="h-full flex flex-col">
      <MatchNavigation
        matchedChunks={matchedChunks || []}
        currentIndex={currentMatchIndex}
        onJump={handleJumpToMatch}
      />
      <div className="flex-1 overflow-hidden">
        <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
          <Viewer
            fileUrl={fileUrl}
            plugins={[defaultLayoutPluginInstance, searchPluginInstance]}
          />
        </Worker>
      </div>
    </div>
  )
}

// HTML Viewer with text highlighting - ISOLATED from dark mode
function HtmlViewer({ fileUrl, matchedChunks }: {
  fileUrl: string
  matchedChunks?: MatchedChunk[]
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0)
  const [htmlContent, setHtmlContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch HTML content directly
  useEffect(() => {
    setLoading(true)
    setError(null)

    fetch(fileUrl)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load: ${res.status}`)
        return res.text()
      })
      .then(html => {
        setHtmlContent(html)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [fileUrl])

  // Apply highlights after content is loaded
  useEffect(() => {
    if (!htmlContent || !containerRef.current || !matchedChunks || matchedChunks.length === 0) return

    setTimeout(() => {
      const container = containerRef.current
      if (!container) return

      const contentRoot = container.querySelector('.html-content-isolated')
      if (!contentRoot) return

      matchedChunks.forEach((chunk, idx) => {
        highlightTextInElement(contentRoot as HTMLElement, chunk.text.slice(0, 80), idx)
      })
    }, 100)
  }, [htmlContent, matchedChunks])

  const highlightTextInElement = (element: HTMLElement, searchText: string, matchIndex: number) => {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT)
    const textNodes: Text[] = []

    while (walker.nextNode()) {
      textNodes.push(walker.currentNode as Text)
    }

    const searchSnippet = searchText.slice(0, 50).toLowerCase().replace(/\s+/g, ' ').trim()

    for (const node of textNodes) {
      const text = (node.textContent || '').toLowerCase().replace(/\s+/g, ' ')
      const index = text.indexOf(searchSnippet)

      if (index !== -1) {
        try {
          const span = document.createElement('mark')
          span.className = 'doc-highlight'
          span.dataset.matchIndex = String(matchIndex)
          span.id = `match-${matchIndex}`
          span.style.cssText = 'background-color: #fef08a !important; color: #000 !important; padding: 2px 4px; border-radius: 2px; scroll-margin: 100px;'

          const range = document.createRange()
          range.setStart(node, index)
          range.setEnd(node, Math.min(index + searchSnippet.length, node.textContent?.length || 0))
          range.surroundContents(span)
        } catch {
          // Skip if range crosses element boundaries
        }
        break
      }
    }
  }

  const scrollToMatch = (index: number) => {
    setCurrentMatchIndex(index)
    const target = document.getElementById(`match-${index}`)
    if (target) {
      document.querySelectorAll('.doc-highlight').forEach(el => {
        (el as HTMLElement).style.backgroundColor = '#fef08a';
        (el as HTMLElement).style.color = '#000'
      })
      target.style.backgroundColor = '#dc2626'
      target.style.color = 'white'
      target.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-[var(--color-bg-secondary)]">
        <div className="w-5 h-5 border-2 border-[var(--color-text-primary)] border-t-transparent rounded-full animate-spin" />
        <span className="ml-3 text-sm text-[var(--color-text-muted)]">Loading...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="p-3 border border-[var(--color-accent)] bg-[var(--color-accent-light)] text-[var(--color-accent)] text-sm">
          Error: {error}
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col" ref={containerRef}>
      <MatchNavigation
        matchedChunks={matchedChunks || []}
        currentIndex={currentMatchIndex}
        onJump={scrollToMatch}
      />
      {/*
        IMPORTANT: This container is ISOLATED from dark mode.
        It always has white background and black text to render HTML documents correctly.
      */}
      <div
        className="flex-1 overflow-auto html-content-isolated"
        dangerouslySetInnerHTML={{ __html: htmlContent || '' }}
        style={{
          padding: '16px',
          backgroundColor: '#ffffff',
          color: '#000000',
          // Reset any inherited dark mode styles
          colorScheme: 'light',
        }}
      />
    </div>
  )
}

// Main DocumentViewer component
export default function DocumentViewer({ fileUrl, fileType, matchedChunks }: DocumentViewerProps) {
  if (fileType === 'pdf') {
    return <PdfViewer fileUrl={fileUrl} matchedChunks={matchedChunks} />
  } else {
    return <HtmlViewer fileUrl={fileUrl} matchedChunks={matchedChunks} />
  }
}
