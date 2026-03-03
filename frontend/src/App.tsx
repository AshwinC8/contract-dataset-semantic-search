import { useState, useRef, useEffect, useCallback } from 'react'
import { QueryClient, QueryClientProvider, useMutation, useQuery } from '@tanstack/react-query'
import axios from 'axios'
import DocumentViewer from './components/DocumentViewer'

// =============================================================================
// CONFIGURATION
// =============================================================================

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const BOOKMARKS_KEY = 'contract-archive-bookmarks'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000,
      retry: 1,
    },
  },
})

// =============================================================================
// TYPES
// =============================================================================

interface HealthResponse {
  status: string
  qdrant_connected: boolean
  total_vectors: number
}

interface ChunkResult {
  chunk_id: number
  text: string
  score: number
  section?: string
}

interface ContractResult {
  contract_id: string
  year: number
  quarter: string
  top_score: number
  chunks: ChunkResult[]
}

interface SearchResponse {
  query: string
  total_contracts: number
  page: number
  per_page: number
  total_pages: number
  min_score: number
  results: ContractResult[]
  search_time_ms: number
}

interface SearchParams {
  query: string
  year_start?: number
  year_end?: number
  quarters?: string[]
  min_score: number
  page: number
  per_page: number
}

interface ContractSummary {
  contract_id: string
  year: number
  quarter: string
  chunk_count: number
  file_path?: string
  file_type?: string
}

interface ContractsListResponse {
  total: number
  page: number
  per_page: number
  contracts: ContractSummary[]
}

interface ContractChunk {
  chunk_id: number
  text: string
  section?: string
}

interface ContractDocument {
  contract_id: string
  year: number
  quarter: string
  total_chunks: number
  file_path?: string
  file_type?: string
  chunks: ContractChunk[]
}

interface BookmarkedContract {
  contract_id: string
  year: number
  quarter: string
  file_type?: string
  bookmarked_at: number
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

const api = {
  health: async (): Promise<HealthResponse> => {
    const { data } = await axios.get(`${API_URL}/health`)
    return data
  },

  search: async (params: SearchParams): Promise<SearchResponse> => {
    const { data } = await axios.post(`${API_URL}/search`, params)
    return data
  },

  contracts: async (page: number, perPage: number, year?: number): Promise<ContractsListResponse> => {
    const params = new URLSearchParams({ page: String(page), per_page: String(perPage) })
    if (year) params.append('year', String(year))
    const { data } = await axios.get(`${API_URL}/contracts?${params}`)
    return data
  },

  contract: async (contractId: string): Promise<ContractDocument> => {
    const { data } = await axios.get(`${API_URL}/contract/${encodeURIComponent(contractId)}`)
    return data
  },
}

// =============================================================================
// BOOKMARK HELPERS
// =============================================================================

const loadBookmarks = (): BookmarkedContract[] => {
  try {
    const stored = localStorage.getItem(BOOKMARKS_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

const saveBookmarks = (bookmarks: BookmarkedContract[]) => {
  localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(bookmarks))
}

// =============================================================================
// ICONS - Minimal, geometric
// =============================================================================

const Icons = {
  Search: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <circle cx="11" cy="11" r="7" />
      <path d="M21 21l-4.35-4.35" />
    </svg>
  ),
  Grid: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <rect x="3" y="3" width="7" height="7" />
      <rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" />
    </svg>
  ),
  Close: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  ),
  ChevronLeft: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path d="M15 18l-6-6 6-6" />
    </svg>
  ),
  ChevronRight: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path d="M9 6l6 6-6 6" />
    </svg>
  ),
  Document: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M16 13H8M16 17H8M10 9H8" />
    </svg>
  ),
  Download: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
    </svg>
  ),
  ArrowDown: ({ className = "w-3 h-3" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path d="M19 14l-7 7m0 0l-7-7m7 7V3" />
    </svg>
  ),
  Sun: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="5" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
    </svg>
  ),
  Moon: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
    </svg>
  ),
  Bookmark: ({ className = "w-4 h-4", filled = false }: { className?: string; filled?: boolean }) => (
    <svg className={className} fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z" />
    </svg>
  ),
  Trash: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14zM10 11v6M14 11v6" />
    </svg>
  ),
  Check: ({ className = "w-4 h-4" }: { className?: string }) => (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  ),
}

// =============================================================================
// MAIN APP
// =============================================================================

function ContractSearchApp() {
  // State
  const [activeTab, setActiveTab] = useState<'search' | 'browse' | 'bookmarks'>('search')
  const [darkMode, setDarkMode] = useState(false)
  const [query, setQuery] = useState('')
  const [yearStart, setYearStart] = useState<number | ''>('')
  const [yearEnd, setYearEnd] = useState<number | ''>('')
  const [selectedQuarters, setSelectedQuarters] = useState<string[]>([])
  const [minScore, setMinScore] = useState(0.5)
  const [searchPage, setSearchPage] = useState(1)
  const [perPage] = useState(20)
  const [selectedContract, setSelectedContract] = useState<ContractResult | ContractSummary | null>(null)
  const [browsePage, setBrowsePage] = useState(1)
  const [browseYear, setBrowseYear] = useState<number | ''>('')
  const [viewMode, setViewMode] = useState<'original' | 'chunks'>('original')

  // Bookmark state
  const [bookmarks, setBookmarks] = useState<BookmarkedContract[]>(loadBookmarks)
  const [selectedBookmarks, setSelectedBookmarks] = useState<Set<string>>(new Set())
  const [isDownloading, setIsDownloading] = useState(false)

  const chunkRefs = useRef<Map<number, HTMLDivElement>>(new Map())

  // Apply dark mode
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  // Clear refs when contract changes
  useEffect(() => {
    chunkRefs.current.clear()
  }, [selectedContract])

  // Save bookmarks to localStorage whenever they change
  useEffect(() => {
    saveBookmarks(bookmarks)
  }, [bookmarks])

  // Queries
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30000,
  })

  const searchMutation = useMutation({
    mutationFn: api.search,
  })

  const browseQuery = useQuery({
    queryKey: ['contracts', browsePage, browseYear],
    queryFn: () => api.contracts(browsePage, 20, browseYear || undefined),
    enabled: activeTab === 'browse',
  })

  const contractQuery = useQuery({
    queryKey: ['contract', selectedContract && 'contract_id' in selectedContract ? selectedContract.contract_id : null],
    queryFn: () => api.contract((selectedContract as ContractSummary).contract_id),
    enabled: !!selectedContract,
  })

  // Handlers
  const handleSearch = (e?: React.FormEvent, page: number = 1) => {
    if (e) e.preventDefault()
    if (!query.trim()) return
    setSelectedContract(null)
    setSearchPage(page)
    searchMutation.mutate({
      query: query.trim(),
      year_start: yearStart || undefined,
      year_end: yearEnd || undefined,
      quarters: selectedQuarters.length > 0 ? selectedQuarters : undefined,
      min_score: minScore,
      page: page,
      per_page: perPage,
    })
  }

  const handleSearchPageChange = (newPage: number) => {
    handleSearch(undefined, newPage)
  }

  const toggleQuarter = (q: string) => {
    setSelectedQuarters(prev =>
      prev.includes(q) ? prev.filter(x => x !== q) : [...prev, q]
    )
  }

  const scrollToChunk = (chunkId: number) => {
    const element = chunkRefs.current.get(chunkId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }

  // Bookmark handlers
  const isBookmarked = useCallback((contractId: string) => {
    return bookmarks.some(b => b.contract_id === contractId)
  }, [bookmarks])

  const toggleBookmark = useCallback((contract: ContractResult | ContractSummary | BookmarkedContract) => {
    const contractId = contract.contract_id
    if (isBookmarked(contractId)) {
      setBookmarks(prev => prev.filter(b => b.contract_id !== contractId))
      setSelectedBookmarks(prev => {
        const next = new Set(prev)
        next.delete(contractId)
        return next
      })
    } else {
      const newBookmark: BookmarkedContract = {
        contract_id: contractId,
        year: contract.year,
        quarter: contract.quarter,
        file_type: 'file_type' in contract ? contract.file_type : undefined,
        bookmarked_at: Date.now(),
      }
      setBookmarks(prev => [...prev, newBookmark])
    }
  }, [isBookmarked])

  const toggleSelectBookmark = (contractId: string) => {
    setSelectedBookmarks(prev => {
      const next = new Set(prev)
      if (next.has(contractId)) {
        next.delete(contractId)
      } else {
        next.add(contractId)
      }
      return next
    })
  }

  const selectAllBookmarks = () => {
    if (selectedBookmarks.size === bookmarks.length) {
      setSelectedBookmarks(new Set())
    } else {
      setSelectedBookmarks(new Set(bookmarks.map(b => b.contract_id)))
    }
  }

  const removeSelectedBookmarks = () => {
    setBookmarks(prev => prev.filter(b => !selectedBookmarks.has(b.contract_id)))
    setSelectedBookmarks(new Set())
  }

  const downloadSelectedBookmarks = async () => {
    if (selectedBookmarks.size === 0) return

    setIsDownloading(true)
    const toDownload = bookmarks.filter(b => selectedBookmarks.has(b.contract_id))

    for (const bookmark of toDownload) {
      try {
        const url = `${API_URL}/contract/${encodeURIComponent(bookmark.contract_id)}/file?download=true`
        const link = document.createElement('a')
        link.href = url
        link.download = `${bookmark.contract_id}.${bookmark.file_type || 'pdf'}`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        // Small delay between downloads
        await new Promise(resolve => setTimeout(resolve, 500))
      } catch (err) {
        console.error(`Failed to download ${bookmark.contract_id}:`, err)
      }
    }

    setIsDownloading(false)
  }

  // Get matched chunk IDs for highlighting
  const matchedChunkIds = new Set(
    selectedContract && 'chunks' in selectedContract && Array.isArray(selectedContract.chunks)
      ? (selectedContract as ContractResult).chunks.map(c => c.chunk_id)
      : []
  )

  const getMatchScore = (chunkId: number): number | undefined => {
    if (selectedContract && 'chunks' in selectedContract && Array.isArray(selectedContract.chunks)) {
      const chunk = (selectedContract as ContractResult).chunks.find(c => c.chunk_id === chunkId)
      return chunk?.score
    }
    return undefined
  }

  return (
    <div className="h-screen flex flex-col bg-[var(--color-bg-primary)] overflow-hidden">
      {/* ========== COMPACT HEADER ========== */}
      <header className="flex-shrink-0 border-b border-[var(--color-border)] bg-[var(--color-bg-primary)]">
        <div className="flex items-center justify-between px-4 h-12">
          {/* Logo + Tabs combined */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 bg-[var(--color-text-primary)] flex items-center justify-center">
                <Icons.Document className="w-3 h-3 text-[var(--color-bg-primary)]" />
              </div>
              <span className="text-display text-xs tracking-tight hidden sm:inline">CONTRACT ARCHIVE</span>
            </div>

            {/* Tabs */}
            <div className="flex">
              <button
                onClick={() => { setActiveTab('search'); setSelectedContract(null) }}
                className={`tab-swiss flex items-center gap-1.5 text-xs py-2 ${activeTab === 'search' ? 'tab-swiss-active' : ''}`}
              >
                <Icons.Search className="w-3.5 h-3.5" />
                Search
              </button>
              <button
                onClick={() => { setActiveTab('browse'); setSelectedContract(null) }}
                className={`tab-swiss flex items-center gap-1.5 text-xs py-2 ${activeTab === 'browse' ? 'tab-swiss-active' : ''}`}
              >
                <Icons.Grid className="w-3.5 h-3.5" />
                Browse
              </button>
              <button
                onClick={() => { setActiveTab('bookmarks'); setSelectedContract(null) }}
                className={`tab-swiss flex items-center gap-1.5 text-xs py-2 ${activeTab === 'bookmarks' ? 'tab-swiss-active' : ''}`}
              >
                <Icons.Bookmark className="w-3.5 h-3.5" filled={bookmarks.length > 0} />
                Saved
                {bookmarks.length > 0 && (
                  <span className="ml-1 px-1.5 py-0.5 text-[10px] bg-[var(--color-accent)] text-white rounded-full">
                    {bookmarks.length}
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Status & Controls */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <span className={`status-indicator ${healthQuery.data?.qdrant_connected ? 'status-online' : 'status-offline'}`} />
              <span className="text-mono text-xs text-[var(--color-text-muted)]">
                {healthQuery.data?.total_vectors?.toLocaleString() || '—'}
              </span>
            </div>
            <button onClick={() => setDarkMode(!darkMode)} className="btn-ghost p-1.5">
              {darkMode ? <Icons.Sun className="w-3.5 h-3.5" /> : <Icons.Moon className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </header>

      {/* ========== MAIN CONTENT ========== */}
      <div className="flex-1 flex min-h-0">
        {/* Left Panel */}
        <div className={`flex flex-col min-h-0 border-r border-[var(--color-border)] ${
          selectedContract
            ? 'w-80 min-w-[320px] max-w-[320px]'
            : 'w-96 min-w-[384px] max-w-[384px]'
        } transition-all duration-200`}>

          {/* Search Tab */}
          {activeTab === 'search' && (
            <>
              {/* Compact Search Form */}
              <form onSubmit={handleSearch} className="flex-shrink-0 p-3 border-b border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
                <div className="space-y-2">
                  <div className="flex gap-1.5">
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search contracts..."
                      className="input-swiss flex-1 text-sm py-2"
                    />
                    <button
                      type="submit"
                      disabled={searchMutation.isPending || !query.trim()}
                      className="btn-primary px-3"
                    >
                      {searchMutation.isPending ? (
                        <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <Icons.Search className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>

                  <div className="space-y-2 text-xs">
                    {/* Year Range */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-label w-8">Year</span>
                      <select
                        value={yearStart}
                        onChange={(e) => setYearStart(e.target.value ? parseInt(e.target.value) : '')}
                        className="input-swiss w-[72px] text-xs py-1 px-2 appearance-none cursor-pointer"
                      >
                        <option value="">From</option>
                        {Array.from({ length: 15 }, (_, i) => 2010 + i).map(year => (
                          <option key={year} value={year}>{year}</option>
                        ))}
                      </select>
                      <span className="text-[var(--color-text-muted)]">–</span>
                      <select
                        value={yearEnd}
                        onChange={(e) => setYearEnd(e.target.value ? parseInt(e.target.value) : '')}
                        className="input-swiss w-[72px] text-xs py-1 px-2 appearance-none cursor-pointer"
                      >
                        <option value="">To</option>
                        {Array.from({ length: 15 }, (_, i) => 2010 + i).map(year => (
                          <option key={year} value={year}>{year}</option>
                        ))}
                      </select>
                    </div>

                    {/* Quarter Selection */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-label w-8">Q</span>
                      <div className="flex">
                        {['Q1', 'Q2', 'Q3', 'Q4'].map((q) => (
                          <button
                            key={q}
                            type="button"
                            onClick={() => toggleQuarter(q)}
                            className={`px-2 py-1 text-xs font-medium border-y border-r first:border-l transition-colors ${
                              selectedQuarters.includes(q)
                                ? 'bg-[var(--color-text-primary)] text-[var(--color-bg-primary)] border-[var(--color-text-primary)]'
                                : 'bg-transparent text-[var(--color-text-muted)] border-[var(--color-border)] hover:text-[var(--color-text-primary)]'
                            }`}
                          >
                            {q.slice(1)}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Minimum Score Filter */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-label w-8">Min</span>
                      <div className="flex">
                        {[0.5, 0.55, 0.6, 0.7].map((score) => (
                          <button
                            key={score}
                            type="button"
                            onClick={() => setMinScore(score)}
                            className={`px-2 py-1 text-xs font-medium border-y border-r first:border-l transition-colors ${
                              minScore === score
                                ? 'bg-[var(--color-text-primary)] text-[var(--color-bg-primary)] border-[var(--color-text-primary)]'
                                : 'bg-transparent text-[var(--color-text-muted)] border-[var(--color-border)] hover:text-[var(--color-text-primary)]'
                            }`}
                          >
                            {(score * 100).toFixed(0)}%
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </form>

              {/* Search Results */}
              <div className="flex-1 overflow-y-auto min-h-0">
                {searchMutation.isPending && (
                  <div className="p-3 space-y-2">
                    {[...Array(5)].map((_, i) => (
                      <div key={i} className="card-swiss p-3">
                        <div className="skeleton h-3 w-3/4 mb-2" />
                        <div className="skeleton h-2 w-1/2" />
                      </div>
                    ))}
                  </div>
                )}

                {searchMutation.data && (
                  <div className="animate-fade-in">
                    <div className="px-3 py-2 text-xs text-[var(--color-text-muted)] border-b border-[var(--color-border)] bg-[var(--color-bg-tertiary)] flex justify-between items-center">
                      <span>
                        <strong className="text-[var(--color-text-primary)]">{searchMutation.data.total_contracts}</strong> results
                        <span className="ml-1 text-[10px]">(≥{(searchMutation.data.min_score * 100).toFixed(0)}%)</span>
                      </span>
                      <div className="flex items-center gap-2">
                        {searchMutation.data.total_pages > 1 && (
                          <div className="flex items-center gap-1">
                            <button
                              onClick={() => handleSearchPageChange(searchPage - 1)}
                              disabled={searchPage === 1 || searchMutation.isPending}
                              className="btn-secondary p-1 disabled:opacity-30"
                            >
                              <Icons.ChevronLeft className="w-3 h-3" />
                            </button>
                            <span className="text-mono text-[10px] px-1">
                              {searchMutation.data.page}/{searchMutation.data.total_pages}
                            </span>
                            <button
                              onClick={() => handleSearchPageChange(searchPage + 1)}
                              disabled={searchPage >= searchMutation.data.total_pages || searchMutation.isPending}
                              className="btn-secondary p-1 disabled:opacity-30"
                            >
                              <Icons.ChevronRight className="w-3 h-3" />
                            </button>
                          </div>
                        )}
                        <span className="text-mono">{searchMutation.data.search_time_ms}ms</span>
                      </div>
                    </div>

                    <div className="divide-y divide-[var(--color-border)]">
                      {searchMutation.data.results.map((contract, idx) => (
                        <div
                          key={contract.contract_id}
                          className={`p-3 transition-colors animate-slide-up hover:bg-[var(--color-bg-secondary)] ${
                            selectedContract && 'contract_id' in selectedContract && selectedContract.contract_id === contract.contract_id
                              ? 'bg-[var(--color-accent-light)] border-l-2 border-l-[var(--color-accent)]'
                              : ''
                          }`}
                          style={{ animationDelay: `${idx * 20}ms` }}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div
                              className="flex-1 min-w-0 cursor-pointer"
                              onClick={() => setSelectedContract(contract)}
                            >
                              <h3 className="text-xs font-medium text-[var(--color-text-primary)] truncate mb-1">
                                {contract.contract_id}
                              </h3>
                              <div className="flex items-center gap-1.5">
                                <span className="tag text-[10px] py-0.5">{contract.year}</span>
                                <span className="tag text-[10px] py-0.5">{contract.quarter}</span>
                                <span className="text-[10px] text-[var(--color-accent)] font-medium">
                                  {contract.chunks.length} match{contract.chunks.length !== 1 ? 'es' : ''}
                                </span>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <button
                                onClick={(e) => { e.stopPropagation(); toggleBookmark(contract) }}
                                className={`p-1 transition-colors ${isBookmarked(contract.contract_id) ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]'}`}
                                title={isBookmarked(contract.contract_id) ? 'Remove from saved' : 'Save for later'}
                              >
                                <Icons.Bookmark className="w-3.5 h-3.5" filled={isBookmarked(contract.contract_id)} />
                              </button>
                              <div className="text-right">
                                <div className="text-sm font-semibold tabular-nums">
                                  {(contract.top_score * 100).toFixed(0)}%
                                </div>
                                <div className="score-bar w-10 mt-0.5">
                                  <div className="score-bar-fill" style={{ width: `${contract.top_score * 100}%` }} />
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {searchMutation.data.results.length === 0 && (
                      <div className="p-8 text-center text-xs text-[var(--color-text-muted)]">No results found.</div>
                    )}
                  </div>
                )}

                {searchMutation.isError && (
                  <div className="p-3">
                    <div className="p-3 border border-[var(--color-accent)] bg-[var(--color-accent-light)] text-[var(--color-accent)] text-xs">
                      Error: {(searchMutation.error as Error).message}
                    </div>
                  </div>
                )}

                {!searchMutation.data && !searchMutation.isPending && !searchMutation.isError && (
                  <div className="p-8 text-center">
                    <Icons.Search className="w-6 h-6 mx-auto mb-2 text-[var(--color-text-muted)]" />
                    <p className="text-xs text-[var(--color-text-muted)]">Search contracts using natural language</p>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Browse Tab */}
          {activeTab === 'browse' && (
            <>
              {/* Header with year filter and pagination */}
              <div className="flex-shrink-0 p-3 border-b border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
                <div className="flex items-center justify-between gap-2 text-xs">
                  {/* Year filter */}
                  <div className="flex items-center gap-2">
                    <span className="text-label">Year</span>
                    <select
                      value={browseYear}
                      onChange={(e) => { setBrowseYear(e.target.value ? parseInt(e.target.value) : ''); setBrowsePage(1) }}
                      className="input-swiss w-[80px] text-xs py-1 px-2 appearance-none cursor-pointer"
                    >
                      <option value="">All</option>
                      {Array.from({ length: 15 }, (_, i) => 2010 + i).map(year => (
                        <option key={year} value={year}>{year}</option>
                      ))}
                    </select>
                    {browseYear && (
                      <button onClick={() => { setBrowseYear(''); setBrowsePage(1) }} className="text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]">
                        ×
                      </button>
                    )}
                  </div>

                  {/* Pagination controls */}
                  {browseQuery.data && browseQuery.data.total > browseQuery.data.per_page && (
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setBrowsePage(p => Math.max(1, p - 1))}
                        disabled={browsePage === 1}
                        className="btn-secondary p-1 disabled:opacity-30"
                      >
                        <Icons.ChevronLeft className="w-3 h-3" />
                      </button>
                      <span className="text-mono text-[11px] px-2 min-w-[60px] text-center">
                        {browsePage} / {Math.ceil(browseQuery.data.total / browseQuery.data.per_page)}
                      </span>
                      <button
                        onClick={() => setBrowsePage(p => p + 1)}
                        disabled={browsePage >= Math.ceil(browseQuery.data.total / browseQuery.data.per_page)}
                        className="btn-secondary p-1 disabled:opacity-30"
                      >
                        <Icons.ChevronRight className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Total count bar */}
              {browseQuery.data && (
                <div className="flex-shrink-0 px-3 py-1.5 text-xs text-[var(--color-text-muted)] border-b border-[var(--color-border)] bg-[var(--color-bg-tertiary)]">
                  <strong className="text-[var(--color-text-primary)]">{browseQuery.data.total}</strong> contracts
                </div>
              )}

              {/* Scrollable contracts list */}
              <div className="flex-1 overflow-y-auto min-h-0">
                {browseQuery.isLoading && (
                  <div className="p-3 space-y-2">
                    {[...Array(8)].map((_, i) => (
                      <div key={i} className="card-swiss p-3">
                        <div className="skeleton h-3 w-3/4 mb-2" />
                        <div className="skeleton h-2 w-1/3" />
                      </div>
                    ))}
                  </div>
                )}

                {browseQuery.data && (
                  <div className="animate-fade-in divide-y divide-[var(--color-border)]">
                    {browseQuery.data.contracts.map((contract, idx) => (
                      <div
                        key={contract.contract_id}
                        className={`p-3 transition-colors hover:bg-[var(--color-bg-secondary)] ${
                          selectedContract && 'contract_id' in selectedContract && selectedContract.contract_id === contract.contract_id
                            ? 'bg-[var(--color-accent-light)] border-l-2 border-l-[var(--color-accent)]'
                            : ''
                        }`}
                        style={{ animationDelay: `${idx * 15}ms` }}
                      >
                        <div className="flex items-center justify-between">
                          <div
                            className="flex-1 min-w-0 cursor-pointer"
                            onClick={() => setSelectedContract(contract)}
                          >
                            <h3 className="text-xs font-medium text-[var(--color-text-primary)] truncate mb-1">
                              {contract.contract_id}
                            </h3>
                            <div className="flex gap-1.5">
                              <span className="tag text-[10px] py-0.5">{contract.year}</span>
                              <span className="tag text-[10px] py-0.5">{contract.quarter}</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={(e) => { e.stopPropagation(); toggleBookmark(contract) }}
                              className={`p-1 transition-colors ${isBookmarked(contract.contract_id) ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]'}`}
                              title={isBookmarked(contract.contract_id) ? 'Remove from saved' : 'Save for later'}
                            >
                              <Icons.Bookmark className="w-3.5 h-3.5" filled={isBookmarked(contract.contract_id)} />
                            </button>
                            <span className="text-mono text-[10px] text-[var(--color-text-muted)]">{contract.chunk_count}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}

          {/* Bookmarks Tab */}
          {activeTab === 'bookmarks' && (
            <>
              {/* Bookmarks Header with Actions */}
              <div className="flex-shrink-0 p-3 border-b border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={selectAllBookmarks}
                      className="btn-secondary text-[10px] px-2 py-1"
                      disabled={bookmarks.length === 0}
                    >
                      {selectedBookmarks.size === bookmarks.length && bookmarks.length > 0 ? 'Deselect All' : 'Select All'}
                    </button>
                    <span className="text-xs text-[var(--color-text-muted)]">
                      {selectedBookmarks.size > 0 && `${selectedBookmarks.size} selected`}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={downloadSelectedBookmarks}
                      disabled={selectedBookmarks.size === 0 || isDownloading}
                      className="btn-primary text-[10px] px-2 py-1 flex items-center gap-1 disabled:opacity-30"
                      title="Download selected"
                    >
                      {isDownloading ? (
                        <span className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <Icons.Download className="w-3 h-3" />
                      )}
                      Download
                    </button>
                    <button
                      onClick={removeSelectedBookmarks}
                      disabled={selectedBookmarks.size === 0}
                      className="btn-secondary text-[10px] px-2 py-1 disabled:opacity-30 text-[var(--color-accent)]"
                      title="Remove selected"
                    >
                      <Icons.Trash className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto min-h-0">
                {bookmarks.length === 0 ? (
                  <div className="p-8 text-center">
                    <Icons.Bookmark className="w-6 h-6 mx-auto mb-2 text-[var(--color-text-muted)]" />
                    <p className="text-xs text-[var(--color-text-muted)]">No saved contracts yet</p>
                    <p className="text-[10px] text-[var(--color-text-muted)] mt-1">Click the bookmark icon on any contract to save it</p>
                  </div>
                ) : (
                  <div className="divide-y divide-[var(--color-border)]">
                    {bookmarks.map((bookmark, idx) => (
                      <div
                        key={bookmark.contract_id}
                        className={`p-3 transition-colors hover:bg-[var(--color-bg-secondary)] ${
                          selectedBookmarks.has(bookmark.contract_id) ? 'bg-[var(--color-accent-light)]' : ''
                        } ${
                          selectedContract && 'contract_id' in selectedContract && selectedContract.contract_id === bookmark.contract_id
                            ? 'border-l-2 border-l-[var(--color-accent)] bg-[var(--color-accent-light)]'
                            : ''
                        }`}
                        style={{ animationDelay: `${idx * 15}ms` }}
                      >
                        <div className="flex items-center gap-3">
                          {/* Checkbox */}
                          <button
                            onClick={() => toggleSelectBookmark(bookmark.contract_id)}
                            className={`w-4 h-4 border flex items-center justify-center transition-colors ${
                              selectedBookmarks.has(bookmark.contract_id)
                                ? 'bg-[var(--color-accent)] border-[var(--color-accent)] text-white'
                                : 'border-[var(--color-border)] hover:border-[var(--color-text-primary)]'
                            }`}
                          >
                            {selectedBookmarks.has(bookmark.contract_id) && <Icons.Check className="w-3 h-3" />}
                          </button>

                          {/* Contract info - clickable to preview */}
                          <div
                            className="flex-1 min-w-0 cursor-pointer"
                            onClick={() => setSelectedContract({
                              contract_id: bookmark.contract_id,
                              year: bookmark.year,
                              quarter: bookmark.quarter,
                              chunk_count: 0,
                              file_type: bookmark.file_type,
                            } as ContractSummary)}
                          >
                            <h3 className="text-xs font-medium text-[var(--color-text-primary)] truncate mb-1">
                              {bookmark.contract_id}
                            </h3>
                            <div className="flex items-center gap-1.5">
                              <span className="tag text-[10px] py-0.5">{bookmark.year}</span>
                              <span className="tag text-[10px] py-0.5">{bookmark.quarter}</span>
                              {bookmark.file_type && (
                                <span className="text-[10px] text-[var(--color-text-muted)] uppercase">{bookmark.file_type}</span>
                              )}
                            </div>
                          </div>

                          {/* Actions */}
                          <div className="flex items-center gap-1">
                            <a
                              href={`${API_URL}/contract/${encodeURIComponent(bookmark.contract_id)}/file?download=true`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="p-1 text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
                              title="Download"
                            >
                              <Icons.Download className="w-3.5 h-3.5" />
                            </a>
                            <button
                              onClick={() => toggleBookmark(bookmark)}
                              className="p-1 text-[var(--color-accent)] hover:opacity-70 transition-opacity"
                              title="Remove from saved"
                            >
                              <Icons.Bookmark className="w-3.5 h-3.5" filled />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* ========== RIGHT PANEL - Document Viewer ========== */}
        {selectedContract && (
          <div className="flex-1 flex flex-col min-h-0 bg-[var(--color-bg-secondary)] animate-slide-in-right">
            {/* Compact Document Header */}
            <div className="flex-shrink-0 px-4 py-2 border-b border-[var(--color-border)] bg-[var(--color-bg-primary)]">
              <div className="flex items-center justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <h2 className="text-sm font-medium truncate">{selectedContract.contract_id}</h2>
                  <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
                    <span>{selectedContract.year} {selectedContract.quarter}</span>
                    <span>•</span>
                    <span>{contractQuery.data?.total_chunks || '—'} chunks</span>
                    {'chunks' in selectedContract && Array.isArray(selectedContract.chunks) && (
                      <>
                        <span>•</span>
                        <span className="text-[var(--color-accent)]">{selectedContract.chunks.length} matches</span>
                      </>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-1">
                  {/* View Mode Toggle */}
                  <div className="flex mr-2">
                    <button
                      onClick={() => setViewMode('original')}
                      className={`px-2 py-1 text-[10px] font-medium border transition-colors ${
                        viewMode === 'original'
                          ? 'bg-[var(--color-text-primary)] text-[var(--color-bg-primary)] border-[var(--color-text-primary)]'
                          : 'text-[var(--color-text-muted)] border-[var(--color-border)] hover:text-[var(--color-text-primary)]'
                      }`}
                    >
                      Doc
                    </button>
                    <button
                      onClick={() => setViewMode('chunks')}
                      className={`px-2 py-1 text-[10px] font-medium border-y border-r transition-colors ${
                        viewMode === 'chunks'
                          ? 'bg-[var(--color-text-primary)] text-[var(--color-bg-primary)] border-[var(--color-text-primary)]'
                          : 'text-[var(--color-text-muted)] border-[var(--color-border)] hover:text-[var(--color-text-primary)]'
                      }`}
                    >
                      Text
                    </button>
                  </div>

                  <button
                    onClick={() => toggleBookmark(selectedContract)}
                    className={`p-1.5 transition-colors ${isBookmarked(selectedContract.contract_id) ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]'}`}
                    title={isBookmarked(selectedContract.contract_id) ? 'Remove from saved' : 'Save for later'}
                  >
                    <Icons.Bookmark className="w-3.5 h-3.5" filled={isBookmarked(selectedContract.contract_id)} />
                  </button>

                  {contractQuery.data?.file_path && (
                    <a
                      href={`${API_URL}/contract/${encodeURIComponent(selectedContract.contract_id)}/file?download=true`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-secondary p-1.5"
                      title="Download"
                    >
                      <Icons.Download className="w-3.5 h-3.5" />
                    </a>
                  )}
                  <button onClick={() => setSelectedContract(null)} className="btn-ghost p-1.5">
                    <Icons.Close className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Document Content */}
            <div className="flex-1 min-h-0 overflow-hidden">
              {contractQuery.isLoading && (
                <div className="flex items-center justify-center h-full">
                  <div className="w-5 h-5 border-2 border-[var(--color-text-primary)] border-t-transparent rounded-full animate-spin" />
                </div>
              )}

              {contractQuery.isError && (
                <div className="p-4">
                  <div className="p-3 border border-[var(--color-accent)] bg-[var(--color-accent-light)] text-[var(--color-accent)] text-xs">
                    Error: {(contractQuery.error as Error).message}
                  </div>
                </div>
              )}

              {contractQuery.data && viewMode === 'original' && (
                <div className="h-full">
                  {contractQuery.data.file_path ? (
                    <DocumentViewer
                      fileUrl={`${API_URL}/contract/${encodeURIComponent(selectedContract.contract_id)}/file`}
                      fileType={contractQuery.data.file_type as 'pdf' | 'htm' | 'html' || 'htm'}
                      matchedChunks={
                        'chunks' in selectedContract && Array.isArray(selectedContract.chunks)
                          ? (selectedContract as ContractResult).chunks
                          : undefined
                      }
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-center p-6">
                      <Icons.Document className="w-8 h-8 mb-3 text-[var(--color-text-muted)]" />
                      <p className="text-xs text-[var(--color-text-muted)] mb-3">Original file not available</p>
                      <button onClick={() => setViewMode('chunks')} className="btn-primary text-xs">
                        View Text
                      </button>
                    </div>
                  )}
                </div>
              )}

              {contractQuery.data && viewMode === 'chunks' && (
                <div className="h-full overflow-y-auto p-4 space-y-2">
                  {'chunks' in selectedContract && Array.isArray(selectedContract.chunks) && selectedContract.chunks.length > 0 && (
                    <div className="flex items-center gap-2 mb-3 flex-wrap">
                      <span className="text-label text-[10px]">Jump:</span>
                      {(selectedContract as ContractResult).chunks.map((chunk, idx) => (
                        <button
                          key={chunk.chunk_id}
                          onClick={() => scrollToChunk(chunk.chunk_id)}
                          className="tag tag-accent text-[10px] py-0.5 cursor-pointer hover:opacity-80"
                        >
                          #{idx + 1}
                        </button>
                      ))}
                    </div>
                  )}

                  {contractQuery.data.chunks.map((chunk) => {
                    const isMatch = matchedChunkIds.has(chunk.chunk_id)
                    const matchScore = getMatchScore(chunk.chunk_id)

                    return (
                      <div
                        key={chunk.chunk_id}
                        ref={(el) => { if (el) chunkRefs.current.set(chunk.chunk_id, el) }}
                        className={`p-3 bg-[var(--color-bg-primary)] border border-[var(--color-border)] text-xs ${
                          isMatch ? 'match-highlight' : ''
                        }`}
                      >
                        {isMatch && (
                          <div className="flex items-center gap-2 mb-2">
                            <span className="tag tag-accent text-[10px] py-0.5">
                              MATCH {(matchScore! * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                        <p className="text-[var(--color-text-primary)] whitespace-pre-wrap leading-relaxed">
                          {chunk.text}
                        </p>
                        <div className="mt-2 text-[10px] text-[var(--color-text-muted)] text-mono">
                          {chunk.chunk_id + 1}/{contractQuery.data.total_chunks}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty state when no document selected */}
        {!selectedContract && (
          <div className="flex-1 flex flex-col items-center justify-center bg-[var(--color-bg-secondary)] text-center p-8">
            <div className="w-16 h-16 mb-4 border border-[var(--color-border)] flex items-center justify-center">
              <Icons.Document className="w-6 h-6 text-[var(--color-text-muted)]" />
            </div>
            <h3 className="text-sm font-medium text-[var(--color-text-primary)] mb-1">No Document Selected</h3>
            <p className="text-xs text-[var(--color-text-muted)] max-w-xs">
              Search for contracts or browse the archive, then select a document to view it here.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

// =============================================================================
// APP WRAPPER
// =============================================================================

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ContractSearchApp />
    </QueryClientProvider>
  )
}

export default App
