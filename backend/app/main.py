"""
Simplified FastAPI backend for contract semantic search.

Uses Qdrant (single container) instead of Milvus (3 containers).
No Redis caching (simpler, can add later if needed).

Endpoints:
    GET  /health              - Health check
    GET  /stats               - Collection statistics
    POST /search              - Semantic search with filters
    GET  /contracts           - Browse all contracts (paginated)
    GET  /contract/{id}       - Get contract metadata and chunks
    GET  /contract/{id}/file  - Serve original PDF/HTML file
"""
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from functools import lru_cache

# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """App settings from environment variables."""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "contracts"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    data_dir: str = "test_data"  # Directory containing original files

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODELS (Pydantic schemas)
# =============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    year_start: Optional[int] = Field(None, ge=2000, le=2030)
    year_end: Optional[int] = Field(None, ge=2000, le=2030)
    quarters: Optional[List[str]] = Field(None, description="Filter by Q1, Q2, Q3, Q4")
    min_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score (0-1)")
    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(20, ge=1, le=100, description="Results per page")
    max_results: int = Field(1000, ge=100, le=5000, description="Maximum chunks to search through")


class ChunkResult(BaseModel):
    chunk_id: int
    text: str
    score: float
    section: Optional[str] = None


class ContractResult(BaseModel):
    contract_id: str
    year: int
    quarter: str
    top_score: float
    chunks: List[ChunkResult]


class SearchResponse(BaseModel):
    query: str
    total_contracts: int
    page: int
    per_page: int
    total_pages: int
    min_score: float
    results: List[ContractResult]
    search_time_ms: float


class ContractSummary(BaseModel):
    contract_id: str
    year: int
    quarter: str
    chunk_count: int
    file_path: Optional[str] = None
    file_type: Optional[str] = None  # pdf, htm, html


class ContractsListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    contracts: List[ContractSummary]


class ContractChunk(BaseModel):
    chunk_id: int
    text: str
    section: Optional[str] = None


class ContractDocument(BaseModel):
    contract_id: str
    year: int
    quarter: str
    total_chunks: int
    file_path: Optional[str] = None
    file_type: Optional[str] = None  # pdf, htm, html
    chunks: List[ContractChunk]


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    total_vectors: int


class StatsResponse(BaseModel):
    total_vectors: int
    total_contracts: int
    years: List[int]


# =============================================================================
# SERVICES
# =============================================================================

# Global instances (initialized on startup)
qdrant_client = None
embedding_model = None


def get_qdrant():
    """Get Qdrant client."""
    from qdrant_client import QdrantClient
    global qdrant_client
    if qdrant_client is None:
        settings = get_settings()
        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return qdrant_client


def get_embedding_model():
    """Get embedding model (lazy load)."""
    from sentence_transformers import SentenceTransformer
    global embedding_model
    if embedding_model is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded")
    return embedding_model


def embed_query(query: str) -> List[float]:
    """Embed a search query."""
    model = get_embedding_model()
    # BGE models need instruction prefix for queries
    if "bge" in get_settings().embedding_model.lower():
        query = f"Represent this sentence for searching relevant passages: {query}"
    vector = model.encode(query, normalize_embeddings=True)
    return vector.tolist()


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    logger.info("Starting Contract Search API...")
    settings = get_settings()
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")

    # Pre-load embedding model (optional, can lazy load)
    try:
        get_embedding_model()
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Contract Search API",
    description="Semantic search for contract documents",
    version="2.0.0",
    lifespan=lifespan
)

# CORS - allow all for development (restrict in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API info."""
    return {
        "name": "Contract Search API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check."""
    try:
        client = get_qdrant()
        settings = get_settings()
        info = client.get_collection(settings.collection_name)
        return HealthResponse(
            status="healthy",
            qdrant_connected=True,
            total_vectors=info.points_count
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            qdrant_connected=False,
            total_vectors=0
        )


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
async def stats():
    """Collection statistics."""
    try:
        client = get_qdrant()
        settings = get_settings()
        info = client.get_collection(settings.collection_name)

        # Get unique years from actual data
        # For now, estimate - could query distinct values
        years = list(range(2015, 2025))  # Reasonable default

        return StatsResponse(
            total_vectors=info.points_count,
            total_contracts=info.points_count // 5,  # Rough estimate
            years=years
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=503, detail="Qdrant not available")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Semantic search for contracts.

    Returns contracts grouped by contract_id, sorted by relevance.
    """
    start_time = time.time()

    try:
        client = get_qdrant()
        settings = get_settings()

        # Embed query
        query_vector = embed_query(request.query)

        # Build filter
        from qdrant_client.models import Filter, FieldCondition, Range, MatchAny

        must_conditions = []

        if request.year_start is not None:
            must_conditions.append(
                FieldCondition(key="year", range=Range(gte=request.year_start))
            )

        if request.year_end is not None:
            must_conditions.append(
                FieldCondition(key="year", range=Range(lte=request.year_end))
            )

        if request.quarters:
            must_conditions.append(
                FieldCondition(key="quarter", match=MatchAny(any=request.quarters))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Search Qdrant - fetch many results and filter by score
        # Using score_threshold to get only results above min_score
        search_result = client.query_points(
            collection_name=settings.collection_name,
            query=query_vector,
            query_filter=query_filter,
            score_threshold=request.min_score,  # Only return results >= min_score
            limit=request.max_results,  # Fetch up to max_results chunks
            with_payload=True,
        )
        results = search_result.points

        # Group by contract_id (Qdrant already filtered by score_threshold)
        from collections import defaultdict
        grouped = defaultdict(list)

        for hit in results:
            contract_id = hit.payload.get("contract_id", "unknown")
            grouped[contract_id].append({
                "chunk_id": hit.payload.get("chunk_index", 0),
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "section": hit.payload.get("section"),
                "year": hit.payload.get("year", 0),
                "quarter": hit.payload.get("quarter", ""),
            })

        # Build response
        contract_results = []
        for contract_id, chunks in grouped.items():
            # Sort chunks by score
            chunks.sort(key=lambda x: x["score"], reverse=True)
            top_chunk = chunks[0]

            contract_results.append(ContractResult(
                contract_id=contract_id,
                year=top_chunk["year"],
                quarter=top_chunk["quarter"],
                top_score=top_chunk["score"],
                chunks=[
                    ChunkResult(
                        chunk_id=c["chunk_id"],
                        text=c["text"],
                        score=c["score"],
                        section=c["section"]
                    )
                    for c in chunks
                ]
            ))

        # Sort by top score
        contract_results.sort(key=lambda x: x.top_score, reverse=True)

        # Calculate pagination
        total_contracts = len(contract_results)
        total_pages = (total_contracts + request.per_page - 1) // request.per_page if total_contracts > 0 else 1
        start_idx = (request.page - 1) * request.per_page
        end_idx = start_idx + request.per_page
        paginated_results = contract_results[start_idx:end_idx]

        elapsed_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query=request.query,
            total_contracts=total_contracts,
            page=request.page,
            per_page=request.per_page,
            total_pages=total_pages,
            min_score=request.min_score,
            results=paginated_results,
            search_time_ms=round(elapsed_ms, 2)
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contracts", response_model=ContractsListResponse, tags=["Browse"])
async def list_contracts(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    year: Optional[int] = Query(None, description="Filter by year"),
):
    """
    Browse all contracts with pagination.
    """
    try:
        client = get_qdrant()
        settings = get_settings()

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from collections import defaultdict

        # Build filter
        query_filter = None
        if year is not None:
            query_filter = Filter(
                must=[FieldCondition(key="year", match=MatchValue(value=year))]
            )

        # Get unique contracts by scrolling in smaller batches
        contracts_map = defaultdict(lambda: {"year": 0, "quarter": "", "count": 0, "file_path": None, "file_type": None})
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=["contract_id", "year", "quarter", "file_path", "file_type"],
                with_vectors=False,
            )

            for point in results:
                cid = point.payload.get("contract_id", "unknown")
                contracts_map[cid]["year"] = point.payload.get("year", 0)
                contracts_map[cid]["quarter"] = point.payload.get("quarter", "")
                contracts_map[cid]["file_path"] = point.payload.get("file_path")
                contracts_map[cid]["file_type"] = point.payload.get("file_type")
                contracts_map[cid]["count"] += 1

            if offset is None or len(results) == 0:
                break

        # Convert to list and sort
        contracts_list = [
            ContractSummary(
                contract_id=cid,
                year=data["year"],
                quarter=data["quarter"],
                chunk_count=data["count"],
                file_path=data["file_path"],
                file_type=data["file_type"]
            )
            for cid, data in contracts_map.items()
        ]
        contracts_list.sort(key=lambda x: (x.year, x.quarter, x.contract_id), reverse=True)

        # Paginate
        total = len(contracts_list)
        start = (page - 1) * per_page
        end = start + per_page
        page_contracts = contracts_list[start:end]

        return ContractsListResponse(
            total=total,
            page=page,
            per_page=per_page,
            contracts=page_contracts
        )

    except Exception as e:
        logger.error(f"List contracts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contract/{contract_id}", response_model=ContractDocument, tags=["Contract"])
async def get_contract(contract_id: str):
    """
    Get full contract document with all chunks.
    """
    try:
        client = get_qdrant()
        settings = get_settings()

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Get all chunks for this contract
        results, _ = client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="contract_id", match=MatchValue(value=contract_id))]
            ),
            limit=1000,  # Max chunks per contract
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")

        # Sort by chunk_index
        chunks_data = sorted(results, key=lambda x: x.payload.get("chunk_index", 0))

        # Get metadata from first chunk
        first = chunks_data[0].payload

        return ContractDocument(
            contract_id=contract_id,
            year=first.get("year", 0),
            quarter=first.get("quarter", ""),
            total_chunks=len(chunks_data),
            file_path=first.get("file_path"),
            file_type=first.get("file_type"),
            chunks=[
                ContractChunk(
                    chunk_id=c.payload.get("chunk_index", 0),
                    text=c.payload.get("text", ""),
                    section=c.payload.get("section")
                )
                for c in chunks_data
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get contract error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contract/{contract_id}/file", tags=["Contract"])
async def get_contract_file(
    contract_id: str,
    download: bool = Query(False, description="Force download instead of inline view")
):
    """
    Serve the original PDF or HTML file for a contract.

    - Default: Opens PDF in browser viewer (inline)
    - With ?download=true: Forces file download
    """
    try:
        client = get_qdrant()
        settings = get_settings()

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from starlette.responses import Response

        # Get one chunk to find file path
        results, _ = client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="contract_id", match=MatchValue(value=contract_id))]
            ),
            limit=1,
            with_payload=["file_path", "file_type"],
            with_vectors=False,
        )

        if not results:
            raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")

        file_path = results[0].payload.get("file_path")
        file_type = results[0].payload.get("file_type", "htm")

        if not file_path:
            raise HTTPException(status_code=404, detail="Original file path not stored for this contract")

        # Construct full path
        full_path = Path(settings.data_dir) / file_path

        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        # Serve based on file type
        if file_type == "pdf":
            # Set Content-Disposition: inline for viewing, attachment for download
            disposition = "attachment" if download else "inline"
            filename = f"{contract_id}.pdf"

            with open(full_path, 'rb') as f:
                content = f.read()

            return Response(
                content=content,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'{disposition}; filename="{filename}"'
                }
            )
        else:
            # HTML/HTM - read and return
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return HTMLResponse(content=content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get contract file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
