"""
Unified ingestion pipeline: PDF/HTML -> Chunks -> Embeddings -> Vector DB

One script to rule them all. Run with:
    python ingest.py --data-dir ./data --years 2019 2020 2021

What this does:
    1. Scans documents in data/{year}/{quarter}/*.pdf and *.htm
    2. Extracts text from each document (PDF or HTML)
    3. Chunks text semantically (preserves contract structure)
    4. Generates embeddings with BGE model
    5. Stores in Qdrant vector database
    6. Auto-builds index when done

Resume: If interrupted, re-run the same command. Already-processed
        contracts are skipped automatically.
"""
import argparse
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Generator, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Edit these if needed
# =============================================================================

DEFAULT_CONFIG = {
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "chunk_size": 512,       # tokens per chunk (smaller = more precise search)
    "chunk_overlap": 50,     # overlap between chunks
    "batch_size": 32,        # embeddings per batch (lower if OOM)
    "qdrant_collection": "contracts",
}


# =============================================================================
# CHUNKING - Semantic splitting for legal documents
# =============================================================================

# Patterns to detect section headers in contracts
SECTION_PATTERNS = [
    r'^ARTICLE\s+[IVXLCDM\d]+',
    r'^SECTION\s+\d+',
    r'^\d+\.\s+[A-Z][A-Z\s]{2,}',
    r'^[A-Z][A-Z\s]{15,}$',
    r'^(?:WHEREAS|RECITALS|DEFINITIONS|TERM|TERMINATION)',
    r'^(?:PAYMENT|CONFIDENTIALITY|INDEMNIFICATION|GOVERNING LAW)',
]

@dataclass
class Chunk:
    text: str
    section: str
    index: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word."""
    return int(len(text.split()) * 1.3)


def is_section_header(line: str) -> bool:
    """Check if line looks like a section header."""
    line = line.strip()
    if not line or len(line) < 3:
        return False
    return any(re.match(p, line, re.IGNORECASE) for p in SECTION_PATTERNS)


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[Chunk]:
    """
    Split document into chunks, respecting section boundaries.

    Strategy:
    1. Split by section headers first (semantic boundaries)
    2. If section > max_tokens, split by paragraphs
    3. If still too big, split by sentences
    """
    chunks = []
    current_section = "PREAMBLE"
    current_text = ""

    for line in text.split('\n'):
        if is_section_header(line):
            # Save current chunk if it has content
            if current_text.strip():
                chunks.extend(_split_large_section(current_text, current_section, max_tokens))
            current_section = line.strip()[:100]
            current_text = ""
        else:
            current_text += line + "\n"

    # Don't forget the last section
    if current_text.strip():
        chunks.extend(_split_large_section(current_text, current_section, max_tokens))

    # Add overlap and create final chunks
    final_chunks = []
    for i, (text, section) in enumerate(chunks):
        # Add context from previous chunk
        if i > 0 and overlap > 0:
            prev_words = chunks[i-1][0].split()
            overlap_words = min(int(overlap / 1.3), len(prev_words))
            prefix = " ".join(prev_words[-overlap_words:]) + " "
            text = prefix + text

        final_chunks.append(Chunk(text=text.strip(), section=section, index=i))

    return final_chunks


def _split_large_section(text: str, section: str, max_tokens: int) -> List[tuple]:
    """Split a section that's too large into smaller pieces."""
    if estimate_tokens(text) <= max_tokens:
        return [(f"{section}\n{text}" if section != "PREAMBLE" else text, section)]

    results = []

    # Try splitting by paragraphs
    paragraphs = text.split('\n\n')
    current = ""

    for para in paragraphs:
        test = current + "\n\n" + para if current else para
        if estimate_tokens(test) <= max_tokens:
            current = test
        else:
            if current:
                prefix = f"{section}\n" if not results else ""
                results.append((prefix + current, section))
            current = para

    if current:
        prefix = f"{section}\n" if not results else ""
        results.append((prefix + current, section))

    return results if results else [(text[:max_tokens * 4], section)]  # Fallback


# =============================================================================
# TEXT EXTRACTION (PDF and HTML)
# =============================================================================

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract all text from a PDF file."""
    try:
        doc = fitz.open(str(pdf_path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text if text.strip() else None
    except Exception as e:
        logger.warning(f"Failed to extract PDF {pdf_path.name}: {e}")
        return None


def extract_text_from_html(html_path: Path) -> Optional[str]:
    """Extract text from an HTML file."""
    try:
        from html.parser import HTMLParser
        from io import StringIO

        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = StringIO()
                self.skip_tags = {'script', 'style', 'head', 'meta'}
                self.current_tag = None

            def handle_starttag(self, tag, attrs):
                self.current_tag = tag.lower()

            def handle_endtag(self, tag):
                if tag.lower() in {'p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr'}:
                    self.text.write('\n')
                self.current_tag = None

            def handle_data(self, data):
                if self.current_tag not in self.skip_tags:
                    self.text.write(data)

            def get_text(self):
                return self.text.getvalue()

        # Read HTML file
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()

        # Parse and extract text
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = parser.get_text()

        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)

        return text if text.strip() else None
    except Exception as e:
        logger.warning(f"Failed to extract HTML {html_path.name}: {e}")
        return None


def extract_text(file_path: Path) -> Optional[str]:
    """Extract text from a file (PDF or HTML)."""
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        return extract_text_from_pdf(file_path)
    elif suffix in ['.htm', '.html']:
        return extract_text_from_html(file_path)
    else:
        logger.warning(f"Unsupported file type: {suffix}")
        return None


def find_documents(data_dir: Path, years: List[int] = None) -> List[Dict]:
    """
    Find all documents organized as: data/{year}/{quarter}/*.pdf or *.htm

    Returns list of dicts with: path, year, quarter, contract_id
    """
    docs = []

    for year_dir in sorted(data_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        # Filter by years if specified
        if years and year not in years:
            continue

        for quarter_dir in year_dir.iterdir():
            if not quarter_dir.is_dir():
                continue

            quarter = quarter_dir.name.upper()
            if quarter not in ["Q1", "Q2", "Q3", "Q4"]:
                continue

            # Find both PDFs and HTML files
            for pattern in ["*.pdf", "*.htm", "*.html"]:
                for doc in quarter_dir.glob(pattern):
                    docs.append({
                        "path": doc,
                        "year": year,
                        "quarter": quarter,
                        "contract_id": doc.stem,
                    })

    logger.info(f"Found {len(docs)} documents to process ({sum(1 for d in docs if d['path'].suffix == '.pdf')} PDFs, {sum(1 for d in docs if d['path'].suffix in ['.htm', '.html'])} HTML)")
    return docs


# =============================================================================
# VECTOR DATABASE (Qdrant - simpler than Milvus)
# =============================================================================

def setup_qdrant(host: str, port: int, collection: str, dimension: int):
    """Initialize Qdrant connection and collection."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance

    client = QdrantClient(host=host, port=port)

    # Create collection if it doesn't exist
    collections = [c.name for c in client.get_collections().collections]

    if collection not in collections:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {collection}")
    else:
        logger.info(f"Using existing collection: {collection}")

    return client


def get_existing_contracts(client, collection: str) -> set:
    """Get set of already-indexed contract IDs for resume capability."""
    from qdrant_client.models import ScrollRequest

    existing = set()
    offset = None

    while True:
        results = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=["contract_id"],
            with_vectors=False,
        )

        points, offset = results

        for point in points:
            if point.payload and "contract_id" in point.payload:
                existing.add(point.payload["contract_id"])

        if offset is None:
            break

    return existing


# =============================================================================
# MAIN INGESTION PIPELINE
# =============================================================================

def ingest(
    data_dir: str,
    years: List[int] = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = "contracts",
    model_name: str = "BAAI/bge-large-en-v1.5",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    batch_size: int = 32,
    no_resume: bool = False,
):
    """
    Main ingestion function. Processes PDFs and HTML files, stores in Qdrant.

    Args:
        data_dir: Path to data/{year}/{quarter}/*.pdf or *.htm
        years: Only process these years (e.g., [2019, 2020, 2021])
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        collection: Collection name
        model_name: Sentence transformer model
        chunk_size: Max tokens per chunk
        chunk_overlap: Overlap between chunks
        batch_size: Embeddings per batch
        no_resume: If True, reprocess everything
    """
    from sentence_transformers import SentenceTransformer
    from qdrant_client.models import PointStruct
    import torch
    import os

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # CPU optimization - use all available cores
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    logger.info(f"CPU threads: {num_cores} cores available, using all")

    # Load embedding model
    logger.info(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)
    dimension = model.get_sentence_embedding_dimension()
    logger.info(f"Model ready. Embedding dimension: {dimension}")

    # Setup Qdrant
    client = setup_qdrant(qdrant_host, qdrant_port, collection, dimension)

    # Get already processed contracts (for resume)
    existing = set() if no_resume else get_existing_contracts(client, collection)
    if existing:
        logger.info(f"Found {len(existing)} already-indexed contracts (will skip)")

    # Find documents to process
    docs = find_documents(data_path, years)
    docs = [d for d in docs if d["contract_id"] not in existing]

    if not docs:
        logger.info("No new documents to process!")
        return

    logger.info(f"Processing {len(docs)} documents...")

    # Process in batches
    total_chunks = 0
    point_id = len(existing) * 100  # Start IDs after existing

    batch_texts = []
    batch_metadata = []

    for doc_info in tqdm(docs, desc="Processing documents"):
        # Extract text
        text = extract_text(doc_info["path"])
        if not text:
            continue

        # Chunk
        chunks = chunk_text(text, max_tokens=chunk_size, overlap=chunk_overlap)

        # Get relative path for file serving
        relative_path = str(doc_info["path"].relative_to(data_path))
        file_type = doc_info["path"].suffix.lower().lstrip('.')

        for chunk in chunks:
            batch_texts.append(chunk.text)
            batch_metadata.append({
                "contract_id": doc_info["contract_id"],
                "year": doc_info["year"],
                "quarter": doc_info["quarter"],
                "section": chunk.section,
                "chunk_index": chunk.index,
                "text": chunk.text,  # Store full text for retrieval
                "file_path": relative_path,  # For serving original file
                "file_type": file_type,  # pdf, htm, html
            })

            # Process batch when full
            if len(batch_texts) >= batch_size:
                _insert_batch(client, collection, model, batch_texts, batch_metadata, point_id)
                total_chunks += len(batch_texts)
                point_id += len(batch_texts)
                batch_texts = []
                batch_metadata = []

    # Insert remaining
    if batch_texts:
        _insert_batch(client, collection, model, batch_texts, batch_metadata, point_id)
        total_chunks += len(batch_texts)

    logger.info(f"Done! Indexed {total_chunks} chunks from {len(docs)} contracts")

    # Print collection stats
    info = client.get_collection(collection)
    logger.info(f"Collection '{collection}' now has {info.points_count} points")


def _insert_batch(client, collection, model, texts, metadata, start_id):
    """Embed and insert a batch of chunks."""
    from qdrant_client.models import PointStruct

    # Generate embeddings
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Create points
    points = [
        PointStruct(
            id=start_id + i,
            vector=embedding.tolist(),
            payload=meta,
        )
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata))
    ]

    # Insert
    client.upsert(collection_name=collection, points=points)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDFs and HTML files into vector database for semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all data
    python ingest.py --data-dir ./data

    # Process only 2019-2021
    python ingest.py --data-dir ./data --years 2019 2020 2021

    # Start fresh (reprocess everything)
    python ingest.py --data-dir ./data --no-resume

    # Use smaller batches (if running out of memory)
    python ingest.py --data-dir ./data --batch-size 16
        """
    )

    parser.add_argument("--data-dir", required=True, help="Path to data/{year}/{quarter}/*.pdf or *.htm")
    parser.add_argument("--years", type=int, nargs="+", help="Only process these years")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="contracts", help="Collection name")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="Embedding model")
    parser.add_argument("--chunk-size", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess everything")

    args = parser.parse_args()

    ingest(
        data_dir=args.data_dir,
        years=args.years,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection=args.collection,
        model_name=args.model,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        no_resume=args.no_resume,
    )


if __name__ == "__main__":
    main()
