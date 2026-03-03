"""
Load pre-computed embeddings from JSONL file into Qdrant.

Usage:
    python load_embeddings.py --input contract_embeddings.jsonl.gz

This is the second step after running the Colab notebook.
Supports both plain .jsonl and compressed .jsonl.gz files.
"""
import argparse
import gzip
import json
from pathlib import Path
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


def load_embeddings(
    input_file: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = "contracts",
    batch_size: int = 100,
):
    """Load embeddings from JSONL file into Qdrant."""

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return

    # Detect if gzip compressed
    is_gzip = input_file.endswith('.gz')
    opener = gzip.open if is_gzip else open
    mode = 'rt' if is_gzip else 'r'

    # Count total records
    print("Counting records...")
    with opener(input_path, mode) as f:
        total = sum(1 for _ in f)
    print(f"Found {total} records to load")

    # Get embedding dimension from first record
    with opener(input_path, mode) as f:
        first = json.loads(f.readline())
        dimension = len(first['embedding'])
    print(f"Embedding dimension: {dimension}")

    # Connect to Qdrant
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Create or recreate collection
    collections = [c.name for c in client.get_collections().collections]
    if collection in collections:
        print(f"Deleting existing collection: {collection}")
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    print(f"Created collection: {collection}")

    # Load in batches
    points = []
    point_id = 0

    with opener(input_path, mode) as f:
        for line in tqdm(f, total=total, desc="Loading"):
            record = json.loads(line)

            # Extract embedding and metadata
            embedding = record.pop('embedding')

            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=record,
            ))
            point_id += 1

            # Insert batch
            if len(points) >= batch_size:
                client.upsert(collection_name=collection, points=points)
                points = []

    # Insert remaining
    if points:
        client.upsert(collection_name=collection, points=points)

    # Verify
    info = client.get_collection(collection)
    print(f"\nDone! Collection '{collection}' has {info.points_count} points")


def main():
    parser = argparse.ArgumentParser(description="Load embeddings into Qdrant")
    parser.add_argument("--input", required=True, help="Path to JSONL file with embeddings")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="contracts", help="Collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Insert batch size")

    args = parser.parse_args()

    load_embeddings(
        input_file=args.input,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection=args.collection,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
