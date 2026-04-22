from __future__ import annotations

import time

from backend.tools.rag.config import get_config
from backend.tools.rag.knowledge_base import KnowledgeBaseBuilder, save_chunks
from backend.tools.rag.retrieval import BM25Index, BgeM3Embedder, QdrantVectorStore


def main() -> None:
    started_at = time.perf_counter()
    config = get_config()
    builder = KnowledgeBaseBuilder(config)
    chunks, errors = builder.build()

    if not chunks:
        raise RuntimeError("No chunks were built from data/local_knowledge.")

    embedder = BgeM3Embedder(config)
    try:
        embeddings = embedder.encode(chunk.text for chunk in chunks)
        vector_store = QdrantVectorStore(config)
        vector_store.replace_collection(chunks, embeddings)
    finally:
        embedder.close()

    bm25_index = BM25Index(chunks)
    bm25_index.save(config.bm25_index_path)
    save_chunks(chunks, config.chunk_manifest_path)

    print("Index build completed.")
    print(f"Documents directory: {config.local_knowledge_dir}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total vectors: {len(chunks)}")
    print(f"Qdrant path: {config.qdrant_path}")
    print(f"BM25 index: {config.bm25_index_path}")
    print(f"Chunk manifest: {config.chunk_manifest_path}")
    print(f"Elapsed ms: {round((time.perf_counter() - started_at) * 1000, 2)}")
    if errors:
        print("Documents skipped due to parse errors:")
        for item in errors:
            print(f"- {item}")


if __name__ == "__main__":
    main()
