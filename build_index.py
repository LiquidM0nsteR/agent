from __future__ import annotations

from .backend.tools.rag.config import get_config
from .backend.tools.rag.knowledge_base import KnowledgeBaseBuilder, save_chunks
from .backend.tools.rag.retrieval import BM25Index, BgeM3Embedder, QdrantVectorStore


def main() -> None:
    config = get_config()
    builder = KnowledgeBaseBuilder(config)
    chunks, errors = builder.build()

    if not chunks:
        raise RuntimeError("No chunks were built from data/local_knowledge.")

    embedder = BgeM3Embedder(config)
    embeddings = embedder.encode(chunk.text for chunk in chunks)

    vector_store = QdrantVectorStore(config)
    vector_store.recreate_collection(vector_size=len(embeddings[0]))
    vector_store.upsert_chunks(chunks, embeddings)

    bm25_index = BM25Index(chunks)
    bm25_index.save(config.bm25_index_path)
    save_chunks(chunks, config.chunk_manifest_path)

    print("Index build completed.")
    print(f"Documents directory: {config.local_knowledge_dir}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Qdrant path: {config.qdrant_path}")
    print(f"BM25 index: {config.bm25_index_path}")
    print(f"Chunk manifest: {config.chunk_manifest_path}")
    if errors:
        print("Documents skipped due to parse errors:")
        for item in errors:
            print(f"- {item}")


if __name__ == "__main__":
    main()
