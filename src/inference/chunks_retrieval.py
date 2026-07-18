import json
from pathlib import Path
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore


class ChunkRetrievalError(Exception):
    """Raised when a chunk retrieval operation fails, wrapping the underlying cause."""


class ChunkRetrieval:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ):
        self._vector_store = vector_store or VectorStore()
        self._embedder = embedder or Embedder()

    async def search(
        self,
        query_text: str,
        limit: int = 5,
        score_threshold: float | None = None,
        filters: dict | Filter | None = None,
    ) -> list[dict]:
        """
        Search by text. Returns entire chunks with similarity score added.
        `filters` accepts either a plain dict (e.g. {'ticker': 'TSLA'}) or a
        pre-built qdrant_client Filter for more complex conditions.
        """
        query_filter = self._build_filter(filters) if isinstance(filters, dict) else filters

        try:
            query_vector = await self._embedder.embed_query(query_text)
        except Exception as e:
            raise ChunkRetrievalError(f"Failed to embed query text: {e}") from e

        try:
            results = await self._vector_store.search(
                query_vector=query_vector,
                top_k=limit,
                filters=query_filter,
                score_threshold=score_threshold,
            )
        except Exception as e:
            raise ChunkRetrievalError(f"Vector store search failed: {e}") from e

        query_results= [
            {
                "content": r["payload"]["content"],
                "metadata": r["payload"].get("metadata", {}),
                "score": r["score"],
            }
            for r in results
        ]

        return query_results

    @staticmethod
    def _build_filter(filters: dict) -> Filter:
        conditions = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        return Filter(must=conditions)