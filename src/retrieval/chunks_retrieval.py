import asyncio
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import settings
from ingestion.embedder import Embedding
from src.ingestion.vectorizer import Vectorizer


class SearchEngine:
    def __init__(self):
        self._vectorizer = Vectorizer()
        self._embedder = Embedding()

    async def search(
        self,
        query_text: str,
        limit: int = 5,
        score_threshold: float | None = None,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        """
        Search by text. Returns entire chunks with similarity score added.
        """
        # Embed query
        query_chunk = [{"content": query_text, "metadata": {}}]
        embedded = await self._embedder.embed_chunks(query_chunk)
        query_vector = embedded[0]["metadata"]["embedding"]

        # Search Qdrant
        results = await self._vectorizer._client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        # Return entire chunks (content + metadata + score)
        return [
            {
                "content": r.payload["content"],
                "metadata": r.payload["metadata"],
                "score": r.score,
            }
            for r in results
        ]

    async def search_with_filters(
        self,
        query_text: str,
        filters: dict,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Search with simple dict filters."""
        query_filter = self._build_filter(filters) if filters else None
        return await self.search(
            query_text=query_text,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

    def _build_filter(self, filters: dict) -> Filter:
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                )
            )
        return Filter(must=conditions)


# ─── Convenience one-shot function ───

async def search(
    query_text: str,
    limit: int = 5,
    score_threshold: float | None = None,
    query_filter: Filter | None = None,
) -> list[dict]:
    engine = SearchEngine()
    return await engine.search(
        query_text=query_text,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )