import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import settings


class Vectorizer:
    def __init__(self):
        self._client = AsyncQdrantClient(settings.QDRANT_URL)
        self._batch_size = getattr(settings, "QDRANT_BATCH_SIZE", 100)

    async def create_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]
        
        if settings.QDRANT_COLLECTION not in existing:
            await self._client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )

    def create_points(self, chunks: list[dict]) -> list[PointStruct]:
        """Create list of PointStruct from chunks."""
        points = []
        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=chunk["metadata"].get("section_no", i),
                vector=chunk["metadata"]["embedding"],
                payload={
                    "content": chunk["content"],
                    "metadata": {
                        k: v for k, v in chunk["metadata"].items() 
                        if k != "embedding"
                    }
                }
            )
            points.append(point)
        return points

    async def _upsert_batch(self, batch: list[PointStruct]) -> None:
        """Upsert a single batch."""
        await self._client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            wait=True,
            points=batch
        )

    async def add_vectors(self, chunks: list[dict]) -> None:
        """Upsert chunks into Qdrant in batches, concurrently."""
        points = self.create_points(chunks)
        
        # Split into batches
        batches = [
            points[i:i + self._batch_size] 
            for i in range(0, len(points), self._batch_size)
        ]
        
        # Upsert all batches in parallel
        await asyncio.gather(*[
            self._upsert_batch(batch) for batch in batches
        ])