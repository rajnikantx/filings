import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import settings


class VectorStore:
    def __init__(self, collection_name: str | None = None, vector_size: int | None = None):
        self._client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self._collection_name = collection_name or settings.QDRANT_COLLECTION
        self._vector_size = vector_size or settings.QDRANT_VECTOR_SIZE

    async def ensure_collection(self):
        exists = await self._client.collection_exists(self._collection_name)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        points = [
            PointStruct(
                id=chunk.get("id", str(uuid.uuid4())),
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    **{k: v for k, v in chunk.items() if k not in ("content", "id")},
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        results = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
        )
        return [
            {"score": point.score, "payload": point.payload}
            for point in results.points
        ]