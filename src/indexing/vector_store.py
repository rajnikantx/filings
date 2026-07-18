import hashlib
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

from src.config import settings


class VectorStore:
    def __init__(self, collection_name: str | None = None, vector_size: int | None = None):
        self._client = AsyncQdrantClient(
            path=settings.QDRANT_URL,
        )
        self._collection_name = collection_name or settings.QDRANT_COLLECTION
        self._vector_size = vector_size or settings.QDRANT_VECTOR_SIZE

    @property
    def collection_name(self) -> str:
        return self._collection_name

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

    @staticmethod
    def _chunk_id(chunk: dict) -> str:
        key = f"{chunk['metadata']['file_id']}::{chunk['metadata']['section_no']}"
        return str(uuid.UUID(hashlib.sha256(key.encode()).hexdigest()[:32]))

    async def upsert_chunks(self, chunks: list[dict]):
        points = [
            PointStruct(
                id=chunk.get("id") or self._chunk_id(chunk),
                vector=chunk["embedding"],
                payload={
                    "content": chunk["content"],
                    **{
                        k: v for k, v in chunk.items()
                        if k not in ("content", "id", "embedding")
                    },
                },
            )
            for chunk in chunks
        ]

        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Filter | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        results = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=filters,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"score": point.score, "payload": point.payload}
            for point in results.points
        ]