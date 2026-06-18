import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from src.config import settings
from src.ingestion.encoder import Encoder


class Vectorizer:
    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder
        self._client = QdrantClient(path = settings.QDRANT_URL)
        self._collection = settings.QDRANT_COLLECTION
        self._query_cache: dict[str, list[float]] = {}

    def create_collection(self) -> None:
        """Create collection only if it doesn't already exist."""
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=rest_models.VectorParams(
                    size=self._encoder.dim,
                    distance=rest_models.Distance.COSINE,
                ),
            )
            print(f"Collection '{self._collection}' created.")
        else:
            print(f"Collection '{self._collection}' already exists.")

    def ingest(self, chunks: list[dict]) -> None:
        """Embed and upsert chunks into Qdrant."""
        texts = [c["content"] for c in chunks]
        vectors = self._encoder.encode(texts)

        points = []
        for idx, chunk in enumerate(chunks):
            payload = {"page_content": chunk["content"], **chunk["metadata"]}
            points.append(
                rest_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors[idx],
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self._collection, points=points)
        print(f"Upserted {len(points)} points into '{self._collection}'.")

    def search(
        self, query: str, filters: dict | None = None, limit: int = 5
    ) -> list[rest_models.ScoredPoint]:
        """Search the existing collection."""
        # Check cache first
        if query in self._query_cache:
            query_vector = self._query_cache[query]
        else:
            query_vector = self._encoder.encode_query(query)
            self._query_cache[query] = query_vector

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    rest_models.FieldCondition(
                        key=key,
                        match=rest_models.MatchValue(value=value),
                    )
                )
            qdrant_filter = rest_models.Filter(must=conditions)

        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
        )
        return result.points
    


if __name__ == "__main__":
    encoder = Encoder()
    vec = Vectorizer(encoder)

    vec.create_collection()

    chunks = [
        {
            "content": "1 Tesla Road, Austin, Texas 78725",
            "metadata": {
                "ticker": "TSLA",
                "fiscal_year": 2025,
                "source_file": "tsla.json",
            },
        },
     
    ]

    vec.ingest(chunks)
    print("Ingestion complete.")