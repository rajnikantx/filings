from typing import List

from src.ingestion.vectorizer import Vectorizer
from src.ingestion.encoder import Encoder
from src.retrieval.query_enhancement import QueryEnhancement

encoder = Encoder()
vec = Vectorizer(encoder=encoder)

class Retrieval:
    def __init__(self):
        self.query_enhancer = QueryEnhancement()

    async def retrieve_chunks(
        self,
        query: str,
        filters: dict | None = None,
        limit: int = 5,
    ) -> List[dict]:
        results = vec.search(query=query, filters=filters, limit=limit)

        chunks = []
        for point in results:
            payload = point.payload or {}
            chunk = {
                "content": payload.get("page_content", ""),
                "score": point.score,
                "metadata": {
                    k: v for k, v in payload.items() if k != "page_content"
                },
            }
            chunks.append(chunk)

        return chunks

    async def retrieve_chunks_enhanced(
        self,
        query: str,
        strategy: str = "hyde",
        filters: dict | None = None,
        limit: int = 5,
    ) -> List[dict]:
        if strategy == "hyde":
            query = await self.query_enhancer.hyde(query)
        elif strategy == "rewrite":
            query = await self.query_enhancer.query_rewrite(query)
        elif strategy == "step_back":
            query = await self.query_enhancer.step_back(query)

        return await self.retrieve_chunks(query=query, filters=filters, limit=limit)