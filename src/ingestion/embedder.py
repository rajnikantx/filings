import asyncio
import copy
from openai import AsyncOpenAI

from src.config import settings


class Embedding:
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in the environment settings.")
            
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.EMBEDDING_MODEL
        self._batch_size = getattr(settings, "EMBEDDING_BATCH_SIZE", 100)
        self._semaphore = asyncio.Semaphore(10)

    async def _chunk_batch(self, chunks: list[dict]) -> list[list[str]]:
        """Split chunks into text batches."""
        batches, current = [], []
        for chunk in chunks:
            current.append(chunk['content'])
            if len(current) == self._batch_size:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches

    async def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed single batch with rate limiting."""
        async with self._semaphore:
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch
            )
            return [d.embedding for d in response.data]

    async def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """Generate embeddings in parallel. Returns new list, doesn't modify input."""
        result = copy.deepcopy(chunks)
        
        text_batches = await self._chunk_batch(result)
        
        # Parallel embedding
        all_results = await asyncio.gather(*[
            self._embed_one_batch(batch) for batch in text_batches
        ])
        
        # Flatten and assign
        all_embeddings = []
        for batch_result in all_results:
            all_embeddings.extend(batch_result)
            
        for i, emb in enumerate(all_embeddings):
            if i < len(result):
                result[i]['metadata']['embedding'] = emb
                
        return result