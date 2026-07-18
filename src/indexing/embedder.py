import asyncio
import tiktoken
from openai import AsyncOpenAI, RateLimitError

from src.config import settings
from src.core.rate_limiter import RateLimiter


class Embedder:
    def __init__(
        self,
        max_tokens_per_batch: int = 8000,
        max_concurrency: int = 5,
        max_rpm: int = 3000,
        max_tpm: int = 1_000_000,
    ):
        self._embedding_model = settings.EMBEDDING_MODEL
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._max_tokens_per_batch = max_tokens_per_batch
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_limiter = RateLimiter(max_rpm, max_tpm)

        try:
            self._encoder = tiktoken.encoding_for_model(self._embedding_model)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _make_chunk_batches(self, chunks: list[dict]) -> list[list[dict]]:
        batches: list[list[dict]] = []
        current_batch: list[dict] = []
        current_tokens = 0

        for chunk in chunks:
            text_tokens = self._count_tokens(chunk["content"])

            if current_batch and current_tokens + text_tokens > self._max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(chunk)
            current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _embed_batch(self, chunks: list[dict]) -> None:
        """Mutates each chunk in place, attaching its embedding under 'embedding'."""
        texts = [c["content"] for c in chunks]
        batch_tokens = sum(self._count_tokens(t) for t in texts)
        max_retries = 5
        delay = 1.0

        async with self._semaphore:
            await self._rate_limiter.acquire(batch_tokens)

            for attempt in range(max_retries):
                try:
                    response = await self._client.embeddings.create(
                        model=self._embedding_model,
                        input=texts,
                    )
                    for chunk, item in zip(chunks, response.data):
                        chunk["embedding"] = item.embedding
                    return
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2

        raise RuntimeError("unreachable")

    async def embed_all(self, chunks: list[dict]) -> list[dict]:
        """Attaches an 'embedding' key to each chunk in place, returns the same list."""
        batches = self._make_chunk_batches(chunks)
        await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        return chunks

    async def embed_query(self, text: str) -> list[float]:
        temp_chunk = [{"content": text}]
        await self._embed_batch(temp_chunk)
        return temp_chunk[0]["embedding"]