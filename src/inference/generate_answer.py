from collections.abc import AsyncGenerator

from loguru import logger
from openai import AsyncOpenAI

from src.config import settings

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant specializing in SEC filings. "
    "Answer the user's question using ONLY the provided context. "
    "Rules:\n"
    "- Be precise and cite specific numbers, dates, and facts from the context\n"
    "- If the context doesn't contain enough information, say so clearly\n"
    "- Do not hallucinate or make up information\n"
    "- Keep answers concise and to the point"
)


class Generation:
    def __init__(self):
        self._model = settings.QUERY_MODEL
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _build_messages(self, context: str, query: str) -> list[dict]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

    async def generate_answer(
        self, context: str, query: str
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(context, query),
                temperature=0,
                stream=True,
            )
            full_answer = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer += delta
                    yield delta
            logger.info("Answer streamed ({} chars)", len(full_answer))
        except Exception as e:
            logger.error("Streaming failed: {}", e)
            yield f"\n\nError: {e}"
