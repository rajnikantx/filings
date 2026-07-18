from loguru import logger
from openai import OpenAI

from src.config import settings


class Generation:
    def __init__(self):
        self._model = "gpt-4o"
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_answer(self, context: str, query: str) -> str:
        system_prompt = (
            "You are a financial analyst assistant specializing in SEC filings. "
            "Answer the user's question using ONLY the provided context. "
            "Rules:\n"
            "- Be precise and cite specific numbers, dates, and facts from the context\n"
            "- If the context doesn't contain enough information, say so clearly\n"
            "- Do not hallucinate or make up information\n"
            "- Keep answers concise and to the point"
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )

            answer = response.choices[0].message.content
            logger.info("Answer generated ({} chars)", len(answer))
            return answer

        except Exception as e:
            logger.error("Answer generation failed: {}", e)
            return f"Error generating answer: {e}"