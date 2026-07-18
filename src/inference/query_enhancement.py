from openai import AsyncOpenAI
from loguru import logger

from src.config import settings

class QueryEnhancement:
    def __init__(self):
        self._model = settings.QUERY_MODEL
        if not settings.OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY in .env file")
        self._client = AsyncOpenAI(api_key = settings.OPENAI_API_KEY)
        logger.info(f"QueryEnhancement initialized | model = {self._model}")

    async def hyde(self, query: str) -> str:
        if not query:
            logger.warning("HYDE rejected: empty query")
            raise ValueError("Empty input. enter valid input")
        
        logger.info(f"HYDE starting...")
        try:
            response = await self._client.responses.create(
                model= self._model,
                reasoning= {"effort": "high"},
                input= [
                    {
                        "role": "developer",
                        "content":"You are a SEC filings analyst. Given a user query, "
                            "write a detailed hypothetical passage that would appear "
                            "in an SEC filing and directly answers the query. "
                            "Include specific numbers, dates, and financial details. "
                            "Do NOT mention this is hypothetical."
                    },
                    {
                        "role": "user",
                        "content": f"query: {query} \n\n Write a hypothetical SEC filing excerpt."
                    }
                ]
            )
            hyde_doc = response.output[0].content[0].text

            if not hyde_doc:
                logger.warning("HYDE returned empty result")
                return query
            
            logger.info(f"HYDE completed")
            return hyde_doc
        
        except Exception as e:
            logger.error(f"HYDE API failed: {e}")
            return query



    async def query_rewrite(self, query: str) -> str : 
        if not query:
            logger.warning("Query Rewriting rejected: empty Query")
            raise ValueError("Empty input. enter valid input")

        logger.info(f"Query Rewriting starting...")
        try:
            response = await self._client.responses.create(
                model= settings.QUERY_MODEL,
                input= [
                    {
                        "role": "developer",
                        "content": "You are a query rewriting assistant for a SEC filings search system. "
                        "Your task: rewrite the user query to be precise, explicit, and optimized for "
                        "retrieval from SEC filings (10-K, 10-Q, 8-K). "
                        "Rules:\n"
                        "- Replace colloquial terms with formal SEC/financial terminology\n"
                        "- Add missing entity names if the query implies a known company\n"
                        "- Include specific document types if implied (10-K, 10-Q)\n"
                        "- Do NOT answer the query. Only output the rewritten query text\n"
                        "- Output ONLY the rewritten query, nothing else"
                    },
                    {
                        "role": "user",
                        "content": f"Rewrite this query for SEC filing retrieval: \n\nQuery: {query} "
                    }
                ]
            )
            rewritten_query = response.output[0].content[0].text

            if not rewritten_query:
                logger.warning("Rewritten returned empty result")
                return query
            
            logger.info(f"Query rewriting completed")
            return rewritten_query
        
        except Exception as e:
            logger.error(f"query rewritten API failed: {e}")
            return query
        


    async def step_back(self, query: str) -> str:
        if not query or not query.strip():
            logger.warning("Step-back rejected: empty query")
            raise ValueError("Empty input. Enter a valid query.")

        logger.info(f"Step-back starting | query='{query[:50]}...'")

        try:
            response = await self._client.responses.create(
                model=settings.QUERY_MODEL,
                input=[
                    {
                        "role": "developer",
                        "content": (
                            "You are a SEC filings analyst. Your task: abstract the user's specific query "
                            "into a broader, more general question that captures the high-level concept. "
                            "This broader query will be used to retrieve relevant SEC filing sections. "
                            "Rules:\n"
                            "- Replace specific product names with general business segments\n"
                            "- Replace colloquial terms with formal SEC/financial terminology\n"
                            "- Focus on the underlying concept, not the specific detail\n"
                            "- The output should be a question or topic, not an answer\n"
                            "- Output ONLY the step-back query, nothing else"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Generate a broader, more general query for SEC filing retrieval:\n\nSpecific query: {query}"
                    }
                ]
            )
            
            stepped_back = response.output[0].content[0].text.strip()
            
            if not stepped_back or len(stepped_back) < 3:
                logger.warning("Step-back returned empty/short result, falling back")
                return query
                    
            logger.info(f"Step-back completed | result='{stepped_back[:80]}...'")
            return stepped_back

        except Exception as e:
            logger.error(f"Step-back API failed: {e}")
            return query