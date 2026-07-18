import io
import os
import json
import asyncio
import hashlib
import pandas as pd
from loguru import logger
from openai import AsyncOpenAI

from src.config import settings


class TableParser:
    def __init__(self):
        self._model = settings.QUERY_MODEL
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def _html_table_to_df(self, table: str):
        try:
            df_list = pd.read_html(io.StringIO(table), flavor="lxml")
            df = df_list[0]

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    "_".join(str(level).strip() for level in col if "Unnamed:" not in str(level))
                    for col in df.columns
                ]
            else:
                df.columns = [str(col).strip() if "Unnamed:" not in str(col) else "" for col in df.columns]

            df.columns = [f"Line_Item_{i}" if col == "" else col for i, col in enumerate(df.columns)]
            df = df.map(lambda x: " ".join(str(x).split()) if pd.notna(x) else x)

            return df

        except Exception as e:
            logger.error(f"Error processing table: {str(e)}")
            return None

    async def _get_table_description(self, table_content, document_context):
        prompt = f"""
        Given the following table and its context from the original document,
        provide a detailed description of the table. Then, include the table in markdown format.

        Original Document Context:
        {document_context}

        Table Content:
        {table_content}

        Please provide:
        1. A comprehensive description of the table.
        2. The table in markdown format.
        """

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes tables and formats them in markdown."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def _build_document_context(self, chunks: list[dict], index: int, window: int = 2) -> str:
        preceding = []
        start = max(0, index - window)
        for c in chunks[start:index]:
            if not c["metadata"].get("has_table", False):
                preceding.append(c["content"])

        succeeding = []
        end = min(len(chunks), index + window + 1)
        for c in chunks[index + 1:end]:
            if not c["metadata"].get("has_table", False):
                succeeding.append(c["content"])

        return "\n\n".join(preceding + succeeding)

    @staticmethod
    def _hash_table(table_content: str) -> str:
        return hashlib.sha256(table_content.encode("utf-8")).hexdigest()

    async def parse(self, chunks: list[dict]) -> list[dict]:
        os.makedirs(settings.EXTRACTED_CSV_PATH, exist_ok=True)

        table_indices = []      # indices into `chunks` that are tables
        description_tasks = []  # matching coroutines, same order as table_indices

        # Pass 1: mark non-tables, compute ids/context/CSVs, queue up description tasks
        for index, chunk in enumerate(chunks):
            if chunk["metadata"]["has_table"] == False:
                chunk["metadata"]["table_id"] = None
                continue

            table_content = chunk["content"]
            table_id = self._hash_table(table_content)
            chunk["metadata"]["table_id"] = table_id

            document_context = self._build_document_context(chunks, index)

            df = await self._html_table_to_df(table_content)

            if df is not None:
                csv_path = f"{settings.EXTRACTED_CSV_PATH}/{table_id}.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8")
            else:
                logger.warning(f"Skipping CSV export for table {table_id}: could not parse table content")

            table_indices.append(index)
            description_tasks.append(
                self._get_table_description(
                    table_content=table_content,
                    document_context=document_context,
                )
            )

        # Pass 2: fire all LLM description calls concurrently
        if description_tasks:
            results = await asyncio.gather(*description_tasks, return_exceptions=True)

            for idx, result in zip(table_indices, results):
                table_id = chunks[idx]["metadata"]["table_id"]

                if isinstance(result, Exception):
                    logger.error(f"Table description failed for chunk index {idx} (table_id={table_id}): {result}")
                    continue

                chunks[idx]["content"] = result
                logger.info(f"Table description generated successfully for chunk index {idx} (table_id={table_id})")

        output_path = "logs/final_chunks.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Total tables processed: {len(table_indices)}")

        return chunks