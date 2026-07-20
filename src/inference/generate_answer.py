import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from loguru import logger
from openai import OpenAI

from src.config import settings

_SYSTEM_PROMPT = (
    "You are a senior financial analyst assistant specializing in SEC filings "
    "(10-K, 10-Q, 8-K, proxy statements, and related disclosures).\n\n"
    "You will be given two sources of information for each query:\n"
    "1. CONTEXT — retrieved text passages from SEC filings (narrative disclosures, "
    "MD&A, footnotes, risk factors, etc.)\n"
    "2. CSV DATA FILES — structured financial tables (e.g., income statements, "
    "balance sheets, cash flow statements, segment data) attached via the code "
    "interpreter tool.\n\n"
    "Your task is to answer the user's question by synthesizing BOTH sources "
    "together, not just one. Treat the CSV data as the authoritative source for "
    "any numerical, tabular, or time-series figures, and use the CONTEXT to "
    "explain, qualify, or provide narrative reasoning behind those figures.\n\n"
    "Rules of engagement:\n"
    "- Ground every factual claim in either the provided context or the CSV data. "
    "Never rely on outside/prior knowledge of the company or its filings.\n"
    "- When you reference structured data, be explicit: name the file, and cite "
    "the relevant row(s), column(s), or metric(s) and the period they refer to "
    "(e.g., 'per the CSV, Net Revenue for FY2023 in row 4 was $12.4B').\n"
    "- When you reference narrative context, cite the specific fact, figure, or "
    "statement as it appears, without directly reproducing large verbatim blocks.\n"
    "- If context and CSV data appear to conflict, flag the discrepancy explicitly "
    "rather than silently reconciling it.\n"
    "- If neither source contains sufficient information to answer confidently, "
    "state this clearly instead of speculating or filling gaps with assumptions.\n"
    "- Do not hallucinate figures, dates, filing details, or company facts.\n"
    "- Use precise financial terminology (e.g., GAAP vs. non-GAAP, YoY vs. QoQ) "
    "and specify units and reporting periods for every figure you cite.\n"
    "- Keep the answer concise, well-structured, and analyst-grade — lead with "
    "the direct answer, then support it with the specific evidence used."
)


class Generation:
    def __init__(self):
        self._model = settings.QUERY_MODEL
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _upload_csvs(self, csv_paths: list[Path]) -> list[str]:
        file_ids: list[str] = []
        for path in csv_paths:
            try:
                with open(path, "rb") as f:
                    resp = self._client.files.create(file=f, purpose="assistants")
                file_ids.append(resp.id)
                logger.info("Uploaded CSV: {} -> {}", path.name, resp.id)
            except Exception as e:
                logger.warning("Failed to upload CSV {}: {}", path, e)
        return file_ids

    def _delete_files(self, file_ids: list[str]) -> None:
        for fid in file_ids:
            try:
                self._client.files.delete(fid)
                logger.debug("Deleted file: {}", fid)
            except Exception as e:
                logger.warning("Failed to delete file {}: {}", fid, e)

    def _build_input(self, context: str, query: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Use the uploaded CSV files to support your analysis. "
            "Reference specific data points from the tables in your answer."
        )

    async def generate_answer(
        self, context: str, query: str, csv_paths: list[Path] | None = None
    ) -> AsyncGenerator[str, None]:
        file_ids: list[str] = []
        try:
            if csv_paths:
                file_ids = await asyncio.to_thread(self._upload_csvs, csv_paths)

            user_input = self._build_input(context, query)

            kwargs: dict = {
                "model": self._model,
                "instructions": _SYSTEM_PROMPT,
                "input": user_input,
            }

            if file_ids:
                kwargs["tools"] = [
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto", "file_ids": file_ids},
                    }
                ]

            response = await asyncio.to_thread(
                self._client.responses.create, **kwargs
            )

            for item in response.output:
                if getattr(item, "type", None) != "message":
                    continue
                for part in item.content:
                    if getattr(part, "type", None) == "output_text":
                        yield part.text

            logger.info("Answer generated ({} output items)", len(response.output))
        except Exception as e:
            logger.error("Generation failed: {}", e)
            yield f"\n\nError: {e}"
        finally:
            if file_ids:
                await asyncio.to_thread(self._delete_files, file_ids)