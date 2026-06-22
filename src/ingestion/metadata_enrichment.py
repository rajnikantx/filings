from __future__ import annotations

import asyncio
import json
import time

from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from src.models import SECFilingDetails
from src.config import settings


class MetadataEnrichment:
    _PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY in .env file")
        
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.METADATA_MODEL
        self._total_tokens = 0
        self._total_cost = 0.0

    async def enrich_folder(self, intro_dir: str | Path, output_path: str | Path) -> dict[str, dict]:
        start_total = time.perf_counter()
        
        intro_dir = Path(intro_dir)
        if not intro_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {intro_dir}")

        files = list(intro_dir.glob("*.txt")) + list(intro_dir.glob("*.md"))
        logger.info("Found {} intro files in {}", len(files), intro_dir)

        tasks = [self._enrich(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, dict] = {}
        for file_path, result in zip(files, results):
            if isinstance(result, Exception):
                logger.error("Failed {}: {}", file_path.name, result)
                output[file_path.name] = {
                    "success": False,
                    "metadata": None,
                    "error": str(result),
                }
            else:
                logger.info("Enriched {}", file_path.name)
                output[file_path.stem] = result

        elapsed = time.perf_counter() - start_total
        logger.info("Batch complete: {} files in {:.2f}s", len(files), elapsed)
        logger.info("Total tokens: {}, Estimated cost: ${:.4f}", self._total_tokens, self._total_cost)

        self._save_json(output, output_path)

        return output

    async def _enrich(self, path: Path) -> dict:
        start = time.perf_counter()
        
        text = path.read_text(encoding="utf-8")
        read_time = time.perf_counter() - start

        api_start = time.perf_counter()
        response = await self._client.responses.parse(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial analyst. Extract SEC filing details accurately. Convert company_name and ticker values to UPPERCASE letters.",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0,
            response_format=SECFilingDetails,
        )
        api_time = time.perf_counter() - api_start

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Model refused to parse the input")

        usage = response.usage
        tokens = usage.total_tokens if usage else 0
        cost = self._estimate_cost(usage)

        self._total_tokens += tokens
        self._total_cost += cost

        total = time.perf_counter() - start
        logger.debug(
            "Enriched {}: read={:.3f}s, api={:.3f}s, tokens={}, cost=${:.6f}, total={:.3f}s",
            path.name, read_time, api_time, tokens, cost, total
        )

        return parsed.model_dump(mode="json")

    def _estimate_cost(self, usage) -> float:
        if not usage or self._model not in self._PRICING:
            return 0.0
        
        rates = self._PRICING[self._model]
        input_cost = (usage.prompt_tokens / 1_000_000) * rates["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    def _save_json(self, results: dict[str, dict], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

if __name__ == "__main__":
    enricher = MetadataEnrichment()
    asyncio.run(enricher.enrich_folder("outputs/intro/", "outputs/metadata.json"))