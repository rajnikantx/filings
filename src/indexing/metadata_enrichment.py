import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from src.config import settings
from src.models import SECFilingDetails

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


class MetadataEnrichment:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=60.0)
        self._model = settings.METADATA_MODEL

    async def enrich_metadata(self, intro_content: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._client.responses.parse(
                    model=self._model,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert financial analyst. Extract SEC filing "
                                "details accurately. Convert company_name and ticker "
                                "values to UPPERCASE letters."
                            ),
                        },
                        {
                            "role": "user",
                            "content": intro_content,
                        },
                    ],
                    temperature=0,
                    text_format=SECFilingDetails,
                )

                parsed = response.output_parsed
                if parsed is None:
                    raise ValueError("Model did not return a parsed response")

                return parsed.model_dump(mode="json")

            except (APITimeoutError, APIConnectionError) as error:
                last_error = error
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Enrichment attempt {}/{} failed: {}. Retrying in {:.1f}s...",
                        attempt, _MAX_RETRIES, error, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Enrichment failed after {} attempts: {}", _MAX_RETRIES, error
                    )
                    raise

            except Exception as error:
                logger.error("Failed to enrich metadata: {}", error)
                raise

        raise last_error  # type: ignore[misc]

    async def enrich_all(
        self, parsed_files: Sequence[tuple[Path, str, str]]
    ) -> dict[str, dict]:
        logger.info("Enriching metadata for {} file(s)", len(parsed_files))

        tasks = [
            self.enrich_metadata(intro_content=intro)
            for _, _, intro in parsed_files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        metadata_by_file: dict[str, dict] = {}
        for (file_path, _, _), result in zip(parsed_files, results):
            if isinstance(result, Exception):
                logger.error(
                    "Metadata enrichment failed for {}: {}", file_path, result
                )
                continue
            metadata_by_file[file_path.stem] = result

        logger.info(
            "Enriched metadata for {}/{} files",
            len(metadata_by_file), len(parsed_files),
        )

        log_path = Path("logs/metadata_enriched.json")
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                json.dumps(metadata_by_file, indent=2, default=str), encoding="utf-8"
            )
            logger.info("Saved enriched metadata to {}", log_path)
        except OSError as error:
            logger.error("Failed to write metadata log to {}: {}", log_path, error)

        return metadata_by_file