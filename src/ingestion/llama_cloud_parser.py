import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud
from loguru import logger

from src.prompts.llamaparse import LLAMAPARSE_PROMPT

load_dotenv()

class LlamaCloudParser:
    def __init__(self, tier: str):
        key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not key:
            raise ValueError("Set LLAMA_CLOUD_API_KEY in .env file")
        
        self._client = AsyncLlamaCloud(api_key=key)
        self._tier = tier

    async def parse_pdfs(
        self,
        pdf_dir: str | Path,
    ) -> list[tuple[Path, str, str]]:
        pdf_dir = Path(pdf_dir)

        if pdf_dir.is_dir():
            files = list(pdf_dir.glob("*.pdf"))
            if not files:
                raise FileNotFoundError("Upload pdf file only")
            logger.info("Found {} PDFs in {}", len(files), pdf_dir)
        else:
            files = [pdf_dir]

        results = await asyncio.gather(
            *[self._parse_file(f) for f in files],
            return_exceptions=True
        )

        parsed: list[tuple[Path, str, str]] = []
        for file_path, result in zip(files, results):
            if isinstance(result, Exception):
                logger.error("Parse failed for {}: {}", file_path.name, result)
                continue

            pages = result.markdown.pages
            full_md = "\n\n".join(page.markdown for page in pages)

            if len(pages) >= 2:
                intro_md = pages[0].markdown + "\n\n" + pages[1].markdown
            else:
                intro_md = pages[0].markdown if pages else ""

            content_path = Path(f"logs/full_content/{file_path.stem}.md")
            intro_path = Path(f"logs/intro/{file_path.stem}_intro.md")
            
            content_path.parent.mkdir(parents=True, exist_ok=True)
            intro_path.parent.mkdir(parents=True, exist_ok=True)

            content_path.write_text(full_md, encoding="utf-8")
            intro_path.write_text(intro_md, encoding="utf-8")

            logger.info("Saved {} and {}", content_path, intro_path)
            parsed.append((file_path, full_md, intro_md))

        return parsed
        
    async def _parse_file(self, pdf_path: str | Path):
        path = Path(pdf_path)

        if not path.is_file():
            logger.error("File not found: {}", path)
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            logger.info("Uploading: {}", path.name)
            file = await self._client.files.create(
                file=path,
                purpose="parse",
            )
            logger.info("Parsing: {}", path.name)
            result = await self._client.parsing.parse(
                file_id=file.id,
                tier=self._tier,
                version="latest",
                expand=["markdown"],
                agentic_options={
                    "custom_prompt": LLAMAPARSE_PROMPT,
                },
            )
            logger.info("Done: {}", path.name)
            return result

        except Exception as e:
            logger.error("Failed {}: {}", path.name, e)
            raise


if __name__ == "__main__":
    parser = LlamaCloudParser(tier="agentic")
    asyncio.run(parser.parse_pdfs("data/raw_filings/"))