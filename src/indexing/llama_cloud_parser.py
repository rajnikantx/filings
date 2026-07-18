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
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("Set LLAMA_CLOUD_API_KEY in .env file")

        self._client = AsyncLlamaCloud(api_key=api_key)
        self._tier = tier

    async def parse_pdfs(
        self,
        pdf_dir: str | Path,
    ) -> list[tuple[Path, str, str]]:
        pdf_dir = Path(pdf_dir)

        if pdf_dir.is_dir():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError("Upload pdf file only")
            logger.info("Found {} PDFs in {}", len(pdf_files), pdf_dir)
        else:
            if not pdf_dir.is_file():
                raise FileNotFoundError(f"PDF not found: {pdf_dir}")
            pdf_files = [pdf_dir]

        parse_results = await asyncio.gather(
            *[self._parse_file(pdf_file) for pdf_file in pdf_files],
            return_exceptions=True,
        )

        parsed_files: list[tuple[Path, str, str]] = []
        for pdf_file, parse_result in zip(pdf_files, parse_results):
            if isinstance(parse_result, Exception):
                logger.error("Parse failed for {}: {}", pdf_file.name, parse_result)
                continue

            pages = parse_result.markdown.pages
            full_markdown = "\n\n".join(page.markdown for page in pages)

            if len(pages) >= 2:
                intro_markdown = pages[0].markdown + "\n\n" + pages[1].markdown
            else:
                intro_markdown = pages[0].markdown if pages else ""

            content_path = Path(f"logs/full_content/{pdf_file.stem}.md")
            intro_path = Path(f"logs/intro/{pdf_file.stem}_intro.md")

            content_path.parent.mkdir(parents=True, exist_ok=True)
            intro_path.parent.mkdir(parents=True, exist_ok=True)

            content_path.write_text(full_markdown, encoding="utf-8")
            intro_path.write_text(intro_markdown, encoding="utf-8")

            logger.info("Saved {} and {}", content_path, intro_path)
            parsed_files.append((pdf_file, full_markdown, intro_markdown))

        return parsed_files

    async def _parse_file(self, pdf_path: str | Path):
        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            logger.error("File not found: {}", pdf_path)
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            logger.info("Uploading: {}", pdf_path.name)
            uploaded_file = await self._client.files.create(
                file=pdf_path,
                purpose="parse",
            )
            logger.info("Parsing: {}", pdf_path.name)
            parse_result = await self._client.parsing.parse(
                file_id=uploaded_file.id,
                tier=self._tier,
                version="latest",
                expand=["markdown"],
                agentic_options={
                    "custom_prompt": LLAMAPARSE_PROMPT,
                },
            )
            logger.info("Done: {}", pdf_path.name)
            return parse_result

        except Exception as error:
            logger.error("Failed {}: {}", pdf_path.name, error)
            raise


if __name__ == "__main__":
    parser = LlamaCloudParser(tier="agentic")
    asyncio.run(parser.parse_pdfs("data/raw_filings/"))