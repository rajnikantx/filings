import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud
from loguru import logger

from src.prompts.llamaparse import LLAMAPARSE_PROMPT

load_dotenv()

logger.add("logs/parser.log", rotation="1 MB", level="INFO")


class LlamaCloudParser:
    def __init__(self, tier: str):
        key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not key:
            raise ValueError("Set LLAMA_CLOUD_API_KEY in .env file")
        
        self._client = AsyncLlamaCloud(api_key=key)
        self._tier = tier

    async def parse_and_save(
        self,
        pdf_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if pdf_dir.is_dir():
            files = list(pdf_dir.glob("*.pdf"))
            if not files:
                raise FileNotFoundError("Upload pdf file only")
            logger.info("Found {} PDFs in {}", len(files), pdf_dir)
        else:
            files = [pdf_dir]

        
        results = await asyncio.gather(
            *[self._parse(f) for f in files],
            return_exceptions=True
        )

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

            content_path = output_dir / "content" / f"{file_path.stem}.md"
            intro_path = output_dir / "intro" / f"{file_path.stem}.md"

            self._save_markdown(full_md, content_path)
            self._save_markdown(intro_md, intro_path)

            logger.info("Saved content + intro for {}", file_path.name)

    async def _parse(self, pdf_path: str | Path):
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

    @staticmethod
    def _save_markdown(markdown: str, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.debug("Wrote {} bytes to {}", len(markdown), save_path.name)


if __name__ == "__main__":
    parser = LlamaCloudParser(tier="agentic")
    asyncio.run(parser.parse_and_save("data/raw_filings/", "outputs/"))