import asyncio

from src.ingestion.llama_cloud_parser import LlamaCloudParser


async def main() -> None:
    parser = LlamaCloudParser(tier="agentic")
    results = await parser.parse_pdfs("data/raw_filings/")


if __name__ == "__main__":
    asyncio.run(main())
