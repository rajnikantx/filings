import asyncio

from loguru import logger

from src.ingestion.chunking import run_chunker
from src.ingestion.llama_cloud_parser import LlamaCloudParser
from src.ingestion.metadata_enrichment import MetadataEnrichment
from src.ingestion.parent_child import SectionPipeline
from src.ingestion.table_parser import TableParser

if __name__ == "__main__":

    async def main():
        parser = LlamaCloudParser(tier="agentic")
        parsed_files = await parser.parse_pdfs("data/raw_filings/")

        enricher = MetadataEnrichment()
        metadata_by_file = await enricher.enrich_all(parsed_files)

        pipeline = SectionPipeline()
        sections_by_file = pipeline.execute(parsed_files, metadata_by_file)

        all_chunks = await run_chunker(sections_by_file)
        logger.info("Total chunks after chunking: {}", len(all_chunks))

        table_parser = TableParser()
        all_chunks = await table_parser.parse(all_chunks)
        logger.info("Table processing complete. Final chunks saved to logs/final_chunks.json")

        print(f"\nDone. Total chunks: {len(all_chunks)}")

    asyncio.run(main())