import asyncio

from src.ingestion.chunking import run_chunker
from src.ingestion.llama_cloud_parser import LlamaCloudParser
from src.ingestion.metadata_enrichment import MetadataEnrichment
from src.ingestion.parent_child import SectionPipeline

if __name__ == "__main__":

    async def main():
        parser = LlamaCloudParser(tier="agentic")
        parsed_files = await parser.parse_pdfs("data/raw_filings/")

        enricher = MetadataEnrichment()
        metadata_by_file = await enricher.enrich_all(parsed_files)

        pipeline = SectionPipeline()
        sections_by_file = pipeline.execute(parsed_files, metadata_by_file)

        all_chunks = await run_chunker(sections_by_file)
        print(f"\nDone. Total chunks: {len(all_chunks)}")

    asyncio.run(main())