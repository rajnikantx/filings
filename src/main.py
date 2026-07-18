import asyncio
import json
from pathlib import Path

from loguru import logger

from src.indexing.chunking import run_chunker
from src.indexing.llama_cloud_parser import LlamaCloudParser
from src.indexing.metadata_enrichment import MetadataEnrichment
from src.indexing.parent_child import SectionPipeline
from src.indexing.table_parser import TableParser
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.inference.query_enhancement import QueryEnhancement
from src.inference.chunks_retrieval import ChunkRetrieval
from src.inference.context_build import Context
from src.inference.generate_answer import Generation

MODE = "query"  # "ingestion" or "query"
QUERY = "what does this sec filings about"


async def ingestion():
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
    logger.info("Table processing complete")

    embedder = Embedder()
    all_chunks = await embedder.embed_all(all_chunks)
    logger.info("Embedding complete for {} chunks", len(all_chunks))

    vector_store = VectorStore()
    await vector_store.ensure_collection()
    await vector_store.upsert_chunks(all_chunks)
    logger.info("Upsert complete. {} chunks stored in Qdrant", len(all_chunks))

    print(f"\nDone. Total chunks: {len(all_chunks)}")


async def query(user_query: str, top_k: int = 5):
    enhancer = QueryEnhancement()
    retriever = ChunkRetrieval()
    context_builder = Context()
    generator = Generation()

    rewritten = await enhancer.query_rewrite(user_query)

    logger.info("Original query: {}", user_query)
    logger.info("Rewritten query: {}", rewritten)

    results = await retriever.search(rewritten, limit=top_k)

    logger.info("Retrieved {} chunks", len(results))

    context = context_builder.build_context(results)
    answer = generator.generate_answer(context, user_query)

    print(f"\nQuery: {user_query}")
    print(f"Rewritten: {rewritten}")
    print(f"\nAnswer:\n{answer}")

    return answer


if __name__ == "__main__":
    if MODE == "ingestion":
        asyncio.run(ingestion())
    elif MODE == "query":
        asyncio.run(query(QUERY))