import asyncio
import json
from loguru import logger
from pathlib import Path

from src.core.cli import parse_args
from src.indexing.chunking import run_chunker
from src.indexing.embedder import Embedder
from src.indexing.llama_cloud_parser import LlamaCloudParser
from src.indexing.metadata_enrichment import MetadataEnrichment
from src.indexing.parent_child import SectionPipeline
from src.indexing.table_parser import TableParser
from src.indexing.vector_store import VectorStore
from src.inference.chunks_retrieval import ChunkRetrieval
from src.inference.context_build import Context
from src.inference.generate_answer import Generation
from src.inference.query_enhancement import QueryEnhancement
from src.inference.build_tree import TreeBuilder

async def ingestion(directory: str = "data/raw_filings/"):
    parser = LlamaCloudParser(tier="agentic")
    parsed_files = await parser.parse_pdfs(directory)

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
    vector_store = VectorStore()
    retriever = ChunkRetrieval(vector_store=vector_store)
    context_builder = Context()
    generator = Generation()
    tree_builder = TreeBuilder(vector_store=vector_store)

    rewritten = await enhancer.query_rewrite(user_query)

    logger.info("Original query: {}", user_query)
    logger.info("Rewritten query: {}", rewritten)

    results = await retriever.search(rewritten, limit=top_k)
    Path("logs").mkdir(parents=True, exist_ok=True)
    with open("logs/retrieved_chunks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("saved builded tree")
    logger.info("Retrieved {} chunks", len(results))

    hey = await tree_builder.build_tree(results)
    Path("logs").mkdir(parents=True, exist_ok=True)
    with open("logs/build_tree.json", "w", encoding="utf-8") as f:
        json.dump(hey, f, indent=2, ensure_ascii=False)
    logger.info("saved builded tree")

    context = context_builder.build_context(results)

    print(f"\nQuery: {user_query}")
    print(f"Rewritten: {rewritten}")

    print("\nAnswer:\n")
    async for token in generator.generate_answer(context, user_query):
        print(token, end="", flush=True)
    print()


def main():
    args = parse_args()

    if args.command == "ingestion":
        asyncio.run(ingestion(directory=args.directory))
    elif args.command == "query":
        asyncio.run(query(args.query, top_k=args.top_k))


if __name__ == "__main__":
    main()
