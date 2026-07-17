import asyncio
import json
import logging
import re
from pathlib import Path

from src.ingestion.parent_child import flatten_sections

logger = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"(<table>.*?</table>)", re.DOTALL)
_NEXT_SEP = {"\n\n": "\n", "\n": "sentence", "sentence": None}


class Chunker:
    """Split section content into chunks and return them."""

    async def chunk(self, sections_by_file: dict[str, list[dict]]) -> list[dict]:
        """Process all files' sections concurrently and return flattened chunks."""
        if not sections_by_file:
            logger.warning("No sections provided to chunk")
            return []

        chunks_per_file = await asyncio.gather(
            *[
                asyncio.to_thread(self._process_file, file_stem, section_items)
                for file_stem, section_items in sections_by_file.items()
            ]
        )

        # Flatten list of per-file chunk lists into one list
        return [chunk_record for file_chunks in chunks_per_file for chunk_record in file_chunks]

    def _process_file(self, file_stem: str, section_items: list[dict]) -> list[dict]:
        """Chunk a single file's sections and return the chunks."""
        flat_sections = flatten_sections(section_items)
        file_chunks = []
        for section_item in flat_sections:
            section_content = section_item.get("content", "")
            section_metadata = section_item.get("metadata", {})

            if not section_content.strip():
                continue

            content_chunks = Chunker.chunk_text(section_content, max_chars=1000)

            for chunk_content in content_chunks:
                chunk_record = {
                    "content": chunk_content,
                    "metadata": {
                        **section_metadata,
                        "source_file": file_stem,
                        "has_table": bool(_TABLE_RE.search(chunk_content)),
                    }
                }
                file_chunks.append(chunk_record)

        logger.info("Produced %d chunks from %s", len(file_chunks), file_stem)
        return file_chunks

    @staticmethod
    def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
        """Split text into chunks. Tables are always kept whole."""
        if len(text) <= max_chars:
            return [text]

        text_chunks = []
        for segment in _TABLE_RE.split(text):
            if not segment:
                continue
            if segment.startswith("<table>") and segment.endswith("</table>"):
                text_chunks.append(segment)
            else:
                text_chunks.extend(Chunker._chunk_recursive(segment, max_chars, "\n\n"))
        return text_chunks

    @staticmethod
    def _chunk_recursive(text: str, max_chars: int, separator: str) -> list[str]:
        """Split text by separator. If a piece is still too big, recurse with a finer separator."""
        if len(text) <= max_chars:
            return [text]

        next_separator = _NEXT_SEP.get(separator)
        text_chunks = []

        for segment in text.split(separator):
            if not segment:
                continue
            if len(segment) <= max_chars:
                text_chunks.append(segment)
            elif next_separator is None:
                for start in range(0, len(segment), max_chars):
                    text_chunks.append(segment[start:start + max_chars])
            elif next_separator == "sentence":
                sentences = re.split(r"(?<=[.!?])\s+", segment)
                text_chunks.extend(Chunker._chunk_sentences(sentences, max_chars))
            else:
                text_chunks.extend(Chunker._chunk_recursive(segment, max_chars, next_separator))

        return text_chunks

    @staticmethod
    def _chunk_sentences(sentences: list[str], max_chars: int) -> list[str]:
        """Merge short sentences together. Hard-split any sentence that is too long."""
        sentence_chunks = []
        current_chunk = ""

        for sentence in sentences:
            candidate_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            if len(candidate_chunk) <= max_chars:
                current_chunk = candidate_chunk
            else:
                if current_chunk:
                    sentence_chunks.append(current_chunk)
                if len(sentence) <= max_chars:
                    current_chunk = sentence
                else:
                    for start in range(0, len(sentence), max_chars):
                        sentence_chunks.append(sentence[start:start + max_chars])
                    current_chunk = ""

        if current_chunk:
            sentence_chunks.append(current_chunk)
        return sentence_chunks


async def run_chunker(sections_by_file: dict[str, list[dict]]) -> list[dict]:
    """Instantiate and run the chunker."""
    chunker = Chunker()
    all_chunks = await chunker.chunk(sections_by_file)
    
    Path("logs").mkdir(parents=True, exist_ok=True)
    with open("logs/chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)
    logger.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks


if __name__ == "__main__":
    from src.ingestion.llama_cloud_parser import LlamaCloudParser
    from src.ingestion.metadata_enrichment import MetadataEnrichment
    from src.ingestion.parent_child import SectionPipeline

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