import re
import json
import asyncio
import logging
import argparse
from typing import List
from pathlib import Path
from src.config import settings

logger = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"(<table>.*?</table>)", re.DOTALL)
_NEXT_SEP = {"\n\n": "\n", "\n": "sentence", "sentence": None}


class Chunker:
    """Read JSON files, split their content into chunks, save and return them."""

    async def chunk(self) -> List[dict]:
        """Process all *.json files, save chunks to disk, and return flattened chunks."""
        parent_child_dir = Path("outputs/sections")
        if not parent_child_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {settings.PARENT_CHILD_PATH}")

        # Ensure save directory exists
        save_dir = Path(settings.CHUNK_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        files = list(parent_child_dir.glob("*.json"))
        if not files:
            logger.warning("No JSON files found in %s", parent_child_dir)
            return []

        # Process all files concurrently
        results = await asyncio.gather(
            *[asyncio.to_thread(self._process_file, file, save_dir) for file in files]
        )

        # Flatten list of lists into one list
        return [chunk for file_chunks in results for chunk in file_chunks]

    def _process_file(self, file: Path, save_dir: Path) -> List[dict]:
        """Load a JSON file, chunk it, save chunks, and return them."""
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize so we always iterate over a list of items
        if isinstance(data, dict):
            items: List[dict] = [data]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError(
                f"Expected JSON list or dict in {file.name}, got {type(data).__name__}"
            )

        chunks = []
        for item_idx, item in enumerate(items):
            content = item.get("content", "")
            metadata = item.get("metadata", {})

            if not content.strip():
                continue

            text_chunks = Chunker.chunk_text(content, max_chars=1000)

            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk = {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "source_file": file.name,
                    }
                }
                chunks.append(chunk)

        # Save all chunks for this file to disk
        save_path = save_dir / f"{file.stem}_chunks.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info("Saved %d chunks from %s to %s", len(chunks), file.name, save_path)
        return chunks

    @staticmethod
    def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
        """Split text into chunks. Tables are always kept whole."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        for part in _TABLE_RE.split(text):
            if not part:
                continue
            if part.startswith("<table>") and part.endswith("</table>"):
                chunks.append(part)
            else:
                chunks.extend(Chunker._chunk_recursive(part, max_chars, "\n\n"))
        return chunks

    @staticmethod
    def _chunk_recursive(text: str, max_chars: int, sep: str) -> List[str]:
        """Split text by sep. If a piece is still too big, recurse with a finer sep."""
        if len(text) <= max_chars:
            return [text]

        next_sep = _NEXT_SEP.get(sep)
        chunks = []

        for part in text.split(sep):
            if not part:
                continue
            if len(part) <= max_chars:
                chunks.append(part)
            elif next_sep is None:
                for i in range(0, len(part), max_chars):
                    chunks.append(part[i:i + max_chars])
            elif next_sep == "sentence":
                sentences = re.split(r"(?<=[.!?])\s+", part)
                chunks.extend(Chunker._chunk_sentences(sentences, max_chars))
            else:
                chunks.extend(Chunker._chunk_recursive(part, max_chars, next_sep))

        return chunks

    @staticmethod
    def _chunk_sentences(sentences: List[str], max_chars: int) -> List[str]:
        """Merge short sentences together. Hard-split any sentence that is too long."""
        chunks = []
        current = ""

        for sent in sentences:
            candidate = f"{current} {sent}".strip() if current else sent
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(sent) <= max_chars:
                    current = sent
                else:
                    for i in range(0, len(sent), max_chars):
                        chunks.append(sent[i:i + max_chars])
                    current = ""

        if current:
            chunks.append(current)
        return chunks

async def run_chunker() -> List[dict]:
    """Instantiate and run the chunker."""
    chunker = Chunker()
    all_chunks = await chunker.chunk()
    logger.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks


if __name__ == "__main__":
    all_chunks = asyncio.run(run_chunker())
    print(f"\nDone. Total chunks: {len(all_chunks)}")