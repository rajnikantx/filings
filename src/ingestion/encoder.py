import asyncio
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from src.config import settings


class Encoder:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self._device = device
        self._model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=device,
        )

    @property
    def dim(self) -> int:
        return settings.EMBEDDING_DIM

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        prefixed = f"{settings.EMBEDDING_QUERY_INSTRUCTION}{query}"
        return self.encode([prefixed])[0]

    async def embed(self, chunk_dir: str | Path) -> list[dict]:
        """Read all chunk JSON files, embed them, and return enriched chunks."""
        path = Path(chunk_dir)

        if not path.is_dir():  # ← fix: is_dir() is a method
            raise NotADirectoryError(f"Directory not found: {chunk_dir}")

        chunk_files = list(path.glob("*.json"))
        if not chunk_files:
            return []

        # ← fix: actually pass tasks to gather
        results = await asyncio.gather(
            *[self._embed_file(f) for f in chunk_files]
        )

        # Flatten list of lists
        return [item for file_result in results for item in file_result]

    async def _embed_file(self, chunk_file_path: str | Path) -> list[dict]:
        """Read a chunk file, embed each chunk's content, attach vectors."""
        path = Path(chunk_file_path)

        # ← fix: "r" for read, not "w" for write
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)  # ← fix: json.load() to parse, not f.write()

        if not isinstance(chunks, list):
            raise ValueError(f"Expected JSON list in {path.name}, got {type(chunks).__name__}")

        # Extract text content from each chunk dict
        texts = [chunk["content"] for chunk in chunks if "content" in chunk]

        if not texts:
            return chunks  # return as-is if no content to embed

        # Encode all texts at once (batching is more efficient)
        embeddings = self.encode(texts)

        # Attach vectors back to chunks
        # Only attach to chunks that had content
        embed_idx = 0
        for chunk in chunks:
            if "content" in chunk and chunk["content"]:
                chunk["vector"] = embeddings[embed_idx]
                embed_idx += 1
            else:
                chunk["vector"] = None

        return chunks