import json
from pathlib import Path

from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.indexing.vector_store import VectorStore


def _sort_key(section_no: str) -> list:
    return [
        int(x) if x.isdigit() else x
        for x in section_no.split("::")[-1].split(".")
    ]


class TreeBuilder:
    def __init__(self, vector_store: VectorStore | None = None):
        self._vector_store = vector_store or VectorStore()

    async def search_parents(self, chunks: list[dict]) -> list[dict]:
        unique_parent_sections = list(dict.fromkeys(
            c.get("metadata", {}).get("parent_section_no")
            for c in chunks
            if c.get("metadata", {}).get("parent_section_no") is not None
        ))
        if not unique_parent_sections:
            logger.warning("No parent sections found in provided chunks")
            return []

        parent_chunks: list[dict] = []
        for section_no in unique_parent_sections:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.parent_section_no",
                        match=MatchValue(value=section_no),
                    )
                ]
            )
            results = await self._vector_store.retrieve(filters=query_filter)
            for hit in results:
                parent_chunks.append({
                    "content": hit["payload"]["content"],
                    "metadata": hit["payload"].get("metadata", {}),
                })

        logger.info("Found {}/{} parent sections", len(parent_chunks), len(unique_parent_sections))
        return parent_chunks

    async def build_tree(self, chunks: list[dict]) -> list[dict]:
        sibling_chunks = await self.search_parents(chunks)

        title_map: dict[str, list[dict]] = {}

        for chunk in sibling_chunks:
            title = chunk["metadata"].get("title", "")
            title_map.setdefault(title, []).append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
            })

        for chunk in chunks:
            title = chunk.get("metadata", {}).get("title", "")
            title_map.setdefault(title, []).append({
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
                "score": chunk.get("score", 0),
            })

        tree = []
        for title, chunks_list in sorted(
            title_map.items(),
            key=lambda item: _sort_key(item[1][0]["metadata"].get("section_no", "")),
        ):
            chunks_list.sort(key=lambda c: _sort_key(c["metadata"].get("section_no", "")))
            tree.append({"title": title, "children": chunks_list})

        orphan_chunks = [
            c for c in chunks
            if not c.get("metadata", {}).get("parent_section_no")
        ]
        if orphan_chunks:
            tree.append({
                "title": "orphan",
                "children": [
                    {"content": c["content"], "metadata": c.get("metadata", {}), "score": c.get("score", 0)}
                    for c in orphan_chunks
                ],
            })

        total = sum(len(n["children"]) for n in tree)
        logger.info("Built tree: {} titles, {} total chunks", len(tree), total)
        self._save_tree(tree)
        return tree

    @staticmethod
    def flatten_tree(tree: list[dict]) -> list[dict]:
        flat: list[dict] = []
        for node in tree:
            flat.extend(node["children"])
        return flat

    @staticmethod
    def _save_tree(tree: list[dict]):
        Path("logs").mkdir(parents=True, exist_ok=True)
        with open("logs/build_tree.json", "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        logger.info("Tree saved to logs/build_tree.json")
