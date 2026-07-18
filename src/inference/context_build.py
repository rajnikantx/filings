from loguru import logger


class Context:
    def build_context(self, chunks: list[dict]) -> str:
        if not chunks:
            logger.warning("No chunks provided to build context")
            return ""

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source_file", "unknown")
            score = chunk.get("score", 0)
            parts.append(
                f"[Source: {source} | Relevance: {score:.4f}]\n{chunk['content']}"
            )

        context = "\n\n---\n\n".join(parts)
        logger.info("Built context from {} chunks ({} chars)", len(chunks), len(context))
        return context