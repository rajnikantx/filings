from pathlib import Path

from loguru import logger

from src.config import settings


class ContextBuilder:
    def __init__(self, max_chars: int = 120_000):
        self._max_chars = max_chars

    def build_context(self, chunks: list[dict]) -> str:
        parts: list[str] = []
        total_chars = 0

        for chunk in chunks:
            title = chunk.get("title", "Untitled")
            header = f"\n{'='*60}\nTitle: [{title}]\n{'='*60}"
            if total_chars + len(header) > self._max_chars:
                logger.warning("Context truncated at title level ({} chars)", total_chars)
                break
            parts.append(header)
            total_chars += len(header)

            for child in chunk.get("children", []):
                if child.get("metadata", {}).get("has_table", False):
                    continue
                content = child.get("content", "")
                if not content:
                    continue

                section_no = child.get("metadata", {}).get("section_no", "")
                score = child.get("score")
                score_tag = f" (relevance: {score:.2f})" if score else ""
                section_tag = f"[Section {section_no}]{score_tag} " if section_no else ""
                snippet = f"\n{section_tag}{content}"

                if total_chars + len(snippet) > self._max_chars:
                    logger.warning("Context truncated at chunk level ({} chars)", total_chars)
                    break
                parts.append(snippet)
                total_chars += len(snippet)

        context = "\n".join(parts)
        logger.info("Built context: {} chars (limit: {})", len(context), self._max_chars)
        return context

    def collect_csv_paths(self, chunks: list[dict]) -> list[Path]:
        csv_dir = Path(settings.EXTRACTED_CSV_PATH)
        csv_paths: list[Path] = []
        seen_ids: set[str] = set()

        for chunk in chunks:
            for child in chunk.get("children", []):
                table_id = child.get("metadata", {}).get("table_id")
                if not table_id or table_id in seen_ids:
                    continue
                seen_ids.add(table_id)
                csv_path = csv_dir / f"{table_id}.csv"
                if csv_path.exists():
                    csv_paths.append(csv_path)
                else:
                    logger.warning("CSV not found for table {}: {}", table_id, csv_path)

        logger.info("Found {} CSV files", len(csv_paths))
        return csv_paths
