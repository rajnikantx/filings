import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from loguru import logger


@dataclass
class _MarkdownBlock:
    level: int
    heading: str
    content: str = ""


class HierarchyBuilder:
    _HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)")
    _TRAILING_JUNK = re.compile(r"[\s:.\-–—*_|/\\]+$")

    def construct(self, md_text: str) -> list[dict]:
        blocks = self._extract_blocks(md_text)
        blocks = self._consolidate_empty_h3(blocks)
        return self._assemble_tree(blocks)

    def _extract_blocks(self, text: str) -> list[_MarkdownBlock]:
        lines = text.splitlines()
        blocks: list[_MarkdownBlock] = []
        current_block: _MarkdownBlock | None = None

        for line in lines:
            heading_match = self._HEADING_RE.match(line)
            if heading_match:
                if current_block is not None:
                    current_block.content = current_block.content.strip()
                    blocks.append(current_block)
                current_block = _MarkdownBlock(
                    level=len(heading_match.group(1)),
                    heading=heading_match.group(2).strip(),
                    content="",
                )
            else:
                if current_block is not None:
                    current_block.content += line + "\n"

        if current_block is not None:
            current_block.content = current_block.content.strip()
            blocks.append(current_block)

        if blocks and blocks[0].heading == "Preamble":
            return blocks

        leading_lines: list[str] = []
        idx = 0
        while idx < len(lines) and not self._HEADING_RE.match(lines[idx]):
            leading_lines.append(lines[idx])
            idx += 1

        if leading_lines:
            preamble = _MarkdownBlock(level=1, heading="Preamble", content="\n".join(leading_lines).strip())
            blocks.insert(0, preamble)

        return blocks

    def _consolidate_empty_h3(self, blocks: list[_MarkdownBlock]) -> list[_MarkdownBlock]:
        consolidated_blocks: list[_MarkdownBlock] = []
        index = 0
        total_blocks = len(blocks)

        while index < total_blocks:
            block = blocks[index]
            if block.level == 3 and not block.content:
                merged_heading = block.heading
                lookahead_index = index + 1
                while lookahead_index < total_blocks:
                    next_block = blocks[lookahead_index]
                    if next_block.level == 3:
                        merged_heading = f"{merged_heading} -> {next_block.heading}"
                        if next_block.content:
                            consolidated_blocks.append(
                                _MarkdownBlock(
                                    level=3,
                                    heading=merged_heading,
                                    content=next_block.content,
                                )
                            )
                            index = lookahead_index + 1
                            break
                        lookahead_index += 1
                    else:
                        consolidated_blocks.append(
                            _MarkdownBlock(level=3, heading=merged_heading, content="")
                        )
                        index = lookahead_index
                        break
                else:
                    consolidated_blocks.append(
                        _MarkdownBlock(level=3, heading=merged_heading, content="")
                    )
                    index = lookahead_index
            else:
                consolidated_blocks.append(block)
                index += 1

        return consolidated_blocks

    def _assemble_tree(self, blocks: list[_MarkdownBlock]) -> list[dict]:
        section_tree: list[dict] = []
        current_section: dict | None = None
        current_subsection: dict | None = None
        section_counters = [0, 0, 0]

        for block in blocks:
            title = self._TRAILING_JUNK.sub("", block.heading).strip()
            raw_content = block.content
            has_content = bool(raw_content)
            has_table = bool(re.search(r"<table", raw_content, re.IGNORECASE)) if raw_content else False
            formatted_text = f"{title}: {raw_content}" if raw_content else ""

            if block.level == 1:
                section_counters[0] += 1
                section_counters[1] = 0
                section_counters[2] = 0
                section_no = f"{section_counters[0]}"
            elif block.level == 2:
                section_counters[1] += 1
                section_counters[2] = 0
                section_no = f"{section_counters[0]}.{section_counters[1]}"
            else:
                section_counters[2] += 1
                section_no = f"{section_counters[0]}.{section_counters[1]}.{section_counters[2]}"

            node: dict[str, Any] = {
                "section_no": section_no,
                "level": block.level,
                "title": title,
                "has_content": has_content,
                "has_table": has_table,
                "children": [],
            }
            if formatted_text:
                node["text"] = formatted_text

            if block.level == 1:
                section_tree.append(node)
                current_section = node
                current_subsection = None
            elif block.level == 2:
                if current_section is None:
                    section_counters[0] += 1
                    current_section = {
                        "section_no": f"{section_counters[0]}",
                        "level": 1,
                        "title": "",
                        "has_content": False,
                        "has_table": False,
                        "children": [],
                    }
                    section_tree.append(current_section)
                current_section["children"].append(node)
                current_subsection = node
            else:
                if current_subsection is None:
                    if current_section is None:
                        section_counters[0] += 1
                        current_section = {
                            "section_no": f"{section_counters[0]}",
                            "level": 1,
                            "title": "",
                            "has_content": False,
                            "has_table": False,
                            "children": [],
                        }
                        section_tree.append(current_section)
                    section_counters[1] += 1
                    current_subsection = {
                        "section_no": f"{section_counters[0]}.{section_counters[1]}",
                        "level": 2,
                        "title": "",
                        "has_content": False,
                        "has_table": False,
                        "children": [],
                    }
                    current_section["children"].append(current_subsection)
                current_subsection["children"].append(node)

        return section_tree


def flatten_sections(sections: list[dict], parent_title: str | None = None) -> list[dict]:
    flat: list[dict] = []
    for section in sections:
        metadata = {**section["metadata"], "parent": parent_title}
        flat.append({"content": section["content"], "metadata": metadata})
        children = section.get("children", [])
        if children:
            flat.extend(flatten_sections(children, parent_title=section["metadata"]["title"]))
    return flat


class SectionBuilder:
    def __init__(self, file_id: str, file_stem: str, filing_metadata: dict):
        self._file_id = file_id
        self._file_stem = file_stem
        self._filing_metadata = filing_metadata
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def build_hierarchical_tree(self, tree: list[dict]) -> list[dict]:
        return self._build_tree(tree, parent_title=None)

    def _build_tree(self, nodes: list[dict], parent_title: str | None) -> list[dict]:
        result: list[dict] = []
        for node in nodes:
            title = node.get("title", "")
            section_metadata = {
                "file_id": self._file_id,
                "source_file": self._file_stem,
                "company_name": self._filing_metadata.get("company_name", ""),
                "ticker": self._filing_metadata.get("ticker", ""),
                "fiscal_year": self._filing_metadata.get("fiscal_year", ""),
                "filing_type": self._filing_metadata.get("filing_type", ""),
                "period_ended": self._filing_metadata.get("period_ended", ""),
                "filing_date": self._filing_metadata.get("filing_date", ""),
                "section_no": f"{self._file_stem}::{node.get('section_no', '')}",
                "level": node["level"],
                "title": title,
                "parent": parent_title,
                "has_content": node.get("has_content", False),
                "has_table": node.get("has_table", False),
                "token_count": len(self._encoding.encode(node.get("text", ""))),
            }

            record: dict[str, Any] = {
                "content": node.get("text", ""),
                "metadata": section_metadata,
            }

            children = node.get("children", [])
            if children:
                record["children"] = self._build_tree(children, parent_title=title)

            result.append(record)
        return result


class SectionPipeline:
    def execute(
        self,
        parsed_files: Sequence[tuple[Path, str, str]],
        metadata_by_file: dict[str, dict],
    ) -> dict[str, list[dict]]:
        start_total = time.perf_counter()
        logger.info("Building sections for {} file(s)", len(parsed_files))

        hierarchical_by_file: dict[str, list[dict]] = {}
        for file_path, full_markdown, _intro_markdown in parsed_files:
            result = self._process_file(file_path.stem, full_markdown, metadata_by_file)
            if result is not None:
                hierarchical_by_file[file_path.stem] = result

        elapsed = time.perf_counter() - start_total

        Path("logs").mkdir(parents=True, exist_ok=True)
        with open("logs/parent_child.json", "w", encoding="utf-8") as f:
            json.dump(hierarchical_by_file, f, indent=2, ensure_ascii=False)

        logger.info(
            "Pipeline complete: {}/{} files in {:.2f}s",
            len(hierarchical_by_file), len(parsed_files), elapsed,
        )
        return hierarchical_by_file

    def _process_file(
        self, file_stem: str, md_text: str, metadata_by_file: dict[str, dict]
    ) -> list[dict] | None:
        file_metadata = metadata_by_file.get(file_stem)
        if file_metadata is None:
            logger.error("No metadata found for {}", file_stem)
            return None

        start = time.perf_counter()

        section_tree = HierarchyBuilder().construct(md_text)
        file_id = hashlib.md5(md_text.encode("utf-8")).hexdigest()[:12]
        builder = SectionBuilder(file_id, file_stem, file_metadata)
        hierarchical = builder.build_hierarchical_tree(section_tree)

        elapsed = time.perf_counter() - start
        logger.info(
            "Built {} top-level sections for {} ({:.3f}s)",
            len(hierarchical), file_stem, elapsed,
        )
        return hierarchical


if __name__ == "__main__":
    import asyncio

    from src.ingestion.llama_cloud_parser import LlamaCloudParser
    from src.ingestion.metadata_enrichment import MetadataEnrichment

    async def main():
        parser = LlamaCloudParser(tier="agentic")
        parsed_files = await parser.parse_pdfs("data/raw_filings/")

        enricher = MetadataEnrichment()
        metadata_by_file = await enricher.enrich_all(parsed_files)

        pipeline = SectionPipeline()
        sections_by_file = pipeline.execute(parsed_files, metadata_by_file)
        print(json.dumps(sections_by_file, indent=2, ensure_ascii=False))

    asyncio.run(main())