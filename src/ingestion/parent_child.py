import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        current: _MarkdownBlock | None = None

        for line in lines:
            m = self._HEADING_RE.match(line)
            if m:
                if current is not None:
                    current.content = current.content.strip()
                    blocks.append(current)
                current = _MarkdownBlock(
                    level=len(m.group(1)),
                    heading=m.group(2).strip(),
                    content="",
                )
            else:
                if current is not None:
                    current.content += line + "\n"

        if current is not None:
            current.content = current.content.strip()
            blocks.append(current)

        return blocks

    def _consolidate_empty_h3(self, blocks: list[_MarkdownBlock]) -> list[_MarkdownBlock]:
        result: list[_MarkdownBlock] = []
        i = 0
        n = len(blocks)

        while i < n:
            block = blocks[i]
            if block.level == 3 and not block.content:
                merged_heading = block.heading
                j = i + 1
                while j < n:
                    nxt = blocks[j]
                    if nxt.level == 3:
                        merged_heading = f"{merged_heading} -> {nxt.heading}"
                        if nxt.content:
                            result.append(
                                _MarkdownBlock(
                                    level=3,
                                    heading=merged_heading,
                                    content=nxt.content,
                                )
                            )
                            i = j + 1
                            break
                        j += 1
                    else:
                        result.append(
                            _MarkdownBlock(level=3, heading=merged_heading, content="")
                        )
                        i = j
                        break
                else:
                    result.append(
                        _MarkdownBlock(level=3, heading=merged_heading, content="")
                    )
                    i = j
            else:
                result.append(block)
                i += 1

        return result

    def _assemble_tree(self, blocks: list[_MarkdownBlock]) -> list[dict]:
        tree: list[dict] = []
        current_l1: dict | None = None
        current_l2: dict | None = None
        counters = [0, 0, 0]

        for block in blocks:
            title = self._TRAILING_JUNK.sub("", block.heading).strip()
            raw = block.content
            has_content = bool(raw)
            has_table = bool(re.search(r"<table", raw, re.IGNORECASE)) if raw else False
            text = f"{title}: {raw}" if raw else ""

            if block.level == 1:
                counters[0] += 1
                counters[1] = 0
                counters[2] = 0
                section_no = f"{counters[0]}"
            elif block.level == 2:
                counters[1] += 1
                counters[2] = 0
                section_no = f"{counters[0]}.{counters[1]}"
            else:
                counters[2] += 1
                section_no = f"{counters[0]}.{counters[1]}.{counters[2]}"

            node: dict[str, Any] = {
                "section_no": section_no,
                "level": block.level,
                "title": title,
                "has_content": has_content,
                "has_table": has_table,
                "children": [],
            }
            if text:
                node["text"] = text

            if block.level == 1:
                tree.append(node)
                current_l1 = node
                current_l2 = None
            elif block.level == 2:
                if current_l1 is None:
                    counters[0] += 1
                    current_l1 = {
                        "section_no": f"{counters[0]}",
                        "level": 1,
                        "title": "",
                        "has_content": False,
                        "has_table": False,
                        "children": [],
                    }
                    tree.append(current_l1)
                current_l1["children"].append(node)
                current_l2 = node
            else:
                if current_l2 is None:
                    if current_l1 is None:
                        counters[0] += 1
                        current_l1 = {
                            "section_no": f"{counters[0]}",
                            "level": 1,
                            "title": "",
                            "has_content": False,
                            "has_table": False,
                            "children": [],
                        }
                        tree.append(current_l1)
                    counters[1] += 1
                    current_l2 = {
                        "section_no": f"{counters[0]}.{counters[1]}",
                        "level": 2,
                        "title": "",
                        "has_content": False,
                        "has_table": False,
                        "children": [],
                    }
                    current_l1["children"].append(current_l2)
                current_l2["children"].append(node)

        return tree


class SectionBuilder:
    def __init__(self, file_id: str, filing_metadata: dict):
        self._file_id = file_id
        self._filing_metadata = filing_metadata

    def generate(self, tree: list[dict]) -> list[dict]:
        records: list[dict] = []
        self._traverse(tree, parent_title=None, accumulator=records)
        return records

    def _traverse(self, nodes: list[dict], parent_title: str | None, accumulator: list[dict]) -> None:
        for node in nodes:
            metadata = {
                "file_id": self._file_id,
                "company_name": self._filing_metadata.get("company_name", ""),
                "ticker": self._filing_metadata.get("ticker", ""),
                "fiscal_year": self._filing_metadata.get("fiscal_year", ""),
                "filing_type": self._filing_metadata.get("filing_type", ""),
                "period_ended": self._filing_metadata.get("period_ended", ""),
                "filing_date": self._filing_metadata.get("filing_date", ""),
                "section_no": node.get("section_no", ""),
                "level": node["level"],
                "title": node["title"],
                "parent": parent_title,
                "has_content": node.get("has_content", False),
                "has_table": node.get("has_table", False),
            }

            accumulator.append({
                "content": node.get("text", ""),
                "metadata": metadata,
            })

            children = node.get("children", [])
            if children:
                self._traverse(children, parent_title=node["title"], accumulator=accumulator)


class SectionPipeline:
    def __init__(self, content_dir: str | Path, output_dir: str | Path):
        self._content_dir = Path(content_dir)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, meta_source: str | Path | dict[str, dict]) -> None:
        start_total = time.perf_counter()

        meta_dict = self._load_metadata(meta_source)
        md_files = list(self._content_dir.glob("*.md"))
        logger.info("Found {} markdown files in {}", len(md_files), self._content_dir)

        for md_file in md_files:
            self._process_file(md_file, meta_dict)

        elapsed = time.perf_counter() - start_total
        logger.info("Pipeline complete: {} files in {:.2f}s", len(md_files), elapsed)

    def _process_file(self, md_file: Path, meta_dict: dict[str, dict]) -> None:
        stem = md_file.stem
        logger.info("Processing {}", stem)

        meta = self._resolve_metadata(stem, meta_dict)
        if meta is None:
            logger.error("No metadata found for {}", stem)
            return

        start = time.perf_counter()

        md_text = md_file.read_text(encoding="utf-8")
        read_time = time.perf_counter() - start

        tree_start = time.perf_counter()
        tree = HierarchyBuilder().construct(md_text)
        tree_time = time.perf_counter() - tree_start

        file_id = self._compute_file_id(md_file)

        gen_start = time.perf_counter()
        sections = SectionBuilder(file_id, meta).generate(tree)
        gen_time = time.perf_counter() - gen_start

        out_path = self._output_dir / f"{stem}.json"
        out_path.write_text(
            json.dumps(sections, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        total = time.perf_counter() - start
        logger.debug(
            "Built {}: read={:.3f}s, tree={:.3f}s, gen={:.3f}s, total={:.3f}s",
            stem, read_time, tree_time, gen_time, total
        )
        logger.info("Wrote {} sections to {}", len(sections), out_path)

    def _load_metadata(self, source: str | Path | dict[str, dict]) -> dict[str, dict]:
        if isinstance(source, dict):
            return source

        path = Path(source)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))

        raise ValueError(f"Unsupported metadata source: {source}")

    @staticmethod
    def _compute_file_id(filepath: str | Path) -> str:
        return hashlib.md5(Path(filepath).read_bytes()).hexdigest()[:12]

    @staticmethod
    def _resolve_metadata(stem: str, meta_dict: dict[str, dict]) -> dict | None:
        return meta_dict.get(stem)


if __name__ == "__main__":
    pipeline = SectionPipeline("outputs/content", "outputs/sections")
    pipeline.execute("outputs/metadata.json")