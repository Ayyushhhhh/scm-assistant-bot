"""Section-aware PDF parser for the governance policy document.

Design decisions
────────────────
• Uses PyMuPDF (fitz) — fastest pure-Python PDF extractor.
• Splits on section headers (e.g. "5. Risk Assessment") using regex,
  preserving natural policy-rule boundaries as atomic chunks.
• Each section is further split into sub-sections (e.g. "5.1 Risk Level
  Definitions") so that each chunk = one complete rule / procedure.
• Metadata (section_id, title, page) attached to every chunk for
  traceability in retrieval and citation.
• The regex is strict: top-level sections MUST start with a single digit
  followed by a period and title (e.g. "3. Service Level Agreements").
  Sub-sections use "X.Y" format.  This prevents table-data numbers
  (90, 75, 60) from being misidentified as section headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz  # PyMuPDF

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class PolicySection:
    """A single section or sub-section of the governance policy."""

    section_id: str         # e.g. "5.1"
    title: str              # e.g. "Risk Level Definitions"
    content: str            # full raw text of this (sub-)section
    page: int = 0           # first page where this section appears
    parent_id: str = ""     # e.g. "5" for sub-section "5.1"

    @property
    def full_title(self) -> str:
        return f"§{self.section_id} {self.title}"


# ── Regex for section headers ────────────────────────────────────────────────
# Top-level: "3. Service Level Agreements" — single digit, period, space, title
# Sub-section: "3.1 On-Time Delivery (OTD)" — digit.digit, space, title
# Title must start with a capital letter (filters out "90 or above", "60 – 74")
_SECTION_RE = re.compile(
    r"^(\d{1,2}\.\d{1,2})\s+([A-Z][A-Za-z].+?)$",
    re.MULTILINE,
)
_TOP_SECTION_RE = re.compile(
    r"^(\d{1,2})\.\s+([A-Z][A-Za-z &\-].{5,})$",
    re.MULTILINE,
)


def parse_pdf(pdf_path: str | object) -> list[PolicySection]:
    """Parse the governance PDF into a flat list of PolicySections.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Ordered list of PolicySection objects (sections + sub-sections).
    """
    doc = fitz.open(str(pdf_path))
    full_text = ""
    page_map: list[tuple[int, int]] = []  # (char_offset, page_number)

    for page in doc:
        offset = len(full_text)
        page_text = page.get_text()
        full_text += page_text
        page_map.append((offset, page.number + 1))

    doc.close()

    log.info("PDF text extracted", extra={"chars": len(full_text), "pages": len(page_map)})

    # Find all headers (both top-level and sub-sections)
    headers: list[tuple[int, str, str, bool]] = []  # (pos, id, title, is_top)

    for match in _TOP_SECTION_RE.finditer(full_text):
        sec_id = match.group(1)
        title = match.group(2).strip()
        if int(sec_id) <= 15:  # policy has sections 1-10
            headers.append((match.start(), sec_id, title, True))

    for match in _SECTION_RE.finditer(full_text):
        sec_id = match.group(1)
        title = match.group(2).strip()
        parent = sec_id.split(".")[0]
        if int(parent) <= 15 and int(sec_id.split(".")[1]) <= 10:
            headers.append((match.start(), sec_id, title, False))

    # Sort by position in document
    headers.sort(key=lambda h: h[0])

    # Deduplicate (same position)
    seen_pos = set()
    unique_headers = []
    for h in headers:
        if h[0] not in seen_pos:
            seen_pos.add(h[0])
            unique_headers.append(h)
    headers = unique_headers

    if not headers:
        log.warning("No section headers found, returning entire document as one chunk")
        return [PolicySection(section_id="0", title="Full Document", content=full_text)]

    log.info(f"Found {len(headers)} section headers")

    # Extract content between consecutive headers
    sections: list[PolicySection] = []
    for i, (pos, sec_id, title, is_top) in enumerate(headers):
        # Content runs from end of this header line to start of next header
        line_end = full_text.find("\n", pos)
        content_start = line_end + 1 if line_end != -1 else pos + len(title)
        content_end = headers[i + 1][0] if i + 1 < len(headers) else len(full_text)
        content = full_text[content_start:content_end].strip()

        # Determine page number from offset
        page_num = 1
        for offset, pnum in page_map:
            if pos >= offset:
                page_num = pnum

        # Determine parent (e.g. "5.1" → parent "5")
        parent = sec_id.split(".")[0] if "." in sec_id else ""

        sections.append(PolicySection(
            section_id=sec_id,
            title=title,
            content=content,
            page=page_num,
            parent_id=parent,
        ))

    log.info("PDF parsed into sections", extra={"section_count": len(sections)})
    for sec in sections:
        log.info(f"  §{sec.section_id} {sec.title} ({len(sec.content)} chars)")

    return sections
