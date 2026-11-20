"""Splits the text of an Indian Act (or the Constitution) into sections and section-level chunks.

Bare-act PDFs have a table "ARRANGEMENT OF SECTIONS" followed by the body, where each section starts on a
new line as ``7. Disposal of request.—(1) Subject to ...``. The parser:

* recognises a section start as ``<number>. <title>.—`` at the beginning of a line (the em dash separates the
  heading from the text, which the arrangement table lacks, so the table is not mistaken for the body);
* keeps only matches that continue the section numbering (1, 2, 2A, 3, ...), which rejects numbered list items
  and cross references that happen to start a line;
* tracks the chapter/part heading in force for each section.

It is heuristic: PDF text extraction can reorder or break lines, so ``ParseReport`` states what was found and
what looks suspicious instead of pretending the parse is exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.documents import Chunk

_SECTION = re.compile(
    r"(?:^|(?<=\. )|(?<=\] )|(?<=\) ))(?P<num>\d{1,3}(?:[A-Z]{1,3})?)\.[ \t]+"
    r"(?P<title>[^\n—–]{2,200}(?:\n(?!\d{1,3}[A-Z]{0,3}\.[ \t])[^\n—–]{1,120})?)\s*\.?\s*(?:[—–]|--)\s*(?P<rest>[^\n]*)",
    re.MULTILINE,
)
_CHAPTER = re.compile(r"^(?P<kind>CHAPTER|PART)\s+(?P<num>[IVXLC]+[A-Z]?)\b[ \t.\-—–]*(?P<title>[^\n]*)$", re.MULTILINE)
_NUM = re.compile(r"^(\d+)([A-Z]*)$")
_SENTENCE_BREAK = re.compile(r"(?<=[.;:])\s+(?=\(?[0-9a-zA-Z]{1,3}\)|[A-Z])")


@dataclass(frozen=True)
class Section:
    number: str
    title: str
    chapter: str | None
    text: str


@dataclass
class ParseReport:
    sections: int = 0
    first: str | None = None
    last: str | None = None
    gaps: list[str] = field(default_factory=list)  # numbers missing from the sequence
    short_sections: int = 0  # sections with almost no text (possible mis-split)
    warnings: list[str] = field(default_factory=list)


def _key(number: str) -> tuple[int, str]:
    m = _NUM.match(number)
    return (int(m.group(1)), m.group(2)) if m else (0, number)


def clean_pdf_text(text: str) -> str:
    """Drops lines made mostly of non-ASCII characters (legacy-font Hindi in bilingual PDFs) and page furniture."""
    lines = []
    for line in text.replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        non_ascii = sum(1 for ch in stripped if ord(ch) > 127)
        if non_ascii / len(stripped) > 0.25:
            continue
        if re.fullmatch(r"(?i)(page\s*)?\d{1,4}", stripped):
            continue  # bare page numbers
        lines.append(stripped)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def parse_sections(text: str) -> tuple[list[Section], ParseReport]:
    chapters = [(m.start(), _chapter_label(m, text)) for m in _CHAPTER.finditer(text)]
    accepted = _longest_increasing(list(_SECTION.finditer(text)))
    sections: list[Section] = []
    for i, m in enumerate(accepted):
        end = accepted[i + 1].start() if i + 1 < len(accepted) else len(text)
        body = re.sub(r"\s+", " ", text[m.start():end]).strip()
        chapter = next((label for pos, label in reversed(chapters) if pos < m.start()), None)
        title = re.sub(r"\s+", " ", m.group("title")).strip(" .")
        sections.append(Section(number=m.group("num"), title=title, chapter=chapter, text=body))

    report = ParseReport(sections=len(sections))
    if sections:
        report.first, report.last = sections[0].number, sections[-1].number
        present = {_key(s.number)[0] for s in sections}
        report.gaps = [str(n) for n in range(1, _key(sections[-1].number)[0] + 1) if n not in present][:50]
        report.short_sections = sum(1 for s in sections if len(s.text) < 60)
        if report.gaps:
            report.warnings.append(f"{len(report.gaps)} section numbers missing from the sequence (repealed or mis-parsed)")
    else:
        report.warnings.append("no sections recognised; the PDF layout may not match the expected format")
    return sections, report


def _longest_increasing(matches: list[re.Match[str]]) -> list[re.Match[str]]:
    """Longest run of candidates whose section numbers strictly increase (1, 2, 2A, 3, ...).

    Numbered list items and cross references that start a line produce out-of-order candidates; picking the
    longest increasing run discards them without requiring every section to be found.
    """
    import bisect

    keys = [_key(m.group("num")) for m in matches]
    tails: list[tuple[int, str]] = []  # smallest tail key of an increasing run of each length
    tail_index: list[int] = []
    parent = [-1] * len(matches)
    for i, key in enumerate(keys):
        pos = bisect.bisect_left(tails, key)
        if pos == len(tails):
            tails.append(key)
            tail_index.append(i)
        else:
            tails[pos] = key
            tail_index[pos] = i
        parent[i] = tail_index[pos - 1] if pos > 0 else -1
    chain, i = [], tail_index[-1] if tail_index else -1
    while i != -1:
        chain.append(matches[i])
        i = parent[i]
    return chain[::-1]


def _chapter_label(m: re.Match[str], text: str) -> str:
    title = m.group("title").strip()
    if not title:  # the title is often on the next line
        nxt = text[m.end():m.end() + 160].strip().split("\n", 1)[0].strip()
        title = nxt if nxt.isupper() or len(nxt) < 80 else ""
    kind = m.group("kind").title()
    return f"{kind} {m.group('num')}" + (f" - {title}" if title else "")


def chunk_sections(
    sections: list[Section], *, act: str, doc_id: str, corpus: str, filename: str, max_chars: int = 800
) -> list[Chunk]:
    """One chunk per section; sections longer than ``max_chars`` are split at clause boundaries.

    Every chunk starts with a header ("<Act>, Section N. <Title>") so retrieval and the claim checker always
    see which provision the text belongs to.
    """
    chunks: list[Chunk] = []
    for section in sections:
        header = f"{act}, {'Article' if 'constitution' in act.lower() else 'Section'} {section.number}. {section.title}"
        for part in _split(section.text, max_chars - len(header) - 2):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{index:05d}",
                    doc_id=doc_id,
                    corpus=corpus,
                    filename=filename,
                    chunk_index=index,
                    text=f"{header}\n{part}",
                    act=act,
                    section=section.number,
                    section_title=section.title,
                    chapter=section.chapter,
                )
            )
    return chunks


def _split(text: str, limit: int) -> list[str]:
    limit = max(limit, 200)
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in _SENTENCE_BREAK.split(text):
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current.strip())
            current = ""
        while len(sentence) > limit:  # a single very long clause: hard split on whitespace
            cut = sentence.rfind(" ", 0, limit) or limit
            pieces.append((current + " " + sentence[:cut]).strip())
            current, sentence = "", sentence[cut:].strip()
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces
