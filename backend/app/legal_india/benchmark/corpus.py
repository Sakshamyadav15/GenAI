"""Loading the judgment corpus and resolving citations against it."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.legal_india.citations import normalise_citation

_WS = re.compile(r"\s+")


def squash(text: str) -> str:
    """Lower-case text with all whitespace removed.

    PDF extraction in the corpus splits words ("sa me", "Refere nce"), so quote matching ignores whitespace.
    """
    return _WS.sub("", text).lower()


def quote_in_text(quote: str, text: str) -> bool:
    q = squash(quote)
    return len(q) >= 25 and q in squash(text)


def parse_date(value: str) -> date:
    day, month, year = value.split("-")
    return date(int(year), int(month), int(day))


@dataclass(frozen=True)
class Paragraph:
    para_no: int
    text: str


@dataclass
class Case:
    case_id: str
    title: str
    date: str
    citations: list[str]
    paragraphs: dict[int, Paragraph]
    petitioner: str = ""
    respondent: str = ""

    @property
    def date_obj(self) -> date:
        return parse_date(self.date)


class Corpus:
    def __init__(self, cases: list[Case]) -> None:
        self.cases = {c.case_id: c for c in cases}
        self._index: dict[str, str] = {}
        for case in cases:
            for key in case.citations:
                self._index[key] = case.case_id

    def resolve(self, citation_text: str) -> Case | None:
        """The corpus case a citation string refers to (by canonical reporter key), if any."""
        for key in normalise_citation(citation_text):
            if key in self._index:
                return self.cases[self._index[key]]
        return None

    def __len__(self) -> int:
        return len(self.cases)


def load_corpus(path: Path) -> Corpus:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys = normalise_citation(row["case_id"]) + normalise_citation(row.get("citation") or "")
        cases.append(
            Case(
                case_id=row["case_id"],
                title=row.get("title", ""),
                date=row["date"],
                citations=keys,
                paragraphs={p["para_no"]: Paragraph(p["para_no"], p["text"]) for p in row["paragraphs"]},
                petitioner=row.get("petitioner", ""),
                respondent=row.get("respondent", ""),
            )
        )
    return Corpus(cases)


def substantive_paragraphs(case: Case, min_chars: int = 350, max_chars: int = 1600) -> list[Paragraph]:
    """Paragraphs long enough to carry a proposition; the header block (para 0) is skipped."""
    return [p for p in case.paragraphs.values() if p.para_no > 0 and min_chars <= len(p.text) <= max_chars]
