"""IPC/CrPC/Evidence Act <-> BNS/BNSS/BSA section mapping and the date rule for criminal law.

The mapping data (data/section_map.json) is partial and UNVERIFIED; see the file's `meta.warning`.
Never treat a mapper result as authoritative without checking the official text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.legal_india.statutes import StatuteRef

CUTOFF = date(2024, 7, 1)  # BNS, BNSS and BSA came into force on this date
DEFAULT_MAP_PATH = Path(__file__).parent / "data" / "section_map.json"
_BASE = re.compile(r"^(\d+[A-Za-z]{0,2})")


def base_section(provision: str) -> str:
    """"103(1)" -> "103", "498a" -> "498A"."""
    m = _BASE.match(provision.strip())
    if not m:
        raise ValueError(f"not a section number: {provision!r}")
    return m.group(1).upper()


@dataclass(frozen=True)
class MappingEntry:
    old_code: str
    old: str
    new_code: str
    new: str
    title: str
    verified: bool
    source_note: str


class SectionMap:
    def __init__(self, entries: list[MappingEntry]) -> None:
        self.entries = entries
        self._by_old: dict[tuple[str, str], list[MappingEntry]] = {}
        self._by_new: dict[tuple[str, str], list[MappingEntry]] = {}
        for e in entries:
            self._by_old.setdefault((e.old_code, base_section(e.old)), []).append(e)
            self._by_new.setdefault((e.new_code, base_section(e.new)), []).append(e)

    def to_new(self, old_code: str, section: str) -> list[MappingEntry]:
        """Entries for an old-code section (e.g. ``to_new("IPC", "302")``); empty if not in the table."""
        return list(self._by_old.get((old_code.upper(), base_section(section)), []))

    def to_old(self, new_code: str, section: str) -> list[MappingEntry]:
        return list(self._by_new.get((new_code.upper(), base_section(section)), []))


def load_section_map(path: Path | None = None) -> SectionMap:
    data = json.loads((path or DEFAULT_MAP_PATH).read_text(encoding="utf-8"))
    entries = [
        MappingEntry(
            old_code=m["old_code"],
            old=m["old"],
            new_code=m["new_code"],
            new=m["new"],
            title=m.get("title", ""),
            verified=bool(m.get("verified", False)),
            source_note=m.get("source_note", ""),
        )
        for m in data["mappings"]
    ]
    return SectionMap(entries)


def applicable_code(offence_date: date, family: str = "substantive") -> str | None:
    """Substantive criminal code governing an offence: IPC if committed on or before 30 June 2024, else BNS.

    Only the "substantive" family is decided by the offence date. Procedure (CrPC/BNSS) and evidence
    (Evidence Act/BSA) depend on the stage of proceedings and the savings clauses, so None is returned.
    An offence spanning the cut-off date is not modelled.
    """
    if family != "substantive":
        return None
    return "IPC" if offence_date < CUTOFF else "BNS"


@dataclass(frozen=True)
class TransitionFinding:
    status: str  # ok | wrong_code | undetermined | not_assessed
    cited_code: str | None
    expected_code: str | None
    suggestion: str | None  # equivalent section in the expected code, if our table has one
    message: str


_SUBSTANTIVE = {"IPC", "BNS"}


def check_cited_section(ref: StatuteRef, offence_date: date | None, section_map: SectionMap | None = None) -> TransitionFinding:
    """Flag a cited IPC/BNS section whose code is wrong for the date of the offence."""
    if ref.act not in _SUBSTANTIVE:
        return TransitionFinding("not_assessed", ref.act, None, None, "Only IPC/BNS citations are assessed by offence date; procedure and evidence depend on the stage of proceedings.")
    if offence_date is None:
        return TransitionFinding("undetermined", ref.act, None, None, "Offence date unknown, so the applicable code cannot be determined.")
    expected = applicable_code(offence_date)
    if ref.act == expected:
        return TransitionFinding("ok", ref.act, expected, None, f"{ref.act} is the applicable code for an offence on {offence_date.isoformat()}.")
    smap = section_map or load_section_map()
    entries = smap.to_new("IPC", ref.base) if expected == "BNS" else smap.to_old("BNS", ref.base)
    suggestion = None
    if entries:
        e = entries[0]
        suggestion = f"BNS {e.new}" if expected == "BNS" else f"IPC {e.old}"
    message = f"{ref.act} {ref.provision} is cited, but an offence on {offence_date.isoformat()} is governed by the {expected}."
    if suggestion:
        message += f" Our (unverified) table suggests {suggestion}."
    return TransitionFinding("wrong_code", ref.act, expected, suggestion, message)
