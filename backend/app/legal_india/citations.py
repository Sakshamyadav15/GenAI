"""Parser and normaliser for Indian case citations found in running text.

Supported reporters: AIR, SCC (incl. ``(Cri)``/``(Civ)``/``(L&S)`` parts and ``Supp (n)``), SCC OnLine,
SCR, SCALE, Supreme Court neutral citations (INSC), and party-name short forms ("A v. B").
Also extracts pinpoints ("para 45", "paras 12-14", "at p. 300") and groups parallel citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

YEAR = r"(?:18|19|20)\d{2}"


@dataclass(frozen=True)
class Pinpoint:
    kind: str  # "para" or "page"
    numbers: tuple[int, ...]  # ranges are expanded, e.g. "paras 12-14" -> (12, 13, 14)
    raw: str
    start: int
    end: int

    @property
    def first(self) -> int:
        return self.numbers[0]

    @property
    def last(self) -> int:
        return self.numbers[-1]


@dataclass(frozen=True)
class Citation:
    reporter: str  # AIR, SCC, SCC_ONLINE, SCR, SCALE, INSC or CASE
    raw: str
    start: int
    end: int
    key: str  # canonical key, e.g. "AIR:1973:SC:1461"
    year: int | None = None
    volume: str | None = None
    page: int | None = None  # page, or serial number for neutral / OnLine citations
    court: str | None = None
    part: str | None = None  # SCC series part, e.g. "CRI"
    petitioner: str | None = None
    respondent: str | None = None
    pinpoint: Pinpoint | None = None


@dataclass(frozen=True)
class CitationGroup:
    """Parallel citations of one case (e.g. ``AIR 1973 SC 1461 : (1973) 4 SCC 225``) and its case name."""

    citations: tuple[Citation, ...]
    case_name: Citation | None
    start: int
    end: int
    pinpoint: Pinpoint | None = None

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.citations)


# ----------------------------------------------------------------------------------------------- reporters

_COURT = r"[A-Z][A-Za-z&]{1,5}"

_AIR = re.compile(
    rf"\bA\.?\s?I\.?\s?R\.?\s*\(?(?P<year>{YEAR})\)?\s+(?P<court>S\.?\s?C\.?|{_COURT}\.?)\s+(?P<page>\d{{1,5}})\b"
)
_SCC = re.compile(
    rf"\((?P<year>{YEAR})\)\s*(?P<vol>\d{{1,3}})\s*S\.?\s?C\.?\s?C\.?"
    rf"(?:\s*\((?P<part>Cri|Civ|L&S|Tax|Jour)\.?\))?\s+(?P<page>\d{{1,5}})\b"
)
_SCC_SUPP = re.compile(rf"\b(?P<year>{YEAR})\s+Supp\.?\s*\((?P<vol>\d{{1,2}})\)\s+SCC\s+(?P<page>\d{{1,5}})\b")
_SCC_ONLINE = re.compile(rf"\b(?P<year>{YEAR})\s+SCC\s+On[Ll]ine\s+(?P<court>{_COURT})\s+(?P<page>\d{{1,6}})\b")
_SCR = re.compile(
    rf"\[(?P<year>{YEAR})\]\s*(?:(?P<supp>Supp\.?)\s*)?(?P<vol>\d{{1,2}})?\s*S\.?\s?C\.?\s?R\.?\s+(?P<page>\d{{1,5}})\b"
)
_SCALE = re.compile(rf"\((?P<year>{YEAR})\)\s*(?P<vol>\d{{1,3}})\s*SCALE\s+(?P<page>\d{{1,5}})\b")
_INSC = re.compile(rf"\b(?P<year>{YEAR})[\s:]+INSC[\s:]+(?P<page>\d{{1,5}})\b")


def _court(raw: str) -> str:
    return re.sub(r"[.\s]", "", raw).upper()


def _build(reporter: str, m: re.Match[str]) -> Citation:
    g = m.groupdict()
    year = int(g["year"])
    page = int(g["page"])
    base = dict(reporter=reporter, raw=m.group(0), start=m.start(), end=m.end(), year=year, page=page)
    if reporter == "AIR":
        court = _court(g["court"])
        return Citation(**base, court=court, key=f"AIR:{year}:{court}:{page}")
    if reporter == "SCC":
        vol = g.get("vol")
        part = g.get("part")
        key = f"SCC:{year}:{vol}:{page}" + (f":{part.upper()}" if part else "")
        return Citation(**base, volume=vol, part=part.upper() if part else None, key=key)
    if reporter == "SCC_SUPP":
        vol = f"supp{g['vol']}"
        return Citation(**{**base, "reporter": "SCC"}, volume=vol, key=f"SCC:{year}:{vol}:{page}")
    if reporter == "SCC_ONLINE":
        court = _court(g["court"])
        return Citation(**base, court=court, key=f"SCCOL:{year}:{court}:{page}")
    if reporter == "SCR":
        vol = "supp" + (g.get("vol") or "") if g.get("supp") else (g.get("vol") or "")
        return Citation(**base, volume=vol or None, court="SC", key=f"SCR:{year}:{vol}:{page}")
    if reporter == "SCALE":
        return Citation(**base, volume=g["vol"], key=f"SCALE:{year}:{g['vol']}:{page}")
    return Citation(**base, court="SC", key=f"INSC:{year}:{page}")  # INSC


_REPORTER_PATTERNS = (
    ("AIR", _AIR),
    ("SCC", _SCC),
    ("SCC_SUPP", _SCC_SUPP),
    ("SCC_ONLINE", _SCC_ONLINE),
    ("SCR", _SCR),
    ("SCALE", _SCALE),
    ("INSC", _INSC),
)

# ----------------------------------------------------------------------------------------------- pinpoints

_NUMLIST = r"\d+(?:\s*(?:-|–|—|to|,|and|&)\s*\d+)*"
_PARA = re.compile(rf"(?:\bat\s+)?(?:\b(?:paras?|paragraphs?)\b\.?|¶{{1,2}})\s*(?P<nums>{_NUMLIST})", re.I)
_PAGE = re.compile(rf"(?:\bat\s+)?(?:\bpp?\.|\bpages?\b)\s*(?P<nums>{_NUMLIST})", re.I)
_RANGE_SEP = re.compile(r"\s*(?:-|–|—|to)\s*", re.I)


def _expand_numbers(text: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for chunk in re.split(r"\s*(?:,|and|&)\s*", text, flags=re.I):
        parts = _RANGE_SEP.split(chunk.strip())
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) <= int(parts[1]) <= int(parts[0]) + 500:
            numbers.extend(range(int(parts[0]), int(parts[1]) + 1))
        else:
            numbers.extend(int(p) for p in parts if p.isdigit())
    return tuple(numbers)


def _pinpoint_from(m: re.Match[str], kind: str, offset: int = 0) -> Pinpoint:
    return Pinpoint(kind=kind, numbers=_expand_numbers(m.group("nums")), raw=m.group(0), start=m.start() + offset, end=m.end() + offset)


def extract_pinpoints(text: str) -> list[Pinpoint]:
    """All paragraph/page references in ``text``, in order of appearance."""
    found = [_pinpoint_from(m, "para") for m in _PARA.finditer(text)]
    found += [_pinpoint_from(m, "page") for m in _PAGE.finditer(text)]
    return sorted(found, key=lambda p: p.start)


_SKIP = re.compile(r"[\s,;:(]*")


def _pinpoint_at(text: str, pos: int) -> Pinpoint | None:
    """A pinpoint starting right after ``pos`` (allowing ``,`` ``(`` and whitespace in between)."""
    start = _SKIP.match(text, pos).end()
    for kind, pattern in (("para", _PARA), ("page", _PAGE)):
        m = pattern.match(text, start)
        if m:
            return _pinpoint_from(m, kind)
    return None


# ----------------------------------------------------------------------------------------------- case names

_STOP_LEFT = {
    "in", "see", "also", "per", "cf", "thus", "held", "reliance", "relying", "following", "as", "on", "the", "case",
    "of", "and", "but", "while", "where", "when", "if", "that", "by", "from", "under", "with", "vide", "ie", "eg",
    "decision", "judgment", "judgement", "decided", "overruled", "approved", "affirmed", "reported", "citing",
    "cited", "to", "at", "this", "court", "observed", "ratio", "applied", "followed", "distinguished",
}
_ABBREV = {
    "ltd", "co", "pvt", "inc", "corp", "ors", "anr", "etc", "dr", "mr", "mrs", "ms", "smt", "shri", "sri", "govt",
    "dept", "engg", "indl", "no", "prof", "hon", "st", "mt", "m/s", "assn", "mfg", "svc",
}
_CONNECTORS = {"of", "and", "the", "&", "for", "in", "de", "la", "on"}
_WEAK_CONNECTORS = {"the", "in", "on"}  # only part of a name directly after another connector ("State of the ...")
_RIGHT_STOP ={"AIR", "SCC", "SCR", "INSC", "SCALE", "Supp", "Para", "Paras", "Section", "Article", "Order", "Rule"}
_VS = re.compile(r"(?<=\S)\s+(?:v\.?|vs\.?|versus)\s+(?=[A-Z])")
_TOKEN = re.compile(r"\S+")


def _is_abbrev(token: str) -> bool:
    t = token.rstrip(".,;:)").lower()
    return t in _ABBREV or bool(re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?\.?", token.rstrip(",;:)")))


def _is_name_word(token: str) -> bool:
    return bool(re.match(r"[A-Z][\w.'’&/-]*", token)) and token.rstrip(".,;:)") not in _RIGHT_STOP


def _left_party(text: str, end: int) -> tuple[int, str] | None:
    tokens = list(_TOKEN.finditer(text[:end]))[-9:]
    picked: list[re.Match[str]] = []
    for m in reversed(tokens):
        tok = m.group(0)
        if tok[-1] in ",;:)":
            break
        if tok[-1] == "." and not _is_abbrev(tok):
            break
        if _is_name_word(tok) or tok.lower() in _CONNECTORS:
            picked.append(m)
        else:
            break
    picked.reverse()
    while picked and (picked[0].group(0).lower().rstrip(".") in _STOP_LEFT or not picked[0].group(0)[0].isupper()):
        picked.pop(0)
    if not picked:
        return None
    return picked[0].start(), text[picked[0].start() : end].strip()


def _right_party(text: str, start: int) -> tuple[int, str] | None:
    picked: list[tuple[int, int]] = []
    prev_is_connector = False
    for m in list(_TOKEN.finditer(text, start))[:9]:
        tok = m.group(0)
        bare = tok.rstrip(".,;:)")
        is_connector = bare.lower() in _CONNECTORS and not _is_name_word(tok)
        if is_connector and bare.lower() in _WEAK_CONNECTORS and not prev_is_connector and picked:
            break  # "Ajaib Singh the Court ..." - "the" right after a name word ends the party name
        if not (_is_name_word(tok) or is_connector):
            break
        prev_is_connector = is_connector
        picked.append((m.start(), m.start() + len(bare) if bare else m.end()))
        if tok[-1] in ",;:)" or (tok[-1] == "." and not _is_abbrev(tok)):
            break
        if tok[-1] == "." and _is_abbrev(tok):
            picked[-1] = (m.start(), m.end())
    while picked and text[picked[-1][0] : picked[-1][1]].lower().strip(".") in _CONNECTORS:
        picked.pop()
    if not picked:
        return None
    return picked[-1][1], text[picked[0][0] : picked[-1][1]].strip()


def _party_key(name: str) -> str:
    words = re.sub(r"[.,'’()]", "", name.lower().replace("&", " and ")).split()
    words = [w for w in words if w not in {"the", "ors", "anr", "others", "etc"}]
    while words and words[-1] == "and":
        words.pop()
    return " ".join(words)


def _case_names(text: str) -> list[Citation]:
    cites = []
    for vs in _VS.finditer(text):
        left = _left_party(text, vs.start())
        right = _right_party(text, vs.end())
        if not left or not right:
            continue
        petitioner, respondent = left[1], right[1]
        start, end = left[0], right[0]
        cites.append(
            Citation(
                reporter="CASE",
                raw=text[start:end],
                start=start,
                end=end,
                key=f"CASE:{_party_key(petitioner)} v {_party_key(respondent)}",
                petitioner=petitioner,
                respondent=respondent,
            )
        )
    return cites


# ----------------------------------------------------------------------------------------------- public API


def _drop_overlaps(cites: Iterable[Citation]) -> list[Citation]:
    kept: list[Citation] = []
    for c in sorted(cites, key=lambda c: (c.start, -(c.end - c.start))):
        if kept and c.start < kept[-1].end:
            continue
        kept.append(c)
    return kept


def extract_citations(text: str, include_case_names: bool = True) -> list[Citation]:
    """Every citation in ``text`` in order of appearance, each with its pinpoint if one follows it."""
    reporter_cites: list[Citation] = []
    for name, pattern in _REPORTER_PATTERNS:
        reporter_cites.extend(_build(name, m) for m in pattern.finditer(text))
    cites = _drop_overlaps(reporter_cites)
    cites = [replace(c, pinpoint=_pinpoint_at(text, c.end)) for c in cites]
    if include_case_names:
        cites = sorted([*cites, *_case_names(text)], key=lambda c: c.start)
    return cites


_PARALLEL_GAP = re.compile(r"^\s*[:;,=]\s*$")
_NAME_GAP = re.compile(r"^[\s,(\[]*$")


def extract_groups(text: str) -> list[CitationGroup]:
    """Group parallel citations of the same case and attach the case name that introduces them.

    Two reporter citations are parallel when only ``: ; , =`` separates them and they use different
    reporters ("AIR ..., AIR ..." is a string cite of two cases, not a parallel pair).
    """
    everything = extract_citations(text)
    reporters = [c for c in everything if c.reporter != "CASE"]
    names = [c for c in everything if c.reporter == "CASE"]
    groups: list[list[Citation]] = []
    for c in reporters:
        if groups:
            last = groups[-1][-1]
            same_reporter = any(g.reporter == c.reporter for g in groups[-1])
            if _PARALLEL_GAP.match(text[last.end : c.start]) and not same_reporter:
                groups[-1].append(c)
                continue
        groups.append([c])
    result = []
    for members in groups:
        first, last = members[0], members[-1]
        name = next((n for n in reversed(names) if n.end <= first.start and _NAME_GAP.match(text[n.end : first.start])), None)
        pin = next((m.pinpoint for m in reversed(members) if m.pinpoint), None)
        result.append(CitationGroup(tuple(members), name, (name or first).start, last.end, pin))
    return result


def normalise_citation(text: str) -> list[str]:
    """Canonical keys of the reporter citations in ``text`` (case names excluded)."""
    return [c.key for c in extract_citations(text, include_case_names=False)]
