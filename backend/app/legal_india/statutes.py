"""Extraction of statute references ("Section 302 of the IPC", "Article 21", "Order VII Rule 11 CPC")."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: longer / more specific aliases first (BNSS before BNS, etc.).
_ACTS: tuple[tuple[str, str], ...] = (
    ("BNSS", r"Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita(?:,?\s*2023)?|B\.?\s?N\.?\s?S\.?\s?S\.?"),
    ("BNS", r"Bharatiya\s+Nyaya\s+Sanhita(?:,?\s*2023)?|B\.?\s?N\.?\s?S\.?"),
    ("BSA", r"Bharatiya\s+Sakshya\s+Adhiniyam(?:,?\s*2023)?|B\.?\s?S\.?\s?A\.?"),
    ("CRPC", r"Code\s+of\s+Criminal\s+Procedure(?:,?\s*1973)?|Cr\.?\s?P\.?\s?C\.?"),
    ("CPC", r"Code\s+of\s+Civil\s+Procedure(?:,?\s*1908)?|C\.?\s?P\.?\s?C\.?"),
    ("IPC", r"Indian\s+Penal\s+Code(?:,?\s*1860)?|I\.?\s?P\.?\s?C\.?"),
    ("EVIDENCE", r"(?:Indian\s+)?Evidence\s+Act(?:,?\s*1872)?"),
    ("CONSTITUTION", r"Constitution(?:\s+of\s+India)?"),
)
_ACT_RES = tuple((code, re.compile(rf"(?:{pat})(?![A-Za-z])", re.I)) for code, pat in _ACTS)
_TAIL_PREFIX = re.compile(r"[\s,]*(?:of|under|in|to)?\s*(?:the\s+)?", re.I)
_OTHER_ACT = re.compile(r"[\s,]*(?:of|under)\s+(?:the\s+)?(?P<name>(?:[A-Z][A-Za-z&]*\s+){1,6}(?:Act|Code|Rules|Regulations)(?:,?\s*\d{4})?)")

_NUM = r"\d+[A-Za-z]{0,2}(?:\(\w{1,4}\))*"
_LIST = rf"{_NUM}(?:\s*(?:,|/|&|and|or)\s*{_NUM})*"
_SECTION = re.compile(rf"(?<![A-Za-z.])(?P<label>Sections?|Secs?\.?|Ss?\.|u/s\.?)\s*(?P<nums>{_LIST})", re.I)
_ARTICLE = re.compile(rf"(?<![A-Za-z])(?P<label>Articles?|Arts?\.?)\s*(?P<nums>{_LIST})", re.I)
_ORDER = re.compile(
    r"(?<![A-Za-z])(?:Order|O\.)\s*(?P<order>[IVXLC]+|\d+)\s*,?\s*(?:Rules?|R\.)\s*"
    rf"(?P<rules>\d+[A-Za-z]?(?:\(\w{{1,4}}\))*(?:\s*(?:,|and|&)\s*\d+[A-Za-z]?)*)",
    re.I,
)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


@dataclass(frozen=True)
class StatuteRef:
    act: str | None  # IPC, BNS, CRPC, BNSS, EVIDENCE, BSA, CPC, CONSTITUTION, OTHER, or None if not stated
    act_name: str | None  # name as written for acts we do not model (act == "OTHER")
    kind: str  # "section", "article" or "order_rule"
    provision: str  # e.g. "302", "498A", "482(1)", "Order 7 Rule 11"
    base: str  # provision without sub-sections, e.g. "482"
    raw: str  # the whole mention, e.g. "Sections 302 and 34 of the IPC"
    start: int
    end: int

    @property
    def key(self) -> str:
        return f"{self.act or 'UNKNOWN'}:{self.provision}"


def normalise_act_name(name: str) -> str | None:
    """Canonical act code for an act name or abbreviation ("Cr.P.C." -> "CRPC"), or None."""
    name = name.strip()
    for code, pattern in _ACT_RES:
        if pattern.fullmatch(name):
            return code
    return None


def _act_after(text: str, pos: int) -> tuple[str | None, str | None, int]:
    prefix = _TAIL_PREFIX.match(text, pos)
    start = prefix.end() if prefix else pos
    for code, pattern in _ACT_RES:
        m = pattern.match(text, start)
        if m:
            return code, None, m.end()
    other = _OTHER_ACT.match(text, pos)
    if other:
        return "OTHER", other.group("name").strip(), other.end()
    return None, None, pos


def _split(nums: str) -> list[str]:
    return [p.strip() for p in re.split(r"\s*(?:,|/|&|\band\b|\bor\b)\s*", nums, flags=re.I) if p.strip()]


def _normalise_provision(p: str) -> tuple[str, str]:
    m = re.match(r"(\d+)([A-Za-z]{0,2})((?:\(\w{1,4}\))*)", p)
    assert m
    base = m.group(1) + m.group(2).upper()
    return base + m.group(3), base


def _roman_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        value = _ROMAN[ch]
        total += -value if value < prev else value
        prev = max(prev, value)
    return total


def extract_statutes(text: str) -> list[StatuteRef]:
    """Statute references in order of appearance; one StatuteRef per provision number.

    The act is the one named right after the provision(s); for "Article ..." without a named act it
    defaults to the Constitution. Chains such as "302 IPC and 304 BNS" are only partly supported:
    the second item needs its own label ("Section 304 BNS").
    """
    refs: list[StatuteRef] = []
    for pattern, kind in ((_SECTION, "section"), (_ARTICLE, "article")):
        for m in pattern.finditer(text):
            act, act_name, end = _act_after(text, m.end())
            if act is None and kind == "article":
                act = "CONSTITUTION"
            raw = text[m.start() : end]
            for item in _split(m.group("nums")):
                provision, base = _normalise_provision(item)
                refs.append(StatuteRef(act, act_name, kind, provision, base, raw, m.start(), end))
    for m in _ORDER.finditer(text):
        act, act_name, end = _act_after(text, m.end())
        order = _roman_to_int(m.group("order"))
        for rule in _split(m.group("rules")):
            provision, base = _normalise_provision(rule)
            refs.append(StatuteRef(act, act_name, "order_rule", f"Order {order} Rule {provision}", f"O{order}R{base}", text[m.start() : end], m.start(), end))
    return sorted(refs, key=lambda r: (r.start, r.provision))
