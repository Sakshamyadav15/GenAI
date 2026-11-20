"""Simple model of precedent hierarchy in India.

Rules modelled:
* Art. 141: law declared by the Supreme Court is binding on all courts in India.
* A High Court's decision binds the subordinate courts within its territorial jurisdiction.
* Everything else is treated as persuasive.

Not modelled: bench strength (larger-bench rule, coordinate benches), ratio vs obiter, tribunals,
overruling, per-curiam decisions, and High Court benches sitting outside the principal seat.
Territory lists below are not exhaustive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SUPREME = "supreme"
HIGH = "high"
SUBORDINATE = "subordinate"
OTHER = "other"  # tribunals, commissions and similar
FOREIGN = "foreign"


@dataclass(frozen=True)
class Court:
    name: str
    level: str
    territory: frozenset[str] = field(default_factory=frozenset)  # states/UTs a High Court exercises jurisdiction over


SUPREME_COURT = Court("Supreme Court of India", SUPREME)


def _hc(name: str, *territory: str) -> Court:
    return Court(f"High Court of {name}", HIGH, frozenset(territory))


HIGH_COURTS: dict[str, Court] = {
    c.name: c
    for c in (
        _hc("Delhi", "Delhi"),
        _hc("Bombay", "Maharashtra", "Goa", "Dadra and Nagar Haveli and Daman and Diu"),
        _hc("Madras", "Tamil Nadu", "Puducherry"),
        _hc("Calcutta", "West Bengal", "Andaman and Nicobar Islands"),
        _hc("Kerala", "Kerala", "Lakshadweep"),
        _hc("Karnataka", "Karnataka"),
        _hc("Allahabad", "Uttar Pradesh"),
        _hc("Gujarat", "Gujarat"),
        _hc("Punjab and Haryana", "Punjab", "Haryana", "Chandigarh"),
    )
}


def subordinate_court(name: str) -> Court:
    return Court(name, SUBORDINATE)


@dataclass(frozen=True)
class BindingResult:
    status: str  # binding | persuasive | undetermined
    reason: str


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("&", "and").split())


def court_from_name(name: str) -> Court:
    """Resolve a court name ("Supreme Court", "Delhi High Court", "High Court of Delhi") to a Court."""
    n = _norm(name)
    if "supreme court" in n:
        return SUPREME_COURT
    for court in HIGH_COURTS.values():
        place = _norm(court.name).removeprefix("high court of ")
        if "high court" in n and place in n:
            return court
    if "high court" in n:
        return Court(name.strip(), HIGH)  # a High Court we hold no territory data for
    return Court(name.strip(), SUBORDINATE)


def _as_court(value: Court | str) -> Court:
    return value if isinstance(value, Court) else court_from_name(value)


def is_binding(citing_forum: Court | str, authority_court: Court | str, territory: str | None = None) -> BindingResult:
    """Is a decision of ``authority_court`` binding on ``citing_forum``?

    ``territory`` is the state/UT in which the citing forum sits (needed for subordinate courts).
    """
    forum, authority = _as_court(citing_forum), _as_court(authority_court)

    if authority.level == FOREIGN:
        return BindingResult("persuasive", "Foreign authority is never binding in Indian courts.")
    if authority.level == SUPREME:
        if forum.level == SUPREME:
            return BindingResult("persuasive", "Bench strength (larger-bench rule) is not modelled for the Supreme Court citing itself.")
        return BindingResult("binding", "Article 141: law declared by the Supreme Court binds all courts in India.")
    if authority.level == HIGH:
        if forum.level == SUPREME:
            return BindingResult("persuasive", "A High Court decision does not bind the Supreme Court.")
        if forum.level == HIGH:
            same = forum.name == authority.name
            return BindingResult("persuasive", "Same High Court: coordinate-bench rules are not modelled." if same else "A High Court decision is persuasive before another High Court.")
        if forum.level in (SUBORDINATE, OTHER):
            if not authority.territory:
                return BindingResult("undetermined", f"No territory data for {authority.name}.")
            if not territory:
                return BindingResult("undetermined", "Territory of the citing forum is needed to apply High Court jurisdiction.")
            if territory in authority.territory:
                return BindingResult("binding", f"{authority.name} binds subordinate courts in {territory}." if forum.level == SUBORDINATE else f"Tribunal binding status is not modelled; {authority.name} is treated as binding in {territory}.")
            return BindingResult("persuasive", f"{territory} lies outside the territorial jurisdiction of {authority.name}.")
    return BindingResult("persuasive", "Decisions of subordinate courts and tribunals are persuasive only.")
