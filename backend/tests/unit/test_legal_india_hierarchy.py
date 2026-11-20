import pytest

from app.legal_india.hierarchy import (
    FOREIGN,
    HIGH_COURTS,
    SUPREME_COURT,
    Court,
    court_from_name,
    is_binding,
    subordinate_court,
)

DELHI_HC = HIGH_COURTS["High Court of Delhi"]
BOMBAY_HC = HIGH_COURTS["High Court of Bombay"]


@pytest.mark.parametrize(
    "forum",
    [DELHI_HC, "Bombay High Court", subordinate_court("Sessions Court, Pune"), subordinate_court("District Court, Delhi"), Court("NCLT", "other")],
)
def test_supreme_court_binds_every_other_forum(forum) -> None:
    result = is_binding(forum, SUPREME_COURT)
    assert result.status == "binding" and "141" in result.reason


def test_supreme_court_citing_itself_is_not_decided_by_this_model() -> None:
    assert is_binding("Supreme Court of India", "Supreme Court of India").status == "persuasive"


def test_high_court_binds_subordinate_courts_in_its_territory_only() -> None:
    forum = subordinate_court("District Court, Mumbai")
    assert is_binding(forum, BOMBAY_HC, territory="Maharashtra").status == "binding"
    assert is_binding(forum, BOMBAY_HC, territory="Goa").status == "binding"
    assert is_binding(forum, DELHI_HC, territory="Maharashtra").status == "persuasive"


def test_high_court_needs_territory_for_subordinate_forum() -> None:
    assert is_binding(subordinate_court("Sessions Court"), DELHI_HC).status == "undetermined"


def test_high_court_unknown_territory_is_undetermined() -> None:
    unknown = Court("High Court of Sikkim", "high")
    assert is_binding(subordinate_court("District Court, Gangtok"), unknown, territory="Sikkim").status == "undetermined"


def test_high_court_is_persuasive_for_other_high_courts_and_supreme_court() -> None:
    assert is_binding(BOMBAY_HC, DELHI_HC).status == "persuasive"
    assert is_binding(SUPREME_COURT, DELHI_HC).status == "persuasive"
    assert "coordinate" in is_binding(DELHI_HC, DELHI_HC).reason


def test_subordinate_and_foreign_authority_are_persuasive() -> None:
    assert is_binding(DELHI_HC, subordinate_court("Sessions Court, Delhi")).status == "persuasive"
    assert is_binding(SUPREME_COURT, Court("House of Lords", FOREIGN)).status == "persuasive"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Supreme Court", "Supreme Court of India"),
        ("Hon'ble Supreme Court of India", "Supreme Court of India"),
        ("Delhi High Court", "High Court of Delhi"),
        ("High Court of Punjab & Haryana", "High Court of Punjab and Haryana"),
        ("Bombay High Court", "High Court of Bombay"),
    ],
)
def test_court_from_name(name: str, expected: str) -> None:
    assert court_from_name(name).name == expected


def test_unknown_high_court_and_trial_court() -> None:
    assert court_from_name("Sikkim High Court").level == "high"
    assert court_from_name("Sikkim High Court").territory == frozenset()
    assert court_from_name("Sessions Court, Pune").level == "subordinate"


def test_strings_are_accepted_by_is_binding() -> None:
    assert is_binding("Sessions Court, Pune", "Supreme Court").status == "binding"
    assert is_binding("Sessions Court, Pune", "Bombay High Court", territory="Maharashtra").status == "binding"
