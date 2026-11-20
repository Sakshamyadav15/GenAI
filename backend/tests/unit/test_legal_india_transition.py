import json
from datetime import date

import pytest

from app.legal_india.statutes import extract_statutes
from app.legal_india.transition import (
    CUTOFF,
    DEFAULT_MAP_PATH,
    applicable_code,
    base_section,
    check_cited_section,
    load_section_map,
)


def ref(text: str):
    return extract_statutes(text)[0]


def test_cutoff_boundaries() -> None:
    assert CUTOFF == date(2024, 7, 1)
    assert applicable_code(date(2024, 6, 30)) == "IPC"
    assert applicable_code(date(2024, 7, 1)) == "BNS"
    assert applicable_code(date(1990, 1, 1)) == "IPC"
    assert applicable_code(date(2025, 3, 3)) == "BNS"


def test_procedure_and_evidence_are_not_decided_by_offence_date() -> None:
    assert applicable_code(date(2024, 8, 1), family="procedure") is None
    assert applicable_code(date(2024, 8, 1), family="evidence") is None


def test_section_map_loads_and_every_entry_is_marked_unverified() -> None:
    raw = json.loads(DEFAULT_MAP_PATH.read_text(encoding="utf-8"))
    assert "NOT COMPLETE AND NOT VERIFIED" in raw["meta"]["warning"]
    assert raw["mappings"] and all(m["verified"] is False and "official" in m["source_note"].lower() for m in raw["mappings"])
    assert {m["old_code"] for m in raw["mappings"]} == {"IPC", "CRPC", "EVIDENCE"}


def test_forward_and_reverse_lookup() -> None:
    smap = load_section_map()
    (e,) = smap.to_new("IPC", "302")
    assert (e.new_code, e.new) == ("BNS", "103(1)")
    assert smap.to_new("ipc", "420")[0].new == "318(4)"
    assert smap.to_new("CRPC", "482")[0].new == "528"
    assert smap.to_new("EVIDENCE", "65B")[0].new == "63"
    assert smap.to_old("BNS", "103")[0].old == "302"
    assert smap.to_old("BNS", "103(1)")[0].old == "302"
    assert smap.to_new("IPC", "999") == []


def test_map_has_no_duplicate_old_sections() -> None:
    smap = load_section_map()
    seen = [(e.old_code, e.old) for e in smap.entries]
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize("value,base", [("103(1)", "103"), ("498a", "498A"), ("65B", "65B"), ("482", "482")])
def test_base_section(value: str, base: str) -> None:
    assert base_section(value) == base


def test_base_section_rejects_non_numbers() -> None:
    with pytest.raises(ValueError):
        base_section("Article")


def test_old_code_for_new_offence_is_flagged_with_suggestion() -> None:
    finding = check_cited_section(ref("Section 302 of the IPC"), date(2024, 9, 15))
    assert finding.status == "wrong_code" and finding.expected_code == "BNS"
    assert finding.suggestion == "BNS 103(1)"
    assert "unverified" in finding.message


def test_new_code_for_old_offence_is_flagged() -> None:
    finding = check_cited_section(ref("Section 103 BNS"), date(2023, 1, 1))
    assert finding.status == "wrong_code" and finding.expected_code == "IPC"
    assert finding.suggestion == "IPC 302"


def test_correct_code_is_ok() -> None:
    assert check_cited_section(ref("Section 302 IPC"), date(2024, 6, 30)).status == "ok"
    assert check_cited_section(ref("Section 103 BNS"), date(2024, 7, 1)).status == "ok"


def test_wrong_code_without_table_entry_has_no_suggestion() -> None:
    finding = check_cited_section(ref("Section 211 IPC"), date(2025, 1, 1))
    assert finding.status == "wrong_code" and finding.suggestion is None


def test_unknown_date_and_non_substantive_acts() -> None:
    assert check_cited_section(ref("Section 302 IPC"), None).status == "undetermined"
    assert check_cited_section(ref("Section 482 CrPC"), date(2025, 1, 1)).status == "not_assessed"
    assert check_cited_section(ref("Article 21"), date(2025, 1, 1)).status == "not_assessed"
