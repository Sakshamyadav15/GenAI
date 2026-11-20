import pytest

from app.legal_india.citations import extract_citations, extract_groups, extract_pinpoints, normalise_citation


def keys(text: str, names: bool = False) -> list[str]:
    return [c.key for c in extract_citations(text, include_case_names=names)]


@pytest.mark.parametrize(
    "text,key",
    [
        ("AIR 1973 SC 1461", "AIR:1973:SC:1461"),
        ("A.I.R. 1973 S.C. 1461", "AIR:1973:SC:1461"),
        ("AIR 1950 SC 27", "AIR:1950:SC:27"),
        ("AIR 2005 Del 12", "AIR:2005:DEL:12"),
        ("AIR 1985 P&H 41", "AIR:1985:P&H:41"),
        ("(2017) 10 SCC 1", "SCC:2017:10:1"),
        ("(2017) 10 S.C.C. 1", "SCC:2017:10:1"),
        ("(2013) 1 SCC (Cri) 100", "SCC:2013:1:100:CRI"),
        ("1992 Supp (1) SCC 335", "SCC:1992:supp1:335"),
        ("2017 SCC OnLine SC 1234", "SCCOL:2017:SC:1234"),
        ("2019 SCC OnLine Del 8754", "SCCOL:2019:DEL:8754"),
        ("[1973] Supp. 1 SCR 1", "SCR:1973:supp1:1"),
        ("[1973] 3 SCR 1", "SCR:1973:3:1"),
        ("(2017) 7 SCALE 12", "SCALE:2017:7:12"),
        ("2023 INSC 1017", "INSC:2023:1017"),
        ("2023:INSC:1017", "INSC:2023:1017"),
    ],
)
def test_single_citation_formats(text: str, key: str) -> None:
    assert keys(f"The Court relied on {text} for this.") == [key]


def test_structured_fields() -> None:
    (c,) = extract_citations("See (1973) 4 SCC 225.", include_case_names=False)
    assert (c.reporter, c.year, c.volume, c.page, c.raw) == ("SCC", 1973, "4", 225, "(1973) 4 SCC 225")
    (a,) = extract_citations("AIR 1973 SC 1461", include_case_names=False)
    assert (a.reporter, a.court, a.page, a.start, a.end) == ("AIR", "SC", 1461, 0, 16)


def test_multiple_citations_in_a_paragraph() -> None:
    text = "Reliance was placed on AIR 1950 SC 27, (2017) 10 SCC 1 and 2023 INSC 1017; see also AIR 1978 SC 597."
    assert keys(text) == ["AIR:1950:SC:27", "SCC:2017:10:1", "INSC:2023:1017", "AIR:1978:SC:597"]


def test_parallel_citations_are_grouped() -> None:
    text = "Kesavananda Bharati v. State of Kerala, AIR 1973 SC 1461 : (1973) 4 SCC 225 settled the basic structure doctrine."
    (group,) = extract_groups(text)
    assert group.keys == ("AIR:1973:SC:1461", "SCC:1973:4:225")
    assert group.case_name is not None
    assert group.case_name.petitioner == "Kesavananda Bharati"
    assert group.case_name.respondent == "State of Kerala"


def test_string_cite_of_same_reporter_is_not_parallel() -> None:
    groups = extract_groups("AIR 1950 SC 27, AIR 1952 SC 196, AIR 1954 SC 1")
    assert [g.keys for g in groups] == [("AIR:1950:SC:27",), ("AIR:1952:SC:196",), ("AIR:1954:SC:1",)]


def test_three_reporters_in_one_group() -> None:
    (group,) = extract_groups("AIR 1973 SC 1461 : (1973) 4 SCC 225 : [1973] Supp. 1 SCR 1")
    assert group.keys == ("AIR:1973:SC:1461", "SCC:1973:4:225", "SCR:1973:supp1:1")


@pytest.mark.parametrize(
    "text,kind,numbers",
    [
        ("AIR 1973 SC 1461, para 45", "para", (45,)),
        ("(2017) 10 SCC 1, at para 45", "para", (45,)),
        ("(2017) 10 SCC 1 (paras 12-14)", "para", (12, 13, 14)),
        ("(2017) 10 SCC 1, paras 12 and 14", "para", (12, 14)),
        ("(2017) 10 SCC 1, paras. 3, 5 and 9", "para", (3, 5, 9)),
        ("AIR 1973 SC 1461, at p. 300", "page", (300,)),
        ("AIR 1973 SC 1461 at pp. 300-302", "page", (300, 301, 302)),
        ("AIR 1973 SC 1461, at page 300", "page", (300,)),
        ("AIR 1973 SC 1461, ¶ 7", "para", (7,)),
    ],
)
def test_pinpoint_attached_to_citation(text: str, kind: str, numbers: tuple[int, ...]) -> None:
    (c,) = extract_citations(text, include_case_names=False)
    assert c.pinpoint is not None
    assert (c.pinpoint.kind, c.pinpoint.numbers) == (kind, numbers)


def test_pinpoint_on_last_of_parallel_group() -> None:
    (group,) = extract_groups("AIR 1973 SC 1461 : (1973) 4 SCC 225, para 45")
    assert group.pinpoint is not None and group.pinpoint.numbers == (45,)


def test_extract_pinpoints_standalone() -> None:
    found = extract_pinpoints("In para 45 and at p. 300 and also paras 12-14 the Court held.")
    assert [(p.kind, p.numbers) for p in found] == [("para", (45,)), ("page", (300,)), ("para", (12, 13, 14))]


def test_no_pinpoint_when_text_continues() -> None:
    (c,) = extract_citations("AIR 1973 SC 1461 is the leading case on paragraph structure.", include_case_names=False)
    assert c.pinpoint is None


@pytest.mark.parametrize(
    "text,petitioner,respondent",
    [
        ("The decision in Kesavananda Bharati v. State of Kerala held", "Kesavananda Bharati", "State of Kerala"),
        ("Maneka Gandhi v. Union of India, AIR 1978 SC 597", "Maneka Gandhi", "Union of India"),
        ("See also K.S. Puttaswamy v. Union of India (2017) 10 SCC 1", "K.S. Puttaswamy", "Union of India"),
        ("As held in Vishaka v. State of Rajasthan AIR 1997 SC 3011.", "Vishaka", "State of Rajasthan"),
        ("In State of Punjab v. Ajaib Singh the Court noted", "State of Punjab", "Ajaib Singh"),
        ("Bharat Petroleum Corporation Ltd. v. Mumbai Shramik Sangra, (2001) 4 SCC 448", "Bharat Petroleum Corporation Ltd.", "Mumbai Shramik Sangra"),
    ],
)
def test_party_name_short_forms(text: str, petitioner: str, respondent: str) -> None:
    cases = [c for c in extract_citations(text) if c.reporter == "CASE"]
    assert len(cases) == 1
    assert (cases[0].petitioner, cases[0].respondent) == (petitioner, respondent)


def test_case_name_key_is_normalised() -> None:
    a = [c for c in extract_citations("Kesavananda Bharati v. State of Kerala") if c.reporter == "CASE"][0]
    b = [c for c in extract_citations("In Kesavananda Bharati vs. The State of Kerala & Ors. held") if c.reporter == "CASE"][0]
    assert a.key == b.key == "CASE:kesavananda bharati v state of kerala"


def test_group_without_case_name() -> None:
    (group,) = extract_groups("It was held (2017) 10 SCC 1 that privacy is fundamental.")
    assert group.case_name is None


def test_normalise_citation_ignores_case_names() -> None:
    assert normalise_citation("Maneka Gandhi v. Union of India, AIR 1978 SC 597") == ["AIR:1978:SC:597"]


def test_no_false_positives_on_plain_numbers() -> None:
    assert keys("In 1973 the court sat for 68 days and decided 13 judges SCC matters 4 times.") == []


def test_citations_are_in_order_and_non_overlapping() -> None:
    text = "AIR 1973 SC 1461 : (1973) 4 SCC 225 and 2023 INSC 1017"
    cites = extract_citations(text, include_case_names=False)
    assert [c.start for c in cites] == sorted(c.start for c in cites)
    assert all(a.end <= b.start for a, b in zip(cites, cites[1:]))
