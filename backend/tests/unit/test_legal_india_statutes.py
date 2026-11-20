import pytest

from app.legal_india.statutes import extract_statutes, normalise_act_name


def refs(text: str) -> list[tuple[str | None, str, str]]:
    return [(r.act, r.kind, r.provision) for r in extract_statutes(text)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Section 302 of the Indian Penal Code", [("IPC", "section", "302")]),
        ("Section 302 IPC", [("IPC", "section", "302")]),
        ("S. 482 CrPC", [("CRPC", "section", "482")]),
        ("Section 482 of the Code of Criminal Procedure, 1973", [("CRPC", "section", "482")]),
        ("Section 482 Cr.P.C.", [("CRPC", "section", "482")]),
        ("Section 103 of the BNS", [("BNS", "section", "103")]),
        ("Section 528 of the BNSS", [("BNSS", "section", "528")]),
        ("Section 63 of the Bharatiya Sakshya Adhiniyam, 2023", [("BSA", "section", "63")]),
        ("Section 65B of the Indian Evidence Act, 1872", [("EVIDENCE", "section", "65B")]),
        ("Section 498A IPC", [("IPC", "section", "498A")]),
        ("Section 482(1) CrPC", [("CRPC", "section", "482(1)")]),
        ("u/s 420 IPC", [("IPC", "section", "420")]),
        ("Section 9 CPC", [("CPC", "section", "9")]),
        ("Article 21 of the Constitution of India", [("CONSTITUTION", "article", "21")]),
        ("Art. 226", [("CONSTITUTION", "article", "226")]),
        ("Article 19(1)(a)", [("CONSTITUTION", "article", "19(1)(a)")]),
        ("Article 21A", [("CONSTITUTION", "article", "21A")]),
    ],
)
def test_single_reference_forms(text: str, expected: list) -> None:
    assert refs(f"The petitioner invoked {text} before the court.") == expected


def test_lists_share_the_named_act() -> None:
    assert refs("charged under Sections 302 and 34 of the IPC") == [("IPC", "section", "302"), ("IPC", "section", "34")]
    assert refs("Sections 302, 34 IPC") == [("IPC", "section", "302"), ("IPC", "section", "34")]
    assert refs("Section 302/34 IPC") == [("IPC", "section", "302"), ("IPC", "section", "34")]
    assert refs("Articles 14, 19 and 21 of the Constitution") == [
        ("CONSTITUTION", "article", "14"),
        ("CONSTITUTION", "article", "19"),
        ("CONSTITUTION", "article", "21"),
    ]


def test_order_and_rule() -> None:
    assert refs("an application under Order VII Rule 11 CPC") == [("CPC", "order_rule", "Order 7 Rule 11")]
    assert refs("O. XXI R. 58 of the Code of Civil Procedure") == [("CPC", "order_rule", "Order 21 Rule 58")]
    assert refs("Order 21 Rules 58 and 97 CPC") == [("CPC", "order_rule", "Order 21 Rule 58"), ("CPC", "order_rule", "Order 21 Rule 97")]


def test_bns_is_not_confused_with_bnss() -> None:
    assert refs("Section 35 BNSS and Section 3 BNS") == [("BNSS", "section", "35"), ("BNS", "section", "3")]


def test_act_not_stated_gives_none_for_sections_but_constitution_for_articles() -> None:
    assert refs("Section 302 was invoked") == [(None, "section", "302")]
    assert refs("Article 32 was invoked") == [("CONSTITUTION", "article", "32")]


def test_unmodelled_act_is_reported_by_name() -> None:
    (ref,) = extract_statutes("a complaint under Section 138 of the Negotiable Instruments Act, 1881")
    assert ref.act == "OTHER" and ref.act_name == "Negotiable Instruments Act, 1881" and ref.provision == "138"


def test_keys_and_base() -> None:
    (ref,) = extract_statutes("Section 482(1) CrPC")
    assert ref.key == "CRPC:482(1)" and ref.base == "482"
    assert ref.raw == "Section 482(1) CrPC"


def test_spans_point_into_the_text() -> None:
    text = "Held: Section 302 of the Indian Penal Code applies."
    (ref,) = extract_statutes(text)
    assert text[ref.start : ref.end] == "Section 302 of the Indian Penal Code"


def test_no_false_positives() -> None:
    assert refs("The Secretary sat in Section B of the hall; U.S. 5 cases were heard.") == []
    assert refs("in the lower article of the agreement") == []


@pytest.mark.parametrize(
    "name,code",
    [("IPC", "IPC"), ("Cr.P.C.", "CRPC"), ("CrPC", "CRPC"), ("BNSS", "BNSS"), ("Bharatiya Nyaya Sanhita, 2023", "BNS"),
     ("Evidence Act", "EVIDENCE"), ("Constitution of India", "CONSTITUTION"), ("C.P.C.", "CPC"), ("Banana Act", None)],
)
def test_act_aliases(name: str, code: str | None) -> None:
    assert normalise_act_name(name) == code
