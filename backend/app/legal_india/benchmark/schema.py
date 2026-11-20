from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PerturbationType = Literal[
    "fabricated_case",
    "wrong_case",
    "wrong_paragraph",
    "wrong_code",
    "altered_quote",
    "near_paragraph",
    "overbroad_holding",
    "swapped_party_or_outcome",
    "none",
]
PERTURBATIONS: tuple[str, ...] = (
    "fabricated_case",
    "wrong_case",
    "wrong_paragraph",
    "wrong_code",
    "altered_quote",
    "near_paragraph",
    "overbroad_holding",
    "swapped_party_or_outcome",
    "none",
)
FLAGGED_TYPES = tuple(p for p in PERTURBATIONS if p != "none")


class CitedAuthority(BaseModel):
    kind: Literal["case", "statute"]
    citation: str = Field(description="Citation as written in the claim, e.g. '2023 INSC 650' or 'Section 103 of the BNS'.")
    act: str | None = None  # statute items: IPC / BNS
    provision: str | None = None


class BenchItem(BaseModel):
    id: str
    split: Literal["dev", "test"]
    perturbation_type: PerturbationType
    should_flag: bool = Field(description="True if a correct auditor must flag this citation.")
    claim_text: str
    cited_authority: CitedAuthority
    cited_para: int | None = Field(description="Paragraph number of the cited case that the claim points to.")
    source_case_id: str
    source_para: int
    supporting_quote: str = Field(description="Quote from the source paragraph that supports the ORIGINAL proposition.")
    original_claim: str = Field(description="The unperturbed proposition this item was derived from.")
    judgment_date: str = Field(description="DD-MM-YYYY date of the source judgment (an upper bound for offence dates).")
    meta: dict[str, str] = Field(default_factory=dict)
