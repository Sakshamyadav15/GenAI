"""Cost ceiling inside the real LangGraph pipeline, and the legal instance's API surface (LLM and NLI faked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import PipelineConfig, Settings
from app.main import create_app
from app.schemas.query import PipelineName
from app.verification.claim_extractor import EXTRACTION_PROMPT
from tests.fakes import HashingEmbeddings, ScriptedChatModel, research_responder
from tests.integration.test_proposed_graph import _pipelines

QUESTION = "Who is the Retrieval and Reasoning Lab led by?"
# Each fake LLM call reports 120 input + 30 output tokens; at USD 1000 per million that is USD 0.15 a call.
PRICED = {"pricing": {"fake-chat": {"input_per_million": 1000, "output_per_million": 1000}}}


def _config(ceiling: float | None, **extra: object) -> PipelineConfig:
    overrides: dict = {
        "proposed": {"adaptive": False, "verify": True},
        "verification": {"use_llm_judge": False, **extra.get("verification", {})},
        **PRICED,
    }
    if ceiling is not None:
        overrides["budget"] = {"max_cost_usd": ceiling}
    return PipelineConfig().with_overrides(overrides)


async def test_verification_is_skipped_when_it_would_cross_the_ceiling(tmp_path: Path) -> None:
    pipelines, chat = _pipelines(tmp_path, research_responder())
    response = await pipelines[PipelineName.PROPOSED].run(QUESTION, "kb", _config(ceiling=0.2))

    assert response.budget is not None and response.budget.skipped_steps == ["claim verification"]
    assert response.budget.ceiling_usd == 0.2
    assert response.verification is None
    assert response.uncertainty is not None and response.uncertainty.flagged
    assert "cost limit" in response.uncertainty.notice
    assert chat.calls_for(EXTRACTION_PROMPT) == []  # the extraction call was never made
    assert response.answer  # the cited answer is still returned
    assert response.variant["verify"] is False


async def test_verification_runs_when_it_fits_and_spend_is_reported(tmp_path: Path) -> None:
    pipelines, chat = _pipelines(tmp_path, research_responder())
    response = await pipelines[PipelineName.PROPOSED].run(QUESTION, "kb", _config(ceiling=5.0, verification={"max_retries": 0}))

    assert response.verification is not None
    assert response.budget is not None and response.budget.skipped_steps == []
    assert 0 < response.budget.spent_usd <= response.budget.ceiling_usd
    assert len(chat.calls_for(EXTRACTION_PROMPT)) == 1


async def test_without_a_ceiling_nothing_changes_and_no_budget_is_reported(tmp_path: Path) -> None:
    pipelines, _ = _pipelines(tmp_path, research_responder())
    response = await pipelines[PipelineName.PROPOSED].run(QUESTION, "kb", _config(ceiling=None))
    assert response.budget is None and response.verification is not None


async def test_retry_is_skipped_when_it_would_cross_the_ceiling(tmp_path: Path) -> None:
    pipelines, chat = _pipelines(tmp_path, research_responder())  # hallucinates once, so verification fails
    config = _config(ceiling=0.6, verification={"max_retries": 2})
    response = await pipelines[PipelineName.PROPOSED].run(QUESTION, "kb", config)

    assert len(response.attempts) == 1 and response.refinements == []
    assert response.budget is not None and "retry" in response.budget.skipped_steps
    assert response.verification is not None and not response.verification.passed
    assert response.uncertainty is not None and response.uncertainty.flagged  # the unsupported claim is still flagged


def _legal_settings(tmp_path: Path, questions: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        log_level="WARNING",
        data_dir=tmp_path / "data",
        experiments_dir=tmp_path / "experiments",
        openai_api_key=None,
        database_url=None,
        preload_nli=False,
        seed_demo_corpus=False,
        demo_cache_dir=tmp_path / "demo_cache",
        demo_questions_path=questions,
        instance_mode="legal_acts",
        legal_budget_mode=True,
        default_corpus="indian_acts",
    )


def test_legal_instance_reports_its_mode_budget_preset_and_own_demo_questions(tmp_path: Path, nli_registry) -> None:
    questions = tmp_path / "legal_questions.json"
    questions.write_text(json.dumps({"questions": [{"category": "simple", "question": "What is Article 21?"}]}), encoding="utf-8")
    app = create_app(
        _legal_settings(tmp_path, questions),
        chat_model=ScriptedChatModel(responder=research_responder()),
        embeddings=HashingEmbeddings(),
        nli=nli_registry,
    )
    with TestClient(app) as client:
        config = client.get("/api/config").json()
        assert config["instance_mode"] == "legal_acts" and config["budget_mode"] is True
        assert config["default_corpus"] == "indian_acts"
        assert config["pipeline"]["budget"] == {"max_cost_usd": 0.001}
        assert config["pipeline"]["baseline"]["top_k"] == 4 and config["pipeline"]["proposed"]["adaptive"] is False
        assert client.get("/api/demo/questions").json()["questions"][0]["question"] == "What is Article 21?"


def test_general_instance_is_unchanged(client: TestClient) -> None:
    config = client.get("/api/config").json()
    assert config["instance_mode"] == "general" and config["budget_mode"] is False
    assert "budget" not in config["pipeline"]
    assert len(client.get("/api/demo/questions").json()["questions"]) >= 5


@pytest.mark.parametrize("preset", ["budget"])
def test_preset_can_be_requested_per_query(client: TestClient, preset: str) -> None:
    upload = client.post(
        "/api/documents?wait=true",
        files={"file": ("labs.md", b"The Retrieval and Reasoning Lab is led by Lena Park. It was founded in 2016.")},
    )
    assert upload.status_code == 201
    response = client.post("/api/query", json={"question": QUESTION, "overrides": {"preset": preset}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["config"]["budget"] == {"max_cost_usd": 0.001}
    assert body["variant"]["adaptive"] is False and body["retrieval"]["k_requested"] == 4
    bad = client.post("/api/query", json={"question": QUESTION, "overrides": {"preset": "unknown"}})
    assert bad.status_code == 422
