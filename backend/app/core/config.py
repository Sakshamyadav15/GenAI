"""Configuration.

Two layers are kept deliberately separate:

* ``Settings``: environment-specific values (API keys, model IDs, storage locations), read from
  environment variables or ``.env``. Never committed.
* ``PipelineConfig``: research parameters (chunking, retrieval depth, prices, ...). Defaults live
  here; an optional YAML file, per-request overrides and experiment configs can change them.
  Every query trace and experiment run stores a snapshot of the ``PipelineConfig`` it used.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import InvalidConfigError

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Adaptive RAG Research Prototype"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    # LLM provider for chat: 'auto' uses Gemini when GEMINI_API_KEY is set, otherwise OpenAI.
    llm_provider: Literal["openai", "gemini", "auto"] = "openai"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # 'local' embeds on this machine (sentence-transformers): free, no API key needed for indexing/search.
    embedding_provider: Literal["openai", "local"] = "openai"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_temperature: float = 0.0
    llm_seed: int | None = 42
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    data_dir: Path = BACKEND_DIR / "data"
    database_url: str | None = None
    chroma_dir: Path | None = None
    upload_dir: Path | None = None
    pipeline_config_path: Path | None = None

    default_corpus: str = "default"
    max_upload_mb: int = 25
    cors_origins: list[str] = ["http://localhost:5173"]

    # Evaluation-only models; never used inside the pipelines being evaluated.
    judge_model: str = "gpt-4.1"
    eval_nli_model: str = "cross-encoder/nli-roberta-base"
    nli_device: str = "cpu"
    preload_nli: bool = True
    seed_demo_corpus: bool = True
    experiments_dir: Path = PROJECT_ROOT / "experiments"
    demo_cache_enabled: bool = True
    demo_cache_dir: Path | None = None  # defaults to backend/seed/demo_cache
    demo_questions_path: Path | None = None  # defaults to backend/seed/demo_questions.json

    # Indian-Acts research instance: 'legal_acts' only changes what the UI says; behaviour comes from the
    # settings below. legal_budget_mode applies BUDGET_PRESET as the default pipeline configuration.
    instance_mode: Literal["general", "legal_acts"] = "general"
    legal_budget_mode: bool = False

    @field_validator(
        "demo_questions_path", "demo_cache_dir", "pipeline_config_path", "chroma_dir", "upload_dir", mode="before"
    )
    @classmethod
    def _empty_path_means_unset(cls, value: Any) -> Any:
        # `NAME=` (or `NAME=  # comment`) in a .env file yields "", which would otherwise become the path ".".
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _derive_paths(self) -> Settings:
        if not self.openai_base_url:
            self.openai_base_url = None
        if not self.database_url:
            self.database_url = f"sqlite+aiosqlite:///{(self.data_dir / 'app.db').as_posix()}"
        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma"
        if self.upload_dir is None:
            self.upload_dir = self.data_dir / "uploads"
        if self.demo_cache_dir is None:
            self.demo_cache_dir = BACKEND_DIR / "seed" / "demo_cache"
        return self

    @property
    def embedding_label(self) -> str:
        """Model name reported in traces; the ``local:`` prefix marks it as free of API cost."""
        if self.embedding_provider == "local":
            return f"local:{self.local_embedding_model}"
        return self.embedding_model

    @staticmethod
    def _has(secret: SecretStr | None) -> bool:
        return secret is not None and bool(secret.get_secret_value().strip())

    @property
    def resolved_llm_provider(self) -> Literal["openai", "gemini"]:
        if self.llm_provider == "auto":
            return "gemini" if self._has(self.gemini_api_key) else "openai"
        return self.llm_provider

    @property
    def active_chat_model(self) -> str:
        """Name of the chat model actually used; it appears in traces and keys the demo cache."""
        return self.gemini_model if self.resolved_llm_provider == "gemini" else self.chat_model

    @property
    def llm_configured(self) -> bool:
        key = self.gemini_api_key if self.resolved_llm_provider == "gemini" else self.openai_api_key
        return self._has(key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


class _StrictModel(BaseModel):
    # Unknown keys are rejected so a typo in an experiment config cannot silently fall back to a default.
    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(_StrictModel):
    chunk_size: int = Field(1000, ge=100, description="Maximum characters per chunk.")
    chunk_overlap: int = Field(150, ge=0, description="Characters shared by consecutive chunks.")

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class BaselineConfig(_StrictModel):
    top_k: int = Field(5, ge=1, le=50, description="Fixed retrieval depth of the conventional RAG baseline.")


class GenerationConfig(_StrictModel):
    max_answer_tokens: int = Field(700, ge=50, le=4000)


class ComplexityWeights(_StrictModel):
    """Weights of the complexity score components (normalised by their sum)."""

    information_needs: float = Field(0.35, ge=0)
    reasoning: float = Field(0.25, ge=0)
    multi_source: float = Field(0.15, ge=0)
    lexical: float = Field(0.15, ge=0)
    llm_label: float = Field(0.10, ge=0)

    @model_validator(mode="after")
    def _positive_total(self) -> ComplexityWeights:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one complexity weight must be positive")
        return self


def _default_reasoning_scores() -> dict[str, float]:
    return {"lookup": 0.0, "aggregation": 0.5, "temporal": 0.5, "comparison": 0.6, "causal": 0.7, "bridge": 1.0}


class ComplexityConfig(_StrictModel):
    weights: ComplexityWeights = Field(default_factory=ComplexityWeights)
    reasoning_scores: dict[str, float] = Field(default_factory=_default_reasoning_scores)
    simple_max: float = Field(0.33, gt=0, lt=1, description="Scores below this are 'simple'.")
    moderate_max: float = Field(0.66, gt=0, lt=1, description="Scores below this (and >= simple_max) are 'moderate'.")
    max_tokens: int = Field(500, ge=100)

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> ComplexityConfig:
        if not self.simple_max < self.moderate_max:
            raise ValueError("simple_max must be smaller than moderate_max")
        missing = {"lookup", "aggregation", "temporal", "comparison", "causal", "bridge"} - set(self.reasoning_scores)
        if missing:
            raise ValueError(f"reasoning_scores is missing {sorted(missing)}")
        return self


class KBand(_StrictModel):
    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> KBand:
        if self.min > self.max:
            raise ValueError("k band min must not exceed max")
        return self


class AdaptiveConfig(_StrictModel):
    k_simple: KBand = Field(default_factory=lambda: KBand(min=1, max=3))
    k_moderate: KBand = Field(default_factory=lambda: KBand(min=4, max=7))
    k_complex: KBand = Field(default_factory=lambda: KBand(min=8, max=12))
    k_cap: int = Field(15, ge=1, description="Upper bound on k, including expansions.")
    decompose: bool = Field(True, description="Retrieve separately for each information need and fuse (RRF).")
    max_subqueries: int = Field(4, ge=1)
    rrf_k: int = Field(60, ge=1)
    # Cosine-similarity normalisation for the confidence estimate; calibrated on the dev split.
    score_floor: float = 0.15
    score_ceiling: float = 0.65
    relevance_threshold: float = Field(0.40, description="A query counts as covered if a chunk scores at least this.")
    confidence_weight_top1: float = Field(0.5, ge=0)
    confidence_weight_mean: float = Field(0.2, ge=0)
    confidence_weight_coverage: float = Field(0.3, ge=0)
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    max_expansions: int = Field(1, ge=0, le=5)
    expansion_step: int = Field(3, ge=1)

    @model_validator(mode="after")
    def _check(self) -> AdaptiveConfig:
        if self.score_ceiling <= self.score_floor:
            raise ValueError("score_ceiling must be greater than score_floor")
        if self.k_complex.max > self.k_cap:
            raise ValueError("k_cap must be at least k_complex.max")
        return self

    def band(self, label: str) -> KBand:
        return {"simple": self.k_simple, "moderate": self.k_moderate, "complex": self.k_complex}[label]


class VerificationConfig(_StrictModel):
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    window_sentences: int = Field(3, ge=1, description="Sentences per evidence window tested with NLI.")
    candidate_windows: int = Field(3, ge=1, description="Windows per claim selected by embedding similarity.")
    support_threshold: float = Field(0.75, gt=0, le=1)
    contradiction_threshold: float = Field(0.75, gt=0, le=1)
    partial_threshold: float = Field(0.40, gt=0, le=1)
    use_llm_judge: bool = Field(True, description="Resolve NLI-ambiguous claims with an LLM judge.")
    pass_support_score: float = Field(0.9, ge=0, le=1)
    max_claims: int = Field(12, ge=1)
    max_retries: int = Field(2, ge=0, le=5)
    refine_k: int = Field(3, ge=1, description="Chunks retrieved per unsupported claim during refinement.")
    max_tokens: int = Field(900, ge=100)
    llm_claims: bool = Field(
        True, description="Split the answer into atomic claims with the LLM; False uses one claim per sentence (no LLM call)."
    )

    @model_serializer(mode="wrap")
    def _omit_default_llm_claims(self, handler: Any) -> dict[str, Any]:
        # Keeps the serialised config identical to configs recorded before this option existed.
        data = handler(self)
        if data.get("llm_claims") is True:
            data.pop("llm_claims", None)
        return data

    @model_validator(mode="after")
    def _ordered(self) -> VerificationConfig:
        if self.partial_threshold > self.support_threshold:
            raise ValueError("partial_threshold must not exceed support_threshold")
        return self


class ProposedConfig(_StrictModel):
    """Feature flags of the proposed pipeline; switching them off yields the ablation variants."""

    adaptive: bool = True
    verify: bool = True


class BudgetConfig(_StrictModel):
    """Hard per-query spending ceiling. Optional steps (claim verification, retries) are skipped when the
    estimated cost would cross it, and the skip is recorded in the response."""

    max_cost_usd: float | None = Field(None, gt=0)


class ModelPrice(_StrictModel):
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(0.0, ge=0)


def _default_pricing() -> dict[str, ModelPrice]:
    # USD per 1M tokens as published by OpenAI at the time of writing. Verify against current
    # pricing before reporting costs; models missing from this table are reported with cost=None.
    return {
        "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60),
        "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00),
        "gpt-4.1": ModelPrice(input_per_million=2.00, output_per_million=8.00),
        "gpt-4.1-mini": ModelPrice(input_per_million=0.40, output_per_million=1.60),
        "text-embedding-3-small": ModelPrice(input_per_million=0.02),
        "text-embedding-3-large": ModelPrice(input_per_million=0.13),
    }


class PipelineConfig(_StrictModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    proposed: ProposedConfig = Field(default_factory=ProposedConfig)
    complexity: ComplexityConfig = Field(default_factory=ComplexityConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    budget: BudgetConfig | None = None
    pricing: dict[str, ModelPrice] = Field(default_factory=_default_pricing)

    @model_serializer(mode="wrap")
    def _omit_unset_budget(self, handler: Any) -> dict[str, Any]:
        # Keeps the serialised config identical to configs recorded before ``budget`` existed.
        data = handler(self)
        if data.get("budget") is None:
            data.pop("budget", None)
        return data

    def with_overrides(self, overrides: dict[str, Any] | None) -> PipelineConfig:
        """Return a copy with ``overrides`` deep-merged in, re-validated as a whole.

        ``{"preset": "budget"}`` first applies BUDGET_PRESET; other keys in the same dict then override it.
        """
        if not overrides:
            return self
        overrides = dict(overrides)
        base = self.model_dump()
        preset = overrides.pop("preset", None)
        if preset is not None:
            if preset != "budget":
                raise InvalidConfigError(f"Unknown preset '{preset}'. Available: 'budget'")
            base = _deep_merge(base, BUDGET_PRESET)
        merged = _deep_merge(base, overrides)
        try:
            return PipelineConfig.model_validate(merged)
        except ValidationError as exc:
            raise InvalidConfigError(f"Invalid pipeline overrides: {exc}") from exc


# Low-cost configuration for the Indian-Acts instance: fixed top-4 retrieval, short answers, local NLI
# verification only (no LLM judge, no retries) and a hard USD 0.001 ceiling per question.
FAST_NLI_MODEL = "cross-encoder/nli-roberta-base"  # quick on CPU; used for live questions

BUDGET_PRESET: dict[str, Any] = {
    "proposed": {"adaptive": False, "verify": True},
    "baseline": {"top_k": 4},
    "generation": {"max_answer_tokens": 250},
    "verification": {
        "use_llm_judge": False,
        "max_retries": 0,
        "max_claims": 6,
        "nli_model": FAST_NLI_MODEL,
        "llm_claims": False,  # one claim per sentence: a question then needs a single LLM call
    },
    "budget": {"max_cost_usd": 0.001},
}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return PipelineConfig.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidConfigError(f"Could not load pipeline config from {path}: {exc}") from exc
