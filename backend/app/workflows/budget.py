"""Per-query cost ceiling: estimates what an optional step would cost before running it.

Estimates use tiktoken token counts and the configured price table, so no API call is needed to decide.
A model without a price entry is treated as free for the estimate (the ceiling only protects priced calls).
"""

from __future__ import annotations

from app.core.config import PipelineConfig
from app.core.usage import UsageTracker, estimate_tokens
from app.verification.claim_extractor import EXTRACTION_PROMPT

REGENERATION_NOTES_TOKENS = 150  # revision feedback appended to the prompt on a retry


def _cost(config: PipelineConfig, model: str, input_tokens: int, output_tokens: int) -> float:
    price = config.pricing.get(model)
    if price is None:
        return 0.0
    return (input_tokens * price.input_per_million + output_tokens * price.output_per_million) / 1e6


def ceiling(config: PipelineConfig) -> float | None:
    return config.budget.max_cost_usd if config.budget is not None else None


def verification_cost(config: PipelineConfig, model: str, answer: str) -> float:
    """Claim extraction: the extraction prompt plus the answer in, a list of claims out (free without LLM claims)."""
    if not config.verification.llm_claims:
        return 0.0
    answer_tokens = estimate_tokens(answer)
    input_tokens = estimate_tokens(EXTRACTION_PROMPT) + answer_tokens + 40
    output_tokens = min(config.verification.max_tokens, int(answer_tokens * 1.2) + 40)
    return _cost(config, model, input_tokens, output_tokens)


def regeneration_cost(config: PipelineConfig, model: str, usage: UsageTracker, answer: str) -> float:
    """A retry: another generation (previous prompt + answer + notes in, full answer budget out) and its verification."""
    last_input = next((r.input_tokens for r in reversed(usage.records) if r.stage == "generation"), 0)
    generation = _cost(
        config, model, last_input + estimate_tokens(answer) + REGENERATION_NOTES_TOKENS, config.generation.max_answer_tokens
    )
    return generation + verification_cost(config, model, answer)


def would_exceed(config: PipelineConfig, usage: UsageTracker, next_step_cost: float) -> bool:
    limit = ceiling(config)
    return limit is not None and usage.spent_usd() + next_step_cost > limit
