"""CLI for the Indian-law adversarial citation benchmark.

    python -m scripts.legal_india_bench pilot          # 10-item pilot, extrapolates the cost
    python -m scripts.legal_india_bench generate       # builds experiments/legal_india/bench_v0.jsonl
    python -m scripts.legal_india_bench run            # runs baselines B0/B1/B2 (+ M1 rule) and writes results
    python -m scripts.legal_india_bench spot -n 20     # prints random items for hand checking
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import BACKEND_DIR, PipelineConfig, get_settings
from app.core.llm import create_chat_model
from app.core.usage import UsageTracker
from app.legal_india.benchmark import baselines
from app.legal_india.benchmark.corpus import load_corpus
from app.legal_india.benchmark.generate import generate_items
from app.legal_india.benchmark.metrics import evaluate
from app.legal_india.benchmark.schema import BenchItem
from app.verification.nli import HFEntailmentModel

OUT = BACKEND_DIR.parent / "experiments" / "legal_india"
CORPUS = OUT / "sample_corpus.jsonl"
BENCH = OUT / "bench_v0.jsonl"
MANIFEST = OUT / "bench_v0_manifest.json"
RESULTS = OUT / "results_baselines.json"
DECISIONS = OUT / "decisions_baselines.json"
COST_CAP_USD = 2.0


def load_items(path: Path = BENCH) -> list[BenchItem]:
    return [BenchItem.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_items(items: list[BenchItem], path: Path = BENCH) -> None:
    path.write_text("\n".join(i.model_dump_json() for i in items) + "\n", encoding="utf-8")


def _cost(usage: UsageTracker) -> dict:
    s = usage.summary()
    return {"input_tokens": s.input_tokens, "output_tokens": s.output_tokens, "cost_usd": s.cost_usd}


async def cmd_generate(args: argparse.Namespace) -> None:
    settings = get_settings()
    model = create_chat_model(settings)
    usage = UsageTracker(PipelineConfig().pricing)
    corpus = load_corpus(CORPUS)
    kw = dict(n_case_bases=args.case_bases, n_wrong_code=args.wrong_code, n_statute_none=args.statute_none)
    items, stats = await generate_items(corpus, model, settings.chat_model, usage, seed=args.seed, **kw)
    cost = _cost(usage)
    if args.pilot:
        full = 50 + 75 + 5  # approx number of generation LLM calls in the full run (bases + oversampling)
        small = max(1, args.case_bases + args.wrong_code + args.statute_none)
        est = (cost["cost_usd"] or 0) * (full + 50) / (small + args.case_bases)
        print(json.dumps({"pilot_items": len(items), "pilot_cost_usd": cost["cost_usd"], "estimated_full_generation_usd": round(est, 4), "cap_usd": COST_CAP_USD}, indent=1))
        if est > COST_CAP_USD:
            sys.exit("estimated generation cost exceeds the cap; shrink the benchmark")
        return
    write_items(items)
    counts = collections.Counter((i.split, i.perturbation_type) for i in items)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model": settings.chat_model,
        "temperature": settings.llm_temperature,
        "corpus": "experiments/legal_india/sample_corpus.jsonl (199 Supreme Court judgments, 2023-2024)",
        "n_items": len(items),
        "counts_by_split_and_type": {f"{s}/{t}": n for (s, t), n in sorted(counts.items())},
        "generation_stats": stats.__dict__,
        "usage": cost,
        "label_note": "labels are automatic by construction; not validated by lawyers",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


async def cmd_run(args: argparse.Namespace) -> None:
    settings = get_settings()
    model = create_chat_model(settings)
    usage = UsageTracker(PipelineConfig().pricing)
    corpus = load_corpus(CORPUS)
    items = load_items()
    dev = [i for i in items if i.split == "dev"]

    b0 = baselines.check_b0(items, corpus)
    b1 = await baselines.check_b1(items, corpus, model, settings.chat_model, usage)
    nli = HFEntailmentModel(settings.eval_nli_model, device=settings.nli_device)
    b2_scored = await asyncio.to_thread(baselines.score_b2, items, corpus, nli)
    dev_ids = {i.id for i in dev}
    threshold = baselines.tune_threshold([d for d in b2_scored if d.item_id in dev_ids], dev)
    b2 = baselines.apply_threshold(b2_scored, threshold)
    rule = baselines.check_rule_wrong_code(items)

    systems = {"B0_existence_only": b0, "B1_llm_only": b1, "B2_nli_only": b2, "R_wrong_code_rule_m1": rule}
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "experiments/legal_india/bench_v0.jsonl",
        "n_items": len(items),
        "llm": settings.chat_model,
        "nli": settings.eval_nli_model,
        "b2_threshold_tuned_on_dev": threshold,
        "usage_b1": _cost(usage),
        "systems": {name: {"test": evaluate(items, d, split="test"), "dev": evaluate(items, d, split="dev")} for name, d in systems.items()},
        "notes": [
            "Items are LLM-generated from judgments; labels are automatic by construction and not validated by lawyers.",
            "Thresholds are tuned on the dev split only; reported numbers are on the test split.",
            "R is a rule check for wrong_code only, shown for reference; it is not a general baseline.",
        ],
    }
    RESULTS.write_text(json.dumps(results, indent=1), encoding="utf-8")
    DECISIONS.write_text(
        json.dumps({n: [{"id": x.item_id, "flagged": x.flagged, "score": x.score, "reason": x.reason} for x in d] for n, d in systems.items()}),
        encoding="utf-8",
    )
    print(json.dumps({"b2_threshold": threshold, "b1_cost": results["usage_b1"]}, indent=1))


BENCH_V1 = OUT / "bench_v1.jsonl"
MANIFEST_V1 = OUT / "bench_v1_manifest.json"
RESULTS_V1 = OUT / "results_baselines_v1.json"
DECISIONS_V1 = OUT / "decisions_baselines_v1.json"
NLI_CACHE_V1 = OUT / "nli_cache_v1.json"
COST_CAP_V1_USD = 3.0
GEN_MODEL = "gpt-4.1"  # generates the perturbations: a different model from the B1 checker
FILTER_MODEL = "gpt-4.1-mini"  # checks that a near_paragraph candidate does not support the claim
B1_MODEL = "gpt-4o-mini"
B1B_MODEL = "gpt-4.1"


RUN_COST_CAP_USD = 0.10  # hard stop for the baseline run (budget withdrawn by the user)
B1_CACHE = OUT / "decisions_b1_v1_test.json"


def _check_cost(usage: UsageTracker, label: str, cap: float = COST_CAP_V1_USD) -> float:
    cost = _cost(usage)["cost_usd"] or 0.0
    print(f"[{label}] cumulative cost USD {cost:.4f} (cap {cap})")
    if cost > cap:
        sys.exit(f"cost cap exceeded after {label}")
    return cost


async def cmd_generate_v1(args: argparse.Namespace) -> None:
    from app.embeddings.factory import create_embeddings
    from app.legal_india.benchmark.generate_v1 import generate_items_v1

    settings = get_settings()
    usage = UsageTracker(PipelineConfig().pricing)
    corpus = load_corpus(CORPUS)
    items, stats = await generate_items_v1(
        corpus,
        create_chat_model(settings, GEN_MODEL), GEN_MODEL,
        create_chat_model(settings, FILTER_MODEL), FILTER_MODEL,
        create_embeddings(settings), settings.embedding_model,
        usage,
        seed=args.seed, n_case_bases=args.case_bases, n_wrong_code=args.wrong_code, n_statute_none=args.statute_none,
        n_conditional_bases=args.conditional_bases,
    )
    cost = _cost(usage)
    counts = collections.Counter((i.split, i.perturbation_type) for i in items)
    if args.pilot:
        # cost is roughly linear in the number of bases (case + conditional); scale to the full configuration
        pilot_bases = max(1, stats.attempted_bases + args.conditional_bases)
        est = (cost["cost_usd"] or 0) / pilot_bases * (50 + 200)
        print(json.dumps({
            "pilot_bases": pilot_bases, "pilot_items": len(items), "pilot_cost_usd": cost["cost_usd"],
            "estimated_full_generation_usd": round(est, 4), "cap_usd": COST_CAP_V1_USD,
            "counts": {f"{t}": n for (s, t), n in sorted(counts.items())}, "rejected": stats.rejected,
        }, indent=1))
        write_items(items, OUT / "bench_v1_pilot.jsonl")
        if est > COST_CAP_V1_USD / 2:
            sys.exit("estimated generation cost is too high; shrink the benchmark")
        return
    write_items(items, BENCH_V1)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "generator_model": GEN_MODEL,
        "near_paragraph_filter_model": FILTER_MODEL,
        "embedding_model": settings.embedding_model,
        "temperature": settings.llm_temperature,
        "corpus": "experiments/legal_india/sample_corpus.jsonl (199 Supreme Court judgments, 2023-2024)",
        "n_items": len(items),
        "counts_by_split_and_type": {f"{s}/{t}": n for (s, t), n in sorted(counts.items())},
        "generation_stats": {"rejected": stats.rejected, "errors": stats.errors, "attempted_bases": stats.attempted_bases, "bases": stats.base_stats},
        "usage": cost,
        "not_built": "wrong_court_binding: all corpus judgments are Supreme Court judgments, so no High Court or bench-strength mismatch is derivable from metadata",
        "label_note": "labels are automatic by construction (mechanical checks per type); not validated by lawyers",
    }
    MANIFEST_V1.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    _check_cost(usage, "generate_v1")


def _decision_confidence(scores: dict[str, float], decisions: list[baselines.Decision], threshold: float | None = None) -> dict[str, float]:
    """Confidence of each decision: how far the support score is from the decision boundary."""
    boundary = 0.5 if threshold is None else threshold
    span = max(boundary, 1 - boundary)
    return {d.item_id: min(1.0, abs((scores.get(d.item_id) or 0.0) - boundary) / span) for d in decisions}


async def cmd_run_v1(args: argparse.Namespace) -> None:
    from app.legal_india.benchmark.metrics import risk_coverage

    settings = get_settings()
    usage = UsageTracker(PipelineConfig().pricing)
    corpus = load_corpus(CORPUS)
    items = load_items(BENCH_V1)
    dev_ids = {i.id for i in items if i.split == "dev"}
    dev = [i for i in items if i.split == "dev"]

    b0 = baselines.check_b0(items, corpus)
    # B1 (gpt-4o-mini) on the TEST split only, decisions cached on disk so a re-run never pays twice.
    # B1b (gpt-4.1) is SKIPPED: the user withdrew the OpenAI budget for it.
    test_items = [i for i in items if i.split == "test"]
    b1: list[baselines.Decision] | None = None
    if B1_CACHE.exists():
        cached = json.loads(B1_CACHE.read_text(encoding="utf-8"))
        b1 = [baselines.Decision(x["id"], x["flagged"], x["score"], x["reason"]) for x in cached]
        print(f"B1: {len(b1)} decisions reused from {B1_CACHE.name} (no API calls)")
    elif args.allow_paid:
        b1 = await baselines.check_b1_conf(test_items, corpus, create_chat_model(settings, B1_MODEL), B1_MODEL, usage, stage="bench_b1v1")
        B1_CACHE.write_text(json.dumps([{"id": d.item_id, "flagged": d.flagged, "score": d.score, "reason": d.reason} for d in b1]), encoding="utf-8")
        _check_cost(usage, "B1", cap=RUN_COST_CAP_USD)
    else:
        print("B1 (LLM) skipped: no cached decisions and paid calls are not allowed (pass --allow-paid to run it)")

    nli = baselines.CachedNLI(HFEntailmentModel(settings.eval_nli_model, device=settings.nli_device), NLI_CACHE_V1)
    b2_scored = await asyncio.to_thread(baselines.score_b2, items, corpus, nli)
    threshold = baselines.tune_threshold([d for d in b2_scored if d.item_id in dev_ids], dev)
    b2 = baselines.apply_threshold(b2_scored, threshold)
    rule = baselines.check_rule_wrong_code(items)
    print(f"NLI cache: {nli.hits} hits, {nli.misses} computed")

    systems = {"B0_existence_only": b0, "B2_nli_only": b2, "R_wrong_code_rule_m1": rule}
    calibration = {"B2_nli_only": risk_coverage(items, b2, _decision_confidence({d.item_id: d.score for d in b2_scored}, b2, threshold))}
    if b1 is not None:
        systems["B1_gpt4o_mini"] = b1
        calibration["B1_gpt4o_mini"] = risk_coverage(items, b1, _decision_confidence({d.item_id: d.score for d in b1}, b1))
    v0 = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else None
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "experiments/legal_india/bench_v1.jsonl",
        "n_items": len(items),
        "checkers": {"nli": settings.eval_nli_model, **({"B1": B1_MODEL} if b1 is not None else {})},
        "skipped": {
            "B1b_gpt41": "skipped for cost: the OpenAI budget was withdrawn; its decisions were never saved",
            **({} if b1 is not None else {"B1_gpt4o_mini": "not run on v1: the OpenAI project reached its spend limit (HTTP 429)"}),
        },
        "b2_threshold_tuned_on_dev": threshold,
        "usage_checkers": _cost(usage),
        "systems": {name: {"test": evaluate(items, d, split="test"), "dev": evaluate(items, d, split="dev")} for name, d in systems.items()},
        "calibration_risk_coverage_test": calibration,
        "v0_reference_overall_f1_test": (
            {k: v["test"]["overall"]["f1"] for k, v in v0["systems"].items()} if v0 else None
        ),
        "notes": [
            "Items are LLM-generated (gpt-4.1) from judgments; labels are automatic by construction and not validated by lawyers.",
            "Thresholds are tuned on the dev split only; reported numbers are on the test split.",
            "B1/B1b confidences are verbalised by the model and are not calibrated probabilities.",
            "R is a rule check for wrong_code only, shown for reference; it is not a general baseline.",
        ],
    }
    RESULTS_V1.write_text(json.dumps(results, indent=1), encoding="utf-8")
    DECISIONS_V1.write_text(
        json.dumps({n: [{"id": x.item_id, "flagged": x.flagged, "score": x.score, "reason": x.reason} for x in d] for n, d in systems.items()}),
        encoding="utf-8",
    )
    print(json.dumps({"b2_threshold": threshold, "checker_cost": results["usage_checkers"]}, indent=1))


def cmd_spot(args: argparse.Namespace) -> None:
    items = load_items(BENCH_V1 if getattr(args, "v1", False) else BENCH)
    rng = random.Random(args.seed)
    for item in rng.sample(items, min(args.n, len(items))):
        print(f"--- {item.id} [{item.perturbation_type}] should_flag={item.should_flag} cited={item.cited_authority.citation} para={item.cited_para}")
        print("CLAIM:", item.claim_text)
        if item.perturbation_type != "none":
            print("ORIGINAL:", item.original_claim)
        print("QUOTE:", item.supporting_quote)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, default=13)
    g.add_argument("--case-bases", type=int, default=50)
    g.add_argument("--wrong-code", type=int, default=50)
    g.add_argument("--statute-none", type=int, default=25)
    g.add_argument("--pilot", action="store_true")
    sub.add_parser("run")
    sp = sub.add_parser("spot")
    sp.add_argument("-n", type=int, default=20)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--v1", action="store_true", help="sample from bench_v1.jsonl instead of bench_v0.jsonl")
    sub.add_parser("pilot")
    g1 = sub.add_parser("generate-v1")
    g1.add_argument("--seed", type=int, default=13)
    g1.add_argument("--case-bases", type=int, default=50)
    g1.add_argument("--wrong-code", type=int, default=25)
    g1.add_argument("--statute-none", type=int, default=15)
    g1.add_argument("--conditional-bases", type=int, default=200)
    g1.add_argument("--pilot", action="store_true")
    sub.add_parser("pilot-v1")
    r1 = sub.add_parser("run-v1")
    r1.add_argument("--allow-paid", action="store_true", help="also run the B1 LLM checker (OpenAI calls); off by default")
    args = parser.parse_args()
    if args.cmd == "pilot":
        args = parser.parse_args(["generate", "--pilot", "--case-bases", "2", "--wrong-code", "3", "--statute-none", "2"])
    if args.cmd == "pilot-v1":
        args = parser.parse_args(
            ["generate-v1", "--pilot", "--case-bases", "3", "--wrong-code", "2", "--statute-none", "1", "--conditional-bases", "10"]
        )
    if args.cmd == "generate":
        asyncio.run(cmd_generate(args))
    elif args.cmd == "generate-v1":
        asyncio.run(cmd_generate_v1(args))
    elif args.cmd == "run":
        asyncio.run(cmd_run(args))
    elif args.cmd == "run-v1":
        asyncio.run(cmd_run_v1(args))
    else:
        cmd_spot(args)


if __name__ == "__main__":
    main()
