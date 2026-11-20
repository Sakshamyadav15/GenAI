"""Generates every figure of the paper from the repository's result files (see data.py).

Run:  python figures.py      (from paper/; writes figures/*.png at 300 dpi and figures/*.pdf)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import data

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#7f7f7f",
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

TYPE_LABEL = {
    "none": "control",
    "fabricated_case": "fabricated\ncase",
    "wrong_case": "wrong\ncase",
    "wrong_paragraph": "wrong\nparagraph",
    "near_paragraph": "near\nparagraph",
    "overbroad_holding": "overbroad\nholding",
    "swapped_party_or_outcome": "swapped party\nor outcome",
    "altered_quote": "altered\nquote",
    "wrong_code": "wrong\ncode",
}
NEAR_MISS = {"near_paragraph", "overbroad_holding", "swapped_party_or_outcome"}


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def err(m):
    """asymmetric error bars from {value, ci95}"""
    lo, hi = m["ci95"]
    return [[max(m["value"] - lo, 0)], [max(hi - m["value"], 0)]]


# ---------------------------------------------------------------------------------------------- fig 1
def fig_pipeline():
    from fig1 import fig_pipeline as _draw
    _draw(save, BLUE, VERM)


# ---------------------------------------------------------------------------------------------- fig 2
def fig_taxonomy():
    order = ["none", "fabricated_case", "wrong_case", "wrong_paragraph", "near_paragraph", "overbroad_holding",
             "swapped_party_or_outcome", "altered_quote", "wrong_code"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, man, title in [(axes[0], data.v0_manifest, f"v0 ({data.v0_manifest['n_items']} items)"),
                           (axes[1], data.v1_manifest, f"v1 ({data.v1_manifest['n_items']} items)")]:
        c = man["counts_by_split_and_type"]
        dev = [c.get(f"dev/{t}", 0) for t in order]
        test = [c.get(f"test/{t}", 0) for t in order]
        y = np.arange(len(order))
        colors = [VERM if t in NEAR_MISS else (GREY if t == "none" else BLUE) for t in order]
        ax.barh(y, test, color=colors, edgecolor="white", lw=0.5, label="test")
        ax.barh(y, dev, left=test, color=colors, alpha=0.45, edgecolor="white", lw=0.5, label="dev")
        for yi, (a, b) in enumerate(zip(test, dev)):
            if a + b:
                ax.text(a + b + 2, yi, f"{a}+{b}", va="center", fontsize=6.8)
        ax.set_title(title)
        ax.set_xlabel("items (test + dev)")
        ax.set_yticks(y)
        ax.set_yticklabels([TYPE_LABEL[t].replace("\n", " ") for t in order])
        ax.invert_yaxis()
        ax.set_xlim(0, max(t + d for t, d in zip(test, dev)) * 1.25)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=GREY, label="faithful control"), Patch(color=BLUE, label="surface-level error"),
                        Patch(color=VERM, label="near-miss error (v1 only)"),
                        Patch(color="#999999", alpha=0.45, label="lighter shade = dev split")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig2_taxonomy")


# ---------------------------------------------------------------------------------------------- fig 3
def fig_f1_v0_v1():
    systems = [("B0_existence_only", "B0\nexistence only"), ("B1_llm_only", "B1\nLLM only\n(gpt-4o-mini)"),
               ("B2_nli_only", "B2\nlocal NLI"), ("R_wrong_code_rule_m1", "R\nIPC/BNS rule")]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(systems))
    w = 0.36
    for i, (key, _) in enumerate(systems):
        m0 = data.v0["systems"][key]["test"]["overall"]["f1"]
        ax.bar(x[i] - w / 2, m0["value"], w, color=SKY, edgecolor="#333", lw=0.5,
               yerr=err(m0), capsize=2, error_kw={"lw": 0.7},
               label="v0 (n=%d)" % data.v0["systems"][key]["test"]["overall"]["n"] if i == 0 else None)
        if key in data.v1["systems"]:
            m1 = data.v1["systems"][key]["test"]["overall"]["f1"]
            ax.bar(x[i] + w / 2, m1["value"], w, color=VERM, edgecolor="#333", lw=0.5,
                   yerr=err(m1), capsize=2, error_kw={"lw": 0.7},
                   label="v1 (n=%d)" % data.v1["systems"][key]["test"]["overall"]["n"] if i == 0 else None)
            ax.text(x[i] + w / 2, m1["ci95"][1] + 0.02, f"{m1['value']:.2f}", ha="center", fontsize=6.5)
        else:
            ax.bar(x[i] + w / 2, 0.04, w, color="white", edgecolor=GREY, lw=0.7, hatch="////")
            ax.text(x[i] + w / 2, 0.08, "not run\non v1", ha="center", fontsize=6.2, color=GREY)
        ax.text(x[i] - w / 2, m0["ci95"][1] + 0.02, f"{m0['value']:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in systems], fontsize=6.8)
    ax.set_ylabel("F1 (flag = positive), test split")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8, bbox_to_anchor=(1.0, 1.05))
    save(fig, "fig3_f1_v0_v1")


# ---------------------------------------------------------------------------------------------- fig 4
def fig_recall_by_type():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [5, 7]})
    specs = [
        (axes[0], data.v0, ["fabricated_case", "wrong_case", "wrong_paragraph", "altered_quote", "wrong_code"],
         [("B0_existence_only", "B0 existence", GREY), ("B1_llm_only", "B1 LLM-only", ORANGE),
          ("B2_nli_only", "B2 local NLI", BLUE), ("R_wrong_code_rule_m1", "R date rule", GREEN)], "v0 (test)"),
        (axes[1], data.v1, ["fabricated_case", "wrong_paragraph", "near_paragraph", "altered_quote",
                            "swapped_party_or_outcome", "overbroad_holding", "wrong_code"],
         [("B0_existence_only", "B0 existence", GREY), ("B2_nli_only", "B2 local NLI", BLUE),
          ("R_wrong_code_rule_m1", "R date rule", GREEN)], "v1 (test)"),
    ]
    for ax, res, types, sysl, title in specs:
        n = len(sysl)
        w = 0.8 / n
        x = np.arange(len(types))
        for i, (key, label, col) in enumerate(sysl):
            vals, lo, hi = [], [], []
            for t in types:
                m = res["systems"][key]["test"]["per_type"][t]["recall"]
                vals.append(m["value"])
                lo.append(max(m["value"] - m["ci95"][0], 0))
                hi.append(max(m["ci95"][1] - m["value"], 0))
            ax.bar(x - 0.4 + w * (i + 0.5), vals, w, color=col, edgecolor="white", lw=0.4, label=label,
                   yerr=[lo, hi], capsize=1.5, error_kw={"lw": 0.6})
        ax.set_xticks(x)
        ax.set_xticklabels([TYPE_LABEL[t] for t in types], fontsize=6.3)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("recall")
        ax.set_title(title)
        for t_i, t in enumerate(types):
            if t in NEAR_MISS:
                ax.get_xticklabels()[t_i].set_color(VERM)
    axes[1].set_ylabel("")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for l, c in [("B0 existence check", GREY), ("B1 LLM-only (v0 only)", ORANGE),
                                                             ("B2 local NLI", BLUE), ("R IPC/BNS date rule", GREEN)]],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig4_recall_by_type")


# ---------------------------------------------------------------------------------------------- fig 5
def fig_false_flags():
    systems = [("B0_existence_only", "B0"), ("B1_llm_only", "B1"), ("B2_nli_only", "B2"),
               ("R_wrong_code_rule_m1", "R")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(systems))
    w = 0.36
    for i, (key, _) in enumerate(systems):
        m0 = data.v0["systems"][key]["test"]["overall"]["false_flag_rate"]
        ax.bar(x[i] - w / 2, m0["value"], w, color=SKY, edgecolor="#333", lw=0.5, yerr=err(m0), capsize=2,
               error_kw={"lw": 0.7}, label="v0" if i == 0 else None)
        ax.text(x[i] - w / 2, m0["ci95"][1] + 0.015, f"{m0['value']:.2f}", ha="center", fontsize=6.5)
        if key in data.v1["systems"]:
            m1 = data.v1["systems"][key]["test"]["overall"]["false_flag_rate"]
            ax.bar(x[i] + w / 2, m1["value"], w, color=VERM, edgecolor="#333", lw=0.5, yerr=err(m1), capsize=2,
                   error_kw={"lw": 0.7}, label="v1" if i == 0 else None)
            ax.text(x[i] + w / 2, m1["ci95"][1] + 0.015, f"{m1['value']:.2f}", ha="center", fontsize=6.5)
        else:
            ax.text(x[i] + w / 2, 0.02, "not run", ha="center", fontsize=6, color=GREY, rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in systems])
    ax.set_ylabel("false-flag rate on faithful controls")
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig5_false_flags")


# ---------------------------------------------------------------------------------------------- fig 6
def fig_risk_coverage():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for ver, col, lab in [("v0", SKY, "v0"), ("v1", VERM, "v1")]:
        cov, acc, overall, aurc, thr = data.risk_coverage(ver)
        ax.plot(cov, acc, color=col, lw=1.4, label=f"{lab} (threshold {thr}, AURC {aurc:.3f})")
        ax.axhline(overall, color=col, lw=0.6, ls=":")
    ax.set_xlabel("coverage (share of test items answered, most confident first)")
    ax.set_ylabel("accuracy on answered items")
    ax.set_ylim(0.7, 1.02)
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False, loc="lower left", fontsize=6.8)
    save(fig, "fig6_risk_coverage")


# ---------------------------------------------------------------------------------------------- fig 7
def fig_musique_main():
    V = data.musique["variants"]
    names = [("B", "B"), ("A", "A"), ("A-", "A-"), ("V", "V"), ("A+V", "A+V")]
    cols = [GREY, BLUE, SKY, GREEN, VERM]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.25, 1.15, 1.0]})
    ax = axes[0]
    for i, (k, _) in enumerate(names):
        m = V[k]["overall"]["correct"]
        ax.bar(i, m["mean"], color=cols[i], edgecolor="#333", lw=0.5,
               yerr=[[m["mean"] - m["lo"]], [m["hi"] - m["mean"]]], capsize=2, error_kw={"lw": 0.7})
        ax.text(i, m["hi"] + 0.02, f"{m['mean']:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[1] for n in names], fontsize=6.2)
    ax.set_ylabel("answer correct (LLM judge)")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) answer correctness, n=36")

    ax = axes[1]
    groups = ["simple", "moderate", "complex"]
    x = np.arange(3)
    sel = [("B", "B (fixed 5)", GREY), ("A", "A", BLUE), ("A+V", "A+V", VERM)]
    w = 0.26
    for i, (k, lab, col) in enumerate(sel):
        vals = [V[k]["by_group"][g]["num_chunks"]["mean"] for g in groups]
        ax.bar(x - w + i * w, vals, w, color=col, edgecolor="white", lw=0.4, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("chunks given to the generator")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("(b) retrieval depth by question group")

    ax = axes[2]
    for i, (k, _) in enumerate(names):
        m = V[k]["overall"]["total_latency_ms"]
        ax.bar(i, m["mean"] / 1000, color=cols[i], edgecolor="#333", lw=0.5,
               yerr=[[(m["mean"] - m["lo"]) / 1000], [(m["hi"] - m["mean"]) / 1000]], capsize=2, error_kw={"lw": 0.7})
        ax.text(i, m["hi"] / 1000 * 1.05 + 0.3, f"{m['mean'] / 1000:.1f}", ha="center", fontsize=6.3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[0] for n in names], fontsize=6.5)
    ax.set_ylabel("total latency (s, log scale)")
    ax.set_ylim(1, 80)
    ax.set_title("(c) latency (relative only)")
    fig.tight_layout()
    save(fig, "fig7_musique_main")


# ---------------------------------------------------------------------------------------------- fig 8
def fig_musique_sweeps():
    V = data.musique["variants"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ks = [("K1", 1), ("K3", 3), ("K5", 5), ("K8", 8), ("K10", 10)]
    ax = axes[0]
    ax.plot([k for _, k in ks], [V[n]["overall"]["correct"]["mean"] for n, _ in ks], "-o", color=BLUE, ms=3.5,
            lw=1.2, label="answer correct")
    ax.plot([k for _, k in ks], [V[n]["overall"]["r_at_k"]["mean"] for n, _ in ks], "-s", color=ORANGE, ms=3.5,
            lw=1.2, label="recall@k")
    ax.set_xlabel("fixed retrieval depth k")
    ax.set_ylabel("mean over 36 questions")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(a) fixed-k sweep")

    ax = axes[1]
    groups = ["simple", "moderate", "complex"]
    for g, col in zip(groups, [GREEN, BLUE, VERM]):
        ax.plot([k for _, k in ks], [V[n]["by_group"][g]["correct"]["mean"] for n, _ in ks], "-o", ms=3, lw=1.1,
                color=col, label=g)
    ax.set_xlabel("fixed retrieval depth k")
    ax.set_ylabel("answer correct (12 per group)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(b) by question group")

    ax = axes[2]
    for i, (k, lab) in enumerate([("R0", "0"), ("R1", "1"), ("R2", "2")]):
        m = V[k]["overall"]["correct"]
        ax.bar(i, m["mean"], color=VERM, edgecolor="#333", lw=0.5,
               yerr=[[m["mean"] - m["lo"]], [m["hi"] - m["mean"]]], capsize=2, error_kw={"lw": 0.7})
        att = V[k]["overall"]["num_attempts"]["mean"]
        ax.text(i, m["hi"] + 0.02, f"{att:.2f} attempts", ha="center", fontsize=6.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["0", "1", "2"])
    ax.set_xlabel("max verification retries (A+V, 18 questions)")
    ax.set_ylabel("answer correct")
    ax.set_ylim(0, 1)
    ax.set_title("(c) retry sweep")
    fig.tight_layout()
    save(fig, "fig8_musique_sweeps")


# ---------------------------------------------------------------------------------------------- fig 9
def fig_classifier():
    cm = data.musique["classifier"]["confusion_true_by_predicted"]
    groups = ["simple", "moderate", "complex"]
    mat = np.array([[cm[t][p] for p in groups] for t in groups])
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, mat[i, j], ha="center", va="center", color="white" if mat[i, j] > 6 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups)
    ax.set_yticks(range(3))
    ax.set_yticklabels(groups)
    ax.set_xlabel("predicted complexity label")
    ax.set_ylabel("hop-count group")
    acc = data.musique["classifier"]["accuracy"]
    ax.set_title(f"agreement {acc:.3f} (n={data.musique['classifier']['n']})")
    for s in ax.spines.values():
        s.set_visible(True)
    save(fig, "fig9_classifier")


if __name__ == "__main__":
    fig_pipeline()
    fig_taxonomy()
    fig_f1_v0_v1()
    fig_recall_by_type()
    fig_false_flags()
    fig_risk_coverage()
    fig_musique_main()
    fig_musique_sweeps()
    fig_classifier()
