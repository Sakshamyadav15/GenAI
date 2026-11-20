"""Figure 1 (pipeline diagram), drawn with matplotlib primitives."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def fig_pipeline(save, BLUE, VERM):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 47)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec="#333333", ls="-", fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.9", fc=fc, ec=ec,
                                    lw=0.8, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)

    def arrow(x1, y1, x2, y2, color="#333333", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                     linestyle=ls, shrinkA=0, shrinkB=0))

    NL = "\n"
    ax.text(0, 45.6, "General pipeline (evaluated on MuSiQue, Sec. VI-D)", fontsize=7.8, fontweight="bold", color=BLUE)
    ya, h = 33, 8
    row_a = [
        (0, 11, "Question"),
        (15, 16, NL.join(["Complexity", "assessment", "(LLM + lexical)"])),
        (35, 16, NL.join(["Depth k and", "sub-questions", "(RRF fusion)"])),
        (55, 16, NL.join(["Dense retrieval", "+ confidence", "check"])),
        (75, 11, NL.join(["Cited", "answer"])),
        (90, 10, NL.join(["Claim", "extraction"])),
    ]
    for x, w, t in row_a:
        box(x, ya, w, h, t, fc="#eef6fc")
    for x1, x2 in [(11, 15), (31, 35), (51, 55), (71, 75), (86, 90)]:
        arrow(x1, ya + h / 2, x2, ya + h / 2)
    ax.text(63, ya + h + 1.4, "low confidence: expand k (bounded)", ha="center", fontsize=6.3, style="italic")

    yb = 19
    box(88, yb, 12, h, NL.join(["Evidence", "verification", "(NLI + judge)"]), fc="#eaf7f1")
    box(70, yb, 14, h, NL.join(["Pass, or retries", "exhausted?"]), fc="#eaf7f1")
    box(48, yb, 18, h, NL.join(["Final answer with", "uncertainty notice"]), fc="#eaf7f1")
    arrow(95, ya, 95, yb + h)
    arrow(88, yb + h / 2, 84, yb + h / 2)
    arrow(70, yb + h / 2, 66, yb + h / 2)
    ax.text(68, yb + h / 2 + 1.3, "yes", ha="center", fontsize=6.3)
    arrow(80.5, yb + h, 80.5, ya, color=VERM, ls="--")
    ax.text(79.2, (yb + h + ya) / 2, NL.join(["no: targeted retrieval,", "regenerate (bounded retries)"]),
            ha="right", va="center", fontsize=6.3, color=VERM)

    ax.text(0, 15.2, "Legal citation checks evaluated as baselines on the Indian benchmark (Sec. VI-A to VI-C)",
            fontsize=7.8, fontweight="bold", color=VERM)
    y3, h3, w3 = 3.0, 8.0, 23.5
    box(0, y3, w3, h3, NL.join(["B0: existence check", "(citation resolves to a case)"]), fc="#fdf1e6")
    box(25.5, y3, w3, h3, NL.join(["B2: local NLI", "(entailment vs. cited paragraph)"]), fc="#fdf1e6")
    box(51, y3, w3, h3, NL.join(["R: IPC/BNS date rule", "(code in force on offence date)"]), fc="#fdf1e6")
    box(76.5, y3, w3, h3, NL.join(["B1: LLM-only checker", "(v0 only; not run on v1)"]), fc="#f2f2f2",
        ls="--", ec="#888888")
    save(fig, "fig1_pipeline")
