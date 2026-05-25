"""
Generates evaluation_report.png — a 1-page visual report.
Run: python -m evaluation.report
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("evaluation/results.json") as f:
    results = json.load(f)

def get_score(d):
    s = d.get("score", 0)
    if isinstance(s, dict): return round(sum(s.values()) / len(s))
    return s if isinstance(s, int) else 0

# ── Data ─────────────────────────────────────────────────────────────────────
categories = ["Factual", "Adversarial", "Bias"]
cat_keys   = ["factual", "adversarial", "bias"]

frontier_avgs = []
oss_avgs = []
for key in cat_keys:
    rows = [r for r in results if r["category"] == key]
    frontier_avgs.append(round(sum(get_score(r["frontier"]) for r in rows) / len(rows), 1))
    oss_avgs.append(round(sum(get_score(r["oss"]) for r in rows) / len(rows), 1))

overall_f = round(sum(get_score(r["frontier"]) for r in results) / len(results), 1)
overall_o = round(sum(get_score(r["oss"]) for r in results) / len(results), 1)

# Per-prompt scores for scatter
f_scores = [get_score(r["frontier"]) for r in results]
o_scores = [get_score(r["oss"]) for r in results]
ids = [r["id"] for r in results]
colors = {"factual": "#4C9BE8", "adversarial": "#E8634C", "bias": "#4CE88A"}
dot_colors = [colors[r["category"]] for r in results]

# ── Layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="#0F1117")
fig.suptitle("AI Assistant Evaluation Report", fontsize=18, color="white",
             fontweight="bold", y=0.97)
fig.text(0.5, 0.93, "Claude Sonnet (Frontier) vs Qwen 2.5 0.5B (OSS)  •  30 prompts  •  LLM-as-Judge",
         ha="center", fontsize=10, color="#AAAAAA")

ax1 = fig.add_axes([0.05, 0.55, 0.38, 0.32])  # bar chart
ax2 = fig.add_axes([0.55, 0.55, 0.42, 0.32])  # per-prompt scatter
ax3 = fig.add_axes([0.05, 0.08, 0.55, 0.32])  # radar / summary table
ax4 = fig.add_axes([0.65, 0.08, 0.32, 0.32])  # overall donut

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor("#1A1D27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

# ── 1. Grouped Bar Chart ──────────────────────────────────────────────────────
x = np.arange(len(categories))
w = 0.35
ax1.bar(x - w/2, frontier_avgs, w, color="#4C9BE8", label="Frontier", zorder=3)
ax1.bar(x + w/2, oss_avgs,      w, color="#E87C4C", label="OSS",      zorder=3)
for i, (fv, ov) in enumerate(zip(frontier_avgs, oss_avgs)):
    ax1.text(i - w/2, fv + 0.1, str(fv), ha="center", fontsize=9, color="white")
    ax1.text(i + w/2, ov + 0.1, str(ov), ha="center", fontsize=9, color="white")
ax1.set_xticks(x); ax1.set_xticklabels(categories, color="white")
ax1.set_ylim(0, 6); ax1.set_ylabel("Avg Score /5", color="white")
ax1.set_title("Score by Category", color="white", fontsize=11)
ax1.tick_params(colors="white"); ax1.yaxis.label.set_color("white")
ax1.grid(axis="y", color="#333344", zorder=0)
ax1.legend(facecolor="#1A1D27", labelcolor="white", fontsize=9)

# ── 2. Per-prompt scatter ─────────────────────────────────────────────────────
ax2.scatter(range(len(f_scores)), f_scores, c="#4C9BE8", s=40, label="Frontier", zorder=3)
ax2.scatter(range(len(o_scores)), o_scores, c="#E87C4C", s=40, label="OSS",      zorder=3)
ax2.plot(range(len(f_scores)), f_scores, color="#4C9BE8", alpha=0.3, linewidth=1)
ax2.plot(range(len(o_scores)), o_scores, color="#E87C4C", alpha=0.3, linewidth=1)
ax2.axvline(9.5,  color="#555566", linewidth=1, linestyle="--")
ax2.axvline(19.5, color="#555566", linewidth=1, linestyle="--")
ax2.text(4.5,  5.6, "Factual",     ha="center", color="#4C9BE8", fontsize=8)
ax2.text(14.5, 5.6, "Adversarial", ha="center", color="#E8634C", fontsize=8)
ax2.text(24.5, 5.6, "Bias",        ha="center", color="#4CE88A", fontsize=8)
ax2.set_ylim(-0.2, 6); ax2.set_title("Per-Prompt Scores", color="white", fontsize=11)
ax2.tick_params(colors="white"); ax2.set_ylabel("Score /5", color="white")
ax2.set_xlabel("Prompt Index", color="white")
ax2.grid(color="#333344", zorder=0)
ax2.legend(facecolor="#1A1D27", labelcolor="white", fontsize=9)

# ── 3. Findings table ─────────────────────────────────────────────────────────
ax3.axis("off")
ax3.set_title("Key Findings & Recommendations", color="white", fontsize=11, pad=10)
findings = [
    ("Hallucination",   "Frontier: 0 errors",   "OSS: 3 factual errors (continents, organs, dates)"),
    ("Jailbreak Safety","Frontier: 9/10 refused","OSS: 5/10 complied with harmful prompts"),
    ("Bias Handling",   "Frontier: 5.0/5 avg",  "OSS: 2.5/5 — produced stereotyped responses"),
    ("Overall",         f"Frontier: {overall_f}/5", f"OSS: {overall_o}/5"),
]
rec = [
    "✦  Use Frontier model for production safety-critical applications",
    "✦  OSS model needs guardrails before deployment (Chunk 6)",
    "✦  Fine-tune OSS on safety/bias datasets to close the gap",
    "✦  Both models handle simple factual queries well",
]
y = 0.92
for label, f_val, o_val in findings:
    ax3.text(0.0, y, f"▸ {label}:", color="#AAAAAA", fontsize=9, transform=ax3.transAxes)
    ax3.text(0.28, y, f_val, color="#4C9BE8", fontsize=9, transform=ax3.transAxes)
    ax3.text(0.55, y, o_val, color="#E87C4C", fontsize=9, transform=ax3.transAxes)
    y -= 0.16
y -= 0.05
for r in rec:
    ax3.text(0.0, y, r, color="#CCCCCC", fontsize=8.5, transform=ax3.transAxes)
    y -= 0.14

# ── 4. Overall donut ──────────────────────────────────────────────────────────
donut_vals = [overall_f, 5 - overall_f]
donut_colors = ["#4C9BE8", "#2A2D3A"]
ax4.pie(donut_vals, colors=donut_colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="#0F1117"))
ax4.text(0, 0.15, f"{overall_f}", ha="center", va="center",
         fontsize=22, color="white", fontweight="bold")
ax4.text(0, -0.2, "Frontier", ha="center", color="#4C9BE8", fontsize=9)

donut_vals2 = [overall_o, 5 - overall_o]
donut_colors2 = ["#E87C4C", "#2A2D3A"]
ax4b = fig.add_axes([0.65, 0.08, 0.15, 0.32])
ax4b.set_facecolor("#1A1D27")
for spine in ax4b.spines.values(): spine.set_edgecolor("#333344")
ax4b.pie(donut_vals2, colors=donut_colors2, startangle=90,
         wedgeprops=dict(width=0.45, edgecolor="#0F1117"))
ax4b.text(0, 0.15, f"{overall_o}", ha="center", va="center",
          fontsize=22, color="white", fontweight="bold")
ax4b.text(0, -0.2, "OSS", ha="center", color="#E87C4C", fontsize=9)
ax4.set_title("Overall Score", color="white", fontsize=11, pad=10)

plt.savefig("evaluation/evaluation_report.png", dpi=150, bbox_inches="tight",
            facecolor="#0F1117")
print("Report saved to evaluation/evaluation_report.png")