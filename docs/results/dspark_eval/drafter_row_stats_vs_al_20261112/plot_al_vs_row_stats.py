"""Paper figure: mean AL (six neutral benchmarks) vs training-data row statistics (a-c) and supervised tokens seen (d), one point per method.
Reads method_summary.csv from this folder (rows with used_in_figure == True); writes al_vs_row_stats_paper.pdf / .png next to it.  python plot_al_vs_row_stats.py"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in csv.DictReader(open(f"{HERE}/method_summary.csv")) if r["used_in_figure"] == "True"]
# label -> (mean AL, prompt set, replies, supervised share %, supervised tokens/row, total tokens/row, supervised tokens seen (B))
M = {r["short_label"]: (float(r["mean_AL_six_neutral"]), r["prompt_set"], r["reply_source"], 100 * float(r["supervised_share"]), float(r["supervised_per_row_mean"]), float(r["tokens_per_row_mean"]), float(r["supervised_seen_B"])) for r in rows}
INK, GRID, SPINE = "#111111", "#e6e5e1", "#222222"
COL = {"Datakit-SFT": "#2a78d6", "BigMix": "#eb6834", "Datakit-SFT w/o finance": "#1baf7a"}
MK = {"original": "o", "targetgen": "s"}
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 4.5, "ytick.major.size": 4.5,
                     "xtick.major.width": 1.0, "ytick.major.width": 1.0})
FS_LAB, FS_TICK, FS_ANN, FS_LEG, FS_PAN = 17, 14.5, 14, 14.5, 17
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6), sharey=True); axes = axes.ravel()
panels = [("Supervised share of row tokens (%)", 3, False,
           {"Datakit orig": (9, -17), "Datakit TG": (-9, 14), "BigMix orig": (11, -15), "BigMix TG": (-11, -18), "NoFin orig": (9, 16)}),
          ("Supervised tokens per row", 4, True,
           {"Datakit orig": (-9, 15), "Datakit TG": (-4, -20), "BigMix orig": (11, 15), "BigMix TG": (-11, -18), "NoFin orig": (10, 14)}),
          ("Total tokens per row", 5, True,
           {"Datakit orig": (11, -18), "Datakit TG": (-4, -21), "BigMix orig": (11, 15), "BigMix TG": (-11, -18), "NoFin orig": (-10, 15)}),
          ("Supervised tokens seen (billions)", 6, False,
           {"Datakit orig": (8, -19), "Datakit TG": (11, -17), "BigMix orig": (-10, 16), "BigMix TG": (12, 0), "NoFin orig": (-8, 16)})]
for ax, (xl, idx, log, off) in zip(axes, panels):
    for name, v in M.items():
        ax.scatter(v[idx], v[0], s=170, marker=MK[v[2]], color=COL[v[1]], edgecolor="white", linewidth=1.4, zorder=3)
        dx, dy = off[name]
        ax.annotate(name, (v[idx], v[0]), xytext=(dx, dy), textcoords="offset points", fontsize=FS_ANN, color=INK, ha="left" if dx >= 0 else "right", va="center")
    if log:
        ax.set_xscale("log"); ax.set_xticks([1000, 2000, 4000, 8000] if idx == 4 else [1500, 3000, 6000, 12000])
        ax.set_xticklabels(["1k", "2k", "4k", "8k"] if idx == 4 else ["1.5k", "3k", "6k", "12k"]); ax.minorticks_off()
        ax.set_xlim(700, 12500) if idx == 4 else ax.set_xlim(1100, 14500)
    elif idx == 6:
        ax.set_xlim(1.0, 8.4); ax.set_xticks([2, 4, 6, 8])
    else:
        ax.set_xlim(48, 100); ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_ylim(3.08, 3.54); ax.set_yticks([3.1, 3.2, 3.3, 3.4, 3.5])
    ax.set_xlabel(xl, fontsize=FS_LAB, color=INK, labelpad=6)
    ax.grid(True, color=GRID, linewidth=0.9, zorder=0); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(True); s.set_color(SPINE)          # full box on every subplot
    ax.tick_params(labelsize=FS_TICK, colors=INK, top=True, right=True)
for ax in (axes[0], axes[2]): ax.set_ylabel("Mean AL (six benchmarks)", fontsize=FS_LAB, color=INK, labelpad=8)
for ax, t in zip(axes, ("(a)", "(b)", "(c)", "(d)")): ax.text(0.03, 0.95, t, transform=ax.transAxes, fontsize=FS_PAN, fontweight="bold", color=INK, va="top", ha="left")
h = [Line2D([], [], marker="o", ls="", color=COL[k], markersize=11, label=k) for k in COL] + \
    [Line2D([], [], marker="o", ls="", color="#555555", markersize=10, label="Original replies"), Line2D([], [], marker="s", ls="", color="#555555", markersize=10, label="Targetgen replies")]
fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=FS_LEG, labelcolor=INK, bbox_to_anchor=(0.5, 1.045), columnspacing=1.6, handletextpad=0.4)
fig.tight_layout(w_pad=1.2, h_pad=1.6)
for ext, kw in (("pdf", {}), ("png", {"dpi": 300})): fig.savefig(f"{HERE}/al_vs_row_stats_paper.{ext}", bbox_inches="tight", **kw)
print("ok")
