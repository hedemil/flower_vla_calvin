#!/usr/bin/env python3
"""Generate the rollout-protocol figure (figures/roll-out-protocol.png).

Corrected protocol: at fine-tuning/evaluation the action window H and multistep
are both 10, so the policy executes the *entire* predicted 10-action chunk
open-loop and re-queries at t+10 (no actions are discarded). The lower panel
shows the per-query DiT wall-time: RF runs N_sampling=4 passes, MF/iMF a single
pass.

Run:  python scripts/figures/rollout_protocol.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

BLUE = "#1f4ec2"
GREEN = "#1b7a34"
RED = "#b3261e"
ORANGE = "#e8a33d"
GRAY = "#7a7a7a"

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")

# ---- geometry of the execution timeline -------------------------------------
t0 = 4.0          # x of step t
dx = 0.62         # x per environment step
ty = 4.7          # timeline y
def sx(i):        # x of step t+i
    return t0 + i * dx

# ---- track labels (left) ----------------------------------------------------
for y, label in [(9.2, "Track 1: Observations"),
                 (7.2, "Track 2: Model Inference"),
                 (ty, "Track 3: Execution")]:
    ax.text(0.2, y, label, style="italic", color=GRAY, fontsize=12, va="center")

# ---- helper: stacked camera-frame glyph -------------------------------------
def obs_glyph(cx, cy):
    for off in (0.06, -0.02):
        ax.add_patch(Rectangle((cx - 0.18 + off, cy - 0.16 - off), 0.36, 0.34,
                               fill=True, facecolor="white", edgecolor="black",
                               lw=1.2, zorder=3))

# ---- helper: inference box --------------------------------------------------
def inf_box(cx):
    box = FancyBboxPatch((cx - 1.15, 6.55), 2.3, 1.25,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         facecolor="#ececf6", edgecolor=BLUE, lw=1.4, zorder=2)
    ax.add_patch(box)
    ax.text(cx, 7.45, "inference_wrapper", ha="center", va="center",
            family="monospace", fontsize=10.5)
    ax.text(cx, 7.05, r"VLM $\rightarrow$ DiT $\rightarrow$ head", ha="center",
            va="center", fontsize=10.5)
    ax.text(cx, 6.72, r"($N_\mathrm{sampling}$ steps)", ha="center", va="center",
            fontsize=10)

# query 1 above step t, query 2 above step t+10 (the re-query point)
q1x, q2x = sx(0) + 0.4, sx(10)

for cx, lab in [(q1x, r"$o_t$ (cameras + language)"), (q2x, r"$o_{t+10}$")]:
    obs_glyph(cx, 9.2)
    ax.text(cx + 0.3, 9.2, lab, ha="left", va="center", fontsize=11)
    inf_box(cx)
    # observation -> inference box
    ax.add_patch(FancyArrowPatch((cx, 8.95), (cx, 7.85), color=GRAY, lw=1.3,
                                 arrowstyle="-|>", mutation_scale=12, zorder=1))

# ---- timeline ---------------------------------------------------------------
x_end = 11.4
ax.add_patch(FancyArrowPatch((sx(0) - 0.4, ty), (x_end, ty), color="black",
                             lw=1.6, arrowstyle="-|>", mutation_scale=16, zorder=1))
ax.text(x_end + 0.05, ty + 0.42, "Env steps", ha="left", va="center", fontsize=10,
        fontweight="bold")
for i in range(0, 11):
    ax.plot([sx(i), sx(i)], [ty - 0.12, ty + 0.12], color="black", lw=1.3)
    lab = "$t$" if i == 0 else (r"$t+%d$" % i)
    ax.text(sx(i), ty + 0.32, lab, ha="center", va="bottom", fontsize=9.5)

# ---- action-chunk arrows (inference box -> timeline start) ------------------
ax.add_patch(FancyArrowPatch((q1x, 6.5), (sx(0), ty + 0.18), color=BLUE, lw=1.5,
                             arrowstyle="-|>", mutation_scale=13,
                             connectionstyle="arc3,rad=0.25", zorder=2))
ax.text((q1x + sx(0)) / 2 + 0.1, 5.95, r"action chunk $a_{t:t+9}$ (length 10)",
        ha="center", va="center", color=BLUE, fontsize=10.5)
ax.add_patch(FancyArrowPatch((q2x, 6.5), (sx(10), ty + 0.18), color=BLUE, lw=1.5,
                             arrowstyle="-|>", mutation_scale=13,
                             connectionstyle="arc3,rad=0.25", zorder=2))
ax.text(sx(10) + 0.15, 5.95, r"$a_{t+10:t+19}$", ha="left", va="center",
        color=BLUE, fontsize=10.5)

# ---- "executed open-loop (all 10 actions)" bracket under t..t+9 -------------
by = ty - 0.55
ax.plot([sx(0), sx(9)], [by, by], color=GREEN, lw=2.0)
for xx in (sx(0), sx(9)):
    ax.plot([xx, xx], [by, by + 0.13], color=GREEN, lw=2.0)
ax.text((sx(0) + sx(9)) / 2, by - 0.4, "executed open-loop (all 10 actions)",
        ha="center", va="center", color=GREEN, fontsize=11, fontweight="bold")

# ---- termination branch at t+10 --------------------------------------------
succ = (14.4, 5.9)
tout = (14.4, 3.5)
ax.add_patch(FancyArrowPatch((x_end - 0.1, ty + 0.1), (succ[0] - 1.05, succ[1]),
                             color="black", lw=1.3, ls="--", arrowstyle="-|>",
                             mutation_scale=13,
                             connectionstyle="arc3,rad=0.18", zorder=1))
ax.add_patch(FancyArrowPatch((x_end - 0.1, ty - 0.1), (tout[0] - 1.05, tout[1]),
                             color="black", lw=1.3, ls="--", arrowstyle="-|>",
                             mutation_scale=13,
                             connectionstyle="arc3,rad=-0.18", zorder=1))
ax.text(13.05, ty, "OR", ha="center", va="center", fontsize=12, fontweight="bold")

def term_box(xy, text, color):
    b = FancyBboxPatch((xy[0] - 1.0, xy[1] - 0.3), 2.0, 0.6,
                       boxstyle="round,pad=0.02,rounding_size=0.1",
                       facecolor="white", edgecolor=color, lw=1.6, zorder=3)
    ax.add_patch(b)
    ax.text(xy[0], xy[1], text, ha="center", va="center", color=color,
            fontsize=10.5)
term_box(succ, "Success (done)", GREEN)
term_box(tout, r"Timeout ($T_\mathrm{max}=520$)", RED)

# ---- lower panel: wall-time comparison -------------------------------------
ax.plot([0.4, 15.6], [2.55, 2.55], color=GRAY, lw=1.0, ls="--")
ax.text(2.2, 2.05, r"Inference wall-time per query ($N_\mathrm{sampling}$)",
        ha="left", va="center", fontsize=12, fontweight="bold")

def pass_row(y, label, n, color, tail):
    ax.text(2.2, y, label, ha="left", va="center", fontsize=11)
    x0 = 6.6
    for k in range(n):
        ax.add_patch(Rectangle((x0 + k * 0.46, y - 0.18), 0.4, 0.36,
                               facecolor=color, edgecolor="black", lw=1.0))
    ax.text(x0 + n * 0.46 + 0.2, y, tail, ha="left", va="center", fontsize=11)

pass_row(1.35, r"RF baseline ($N{=}4$)", 4, ORANGE, r"$= 4\times$ DiT passes")
pass_row(0.6, r"MF / iMF ($N{=}1$)", 1, GREEN, r"$= 1\times$ DiT pass")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = "docs/thesis_final/figures/roll-out-protocol.png"
fig.savefig(out, dpi=200)
print("wrote", out)
