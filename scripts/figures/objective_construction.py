import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Clean formatting inspired by the MeanFlow paper
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.labelbottom": False,
    "ytick.labelleft": False
})

fig, axes = plt.subplots(3, 1, figsize=(8, 12.5))
plt.subplots_adjust(hspace=0.4)

def draw_base_interpolant(ax, title):
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.3)
    
    # Time axis
    ax.annotate('', xy=(1.05, 0), xytext=(-0.05, 0), arrowprops=dict(arrowstyle="->", color='black', lw=1))
    ax.text(1.08, 0, r'$t$', fontsize=12, va='center')
    ax.plot([0, 0], [-0.02, 0.02], 'k-', lw=1)
    ax.text(0, -0.06, r'$0$', ha='center')
    ax.plot([1, 1], [-0.02, 0.02], 'k-', lw=1)
    ax.text(1, -0.06, r'$1$', ha='center')
    
    # Straight Interpolant line (Slate blue)
    ax.plot([0, 1], [0.2, 1.0], color='#5c5cba', lw=2, alpha=0.6)
    ax.text(-0.03, 0.2, r'$x_1$ (data)', ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(1.03, 1.0, r'$e$ (noise)', ha='left', va='center', fontsize=11, fontweight='bold')

# ==========================================
# Panel (a) — Rectified Flow (RF)
# ==========================================
ax = axes[0]
draw_base_interpolant(ax, r'(a) Rectified Flow (RF)')
t = 0.5
zt = 0.2 + t * 0.8
ax.scatter([t], [zt], color='black', zorder=5, s=30)
ax.text(t - 0.02, zt + 0.06, r'$z_t$', ha='right', fontsize=12)

# Instantaneous velocity arrow (Parallel)
ax.quiver(t, zt, 0.2, 0.16, angles='xy', scale_units='xy', scale=1, color='#d62728', width=0.006, zorder=6)
ax.text(t + 0.1, zt + 0.12, r'$v = e - x_1$', color='#d62728', fontsize=12)

ax.text(0.75, 0.4, r'$t \sim \mathcal{U}(0,1)$' + '\ntarget $= v$', fontsize=11)
ax.text(0.5, -0.15, r'$\mathcal{L}_{\mathrm{RF}} = \| f_\theta(z_t, t) - v \|^2$', 
        ha='center', va='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", ls="--"))


# ==========================================
# Panel (b) — MeanFlow
# ==========================================
ax = axes[1]
draw_base_interpolant(ax, r'(b) MeanFlow')
r, t_val = 0.3, 0.7
zr = 0.2 + r * 0.8
zt_val = 0.2 + t_val * 0.8
ax.scatter([r, t_val], [zr, zt_val], color='black', zorder=5, s=30)
ax.text(r - 0.02, zr + 0.06, r'$z_r$', ha='right', fontsize=12)
ax.text(t_val - 0.02, zt_val + 0.06, r'$z_t$', ha='right', fontsize=12)

# Curved average velocity arrow (Orange/Gold)
kw = dict(arrowstyle="->", color="#e69100", connectionstyle="arc3,rad=-0.2", lw=2)
ax.add_patch(patches.FancyArrowPatch((r, zr), (t_val, zt_val), **kw))
ax.text(0.45, 0.85, r'$u^\star = \frac{1}{t-r}\int_r^t v\,\mathrm{d}\tau$', color='#e69100', ha='center', fontsize=12)

# Fixed point overlay
ax.text(0.45, 0.65, r'$u^\star = v - h\,\partial_t u^\star$', color='#e69100', ha='center', fontsize=11)

# h span bracket
ax.annotate('', xy=(r, -0.05), xytext=(t_val, -0.05), arrowprops=dict(arrowstyle='<->', color='gray'))
ax.text((r+t_val)/2, -0.12, r'$h = t - r$', ha='center', color='gray')

loss_b = (r"$u_{\mathrm{tgt}} = \mathrm{sg}(v - h \cdot \mathrm{JVP}_t f_\theta(z_t, t, h))$" + "\n" +
          r"$\mathcal{L}_{\mathrm{MF}} = \frac{\| f_\theta - u_{\mathrm{tgt}} \|^2}{(\| f_\theta - u_{\mathrm{tgt}} \|^2 + \varepsilon)^p}$")
ax.text(0.4, -0.18, loss_b, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", ls="--"))

# Small side-note annotations
sidenote = (r"$\varepsilon = 10^{-2}, p = 1$" + "\n" + 
            r"$(t,r)$ logit-normal with" + "\n" + 
            r"$P_{\mathrm{mean}}=-2, P_{\mathrm{std}}=2$" + "\n" + 
            r"vel. mode ($r=t$) w.p. $1-\rho$" + "\n" + 
            r"int. mode ($r<t$) w.p. $\rho$")
ax.text(0.75, -0.18, sidenote, ha='left', va='top', fontsize=9, bbox=dict(boxstyle="square,pad=0.4", fc="#f8f9fa", ec="lightgray", lw=1))


# ==========================================
# Panel (c) — iMF
# ==========================================
ax = axes[2]
draw_base_interpolant(ax, r'(c) iMF')
ax.scatter([r, t_val], [zr, zt_val], color='black', zorder=5, s=30)
ax.annotate('', xy=(r, -0.05), xytext=(t_val, -0.05), arrowprops=dict(arrowstyle='<->', color='gray'))
ax.text((r+t_val)/2, -0.12, r'$h = t - r$', ha='center', color='gray')

# u-head branch (Green)
ax.text(0.25, 1.1, r'$u$-head $\rightarrow u$', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#d9f0d3", ec="green", alpha=0.8))
ax.annotate('', xy=(0.25, 1.0), xytext=(t_val, zt_val), arrowprops=dict(arrowstyle="->", color="green", connectionstyle="arc3,rad=0.15", lw=1.5))
ax.text(0.25, 0.85, r'$V = u + h \cdot \mathrm{sg}(\partial_t u)$' + '\n' + r'$\downarrow$' + '\n' + r'target $v$', ha='center', va='top', fontsize=11)

# v-head branch (Orange)
ax.text(0.8, 1.1, r'$v$-head $\rightarrow v_c$ (no grad)', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#fddbc7", ec="orange", alpha=0.8))
ax.annotate('', xy=(0.8, 1.0), xytext=(t_val, zt_val), arrowprops=dict(arrowstyle="->", color="orange", connectionstyle="arc3,rad=-0.15", lw=1.5))
ax.text(0.8, 0.85, r'$v_c$' + '\n' + r'$\downarrow$' + '\n' + r'target $v$', ha='center', va='top', fontsize=11)

# Loss Box
loss_c = r"$\mathcal{L}_{\mathrm{iMF}} = \| V - v \|^2_{\mathrm{adp}} + \| v_c - v \|^2_{\mathrm{adp}}$"
ax.text(0.5, -0.18, loss_c, ha='center', va='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", ls="--"))

# Save as a high-resolution lossless vector PDF for your thesis
plt.savefig("figure_2_objectives_final.pdf", bbox_inches='tight', format='pdf')
print("Generated figure_2_objectives_final.pdf successfully.")