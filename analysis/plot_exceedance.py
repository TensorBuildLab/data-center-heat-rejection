"""
plot_exceedance.py - Fig. 7 of the paper: hours per year with dT > 1 K around air-cooled
campuses of 100 MW, 300 MW and 1 GW (IT) and a 3 GW cluster, Phoenix TMY3.
Reads the arrays written by analysis/exceedance.py (results/exceedance/hours1_*.npy).

Run from the repo root:  python analysis/plot_exceedance.py   -> figures/fig7_exceedance_maps.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN = os.path.join("results", "exceedance")
SCEN = [("100 MW campus", 100, 316), ("300 MW campus", 300, 548), ("1 GW campus", 1000, 1000),
        ("3 GW cluster", 3000, 3000)]
G = np.arange(-6000, 6001, 100.0) / 1000
X, Y = np.meshgrid(G, G, indexing="xy")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.9), constrained_layout=True)
for a, (name, it, L) in zip(ax, SCEN):
    h = np.load(os.path.join(IN, f"hours1_{it}.npy"))
    inside = (np.abs(X) <= L / 2000) & (np.abs(Y) <= L / 2000)
    hm = np.where(inside | (h < 1), np.nan, h)
    im = a.pcolormesh(G, G, hm, cmap="magma_r", vmin=0, vmax=2400, shading="auto")
    a.add_patch(plt.Rectangle((-L / 2000, -L / 2000), L / 1000, L / 1000, color="0.55", zorder=3))
    hc = np.where(inside, np.nan, h)
    cs = a.contour(G, G, hc, levels=[100, 500], colors=["0.35", "k"], linewidths=[0.6, 1.0])
    a.clabel(cs, fmt=lambda v: f"{v:.0f} h", fontsize=6)
    a.set(title=f"{name}: max {np.nanmax(hc):.0f} h", xlabel="East (km)", aspect="equal",
          xlim=(-6, 6), ylim=(-6, 6))
    a.title.set_fontsize(9)
ax[0].set_ylabel("North (km)")
fig.colorbar(im, ax=ax, shrink=0.85, label="Hours per year with ΔT > 1 K at 2 m")
os.makedirs("figures", exist_ok=True)
out = os.path.join("figures", "fig7_exceedance_maps.png")
fig.savefig(out, dpi=200)
print("wrote", out)
