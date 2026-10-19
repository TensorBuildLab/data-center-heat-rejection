"""
fig3_fig4_profiles_maps.py - Figures 3 and 4 of the manuscript (Section 4.1).

Fig. 3  Plume-axis profiles of the 2 m temperature rise (a) and relative-humidity change (b)
        downwind of the reference air-cooled and water-cooled campuses on the reference night.
        The shaded band spans the two along-wind treatments of the area source (release at the
        downwind edge = upper estimate; two Gauss-Legendre strips = lower estimate); the dotted
        line is the merged point-plume lower bound.
Fig. 4  2-D maps of dT (air- and water-cooled) and dRH (water-cooled) at 2 m.

Run from the repo root:  python paper_figures/fig3_fig4_profiles_maps.py   -> figures/
"""
import warnings

import numpy as np
from matplotlib.patches import Rectangle

from reference_case import AIR, GREY, NIGHT, WATER, air_cooled, fig_path, plt, water_cooled   # also puts the repo root on sys.path
from dc_plume import plume

warnings.filterwarnings("ignore")      # condensation / calm-wind notices are expected here
dry, tow = air_cooled(), water_cooled()

# ---------------------------------------------------------------- Fig. 3: profiles
x = np.linspace(310, 5000, 500)        # starts just beyond the air-cooled footprint edge (300 m)
xt = np.linspace(160, 5000, 500)       # water-cooled footprint edge is at 150 m
a = plume(dry, NIGHT, x)                           # edge release (upper near-field estimate)
s = plume(dry, NIGHT, x, alongwind_strips=2)       # along-wind strips (lower estimate)
t = plume(tow, NIGHT, xt)
p = plume(dry, NIGHT, x, regime="point")           # merged point plume, lower bound

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].fill_between(x / 1e3, s["dT"], a["dT"], color=AIR, alpha=.25, lw=0,
                   label="Air-cooled, edge release to along-wind strips")
ax[0].plot(x / 1e3, a["dT"], color=AIR, lw=1.8, label="Air-cooled, plume axis")
ax[0].plot(x / 1e3, p["dT"], color=GREY, lw=1.2, ls=":", label="Air-cooled, merged point plume (lower bound)")
ax[0].plot(xt / 1e3, t["dT"], color=WATER, lw=1.8, label="Water-cooled, plume axis")
ax[0].set(xlabel="Downwind distance from campus centre (km)", ylabel="ΔT at 2 m (K)", ylim=(0, 0.62), xlim=(0, 5))
ax[0].text(0.02, 0.97, "(a)", transform=ax[0].transAxes, va="top", fontweight="bold")
ax[0].legend(frameon=False, fontsize=7, loc="upper right")
ax[1].plot(x / 1e3, 100 * (a["RH"] - NIGHT.RH), color=AIR, lw=1.8, label="Air-cooled")
ax[1].plot(xt / 1e3, 100 * (t["RH"] - NIGHT.RH), color=WATER, lw=1.8, label="Water-cooled")
ax[1].axhline(0, color="k", lw=.6)
ax[1].set(xlabel="Downwind distance from campus centre (km)", ylabel="ΔRH at 2 m (percentage points)", xlim=(0, 5))
ax[1].text(0.95, 0.97, "(b)", transform=ax[1].transAxes, va="top", fontweight="bold")
ax[1].legend(frameon=False, fontsize=7, loc="center right")
fig.tight_layout()
fig.savefig(fig_path("fig3_profiles.png"), dpi=200)

# ---------------------------------------------------------------- Fig. 4: 2-D maps
X = np.linspace(-500, 5000, 331)[:, None]          # along-wind, m (broadcast to a grid)
Y = np.linspace(-2000, 2000, 241)[None, :]         # crosswind, m
ma = plume(dry, NIGHT, X, Y, z=2.0)
mt = plume(tow, NIGHT, X, Y, z=2.0)

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
ext = [-0.5, 5, -2, 2]
for i, (f, lab, src) in enumerate([(ma["dT"], "(a) ΔT, air-cooled (K)", dry),
                                   (mt["dT"], "(b) ΔT, water-cooled (K)", tow)]):
    im = ax[i].imshow(f.T, origin="lower", extent=ext, cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")
    cs = ax[i].contour(np.linspace(-0.5, 5, 331), np.linspace(-2, 2, 241), np.nan_to_num(f.T),
                       levels=[0.05, 0.1, 0.2, 0.4], colors="k", linewidths=.5)
    ax[i].clabel(cs, fontsize=6, fmt="%.2f")
    ax[i].add_patch(Rectangle((-src.footprint_x / 2e3, -src.footprint_y / 2e3),
                              src.footprint_x / 1e3, src.footprint_y / 1e3, color="0.6"))
    ax[i].set_title(lab, fontsize=8.5, loc="left")
cb = fig.colorbar(im, ax=ax[:2], shrink=.85, pad=.02)
cb.set_label("ΔT (K)")
rh = 100 * (mt["RH"] - NIGHT.RH)
im2 = ax[2].imshow(rh.T, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=4, aspect="auto")
ax[2].add_patch(Rectangle((-0.15, -0.1), 0.3, 0.2, color="0.6"))
ax[2].set_title("(c) ΔRH, water-cooled (pp)", fontsize=8.5, loc="left")
cb2 = fig.colorbar(im2, ax=ax[2], shrink=.85, pad=.03)
cb2.set_label("ΔRH (pp)")
for a_ in ax:
    a_.set_xlabel("Along-wind x (km)")
ax[0].set_ylabel("Crosswind y (km)")
fig.savefig(fig_path("fig4_maps.png"), dpi=200, bbox_inches="tight")
print("wrote", fig_path("fig3_profiles.png"), "and", fig_path("fig4_maps.png"))
