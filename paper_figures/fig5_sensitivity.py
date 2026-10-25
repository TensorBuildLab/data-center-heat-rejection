"""
fig5_sensitivity.py - one-at-a-time sensitivity of the air-cooled reference case (Section 4.3,
Fig. 5) and the stability x wind-speed matrix at 1 km (Section 4.2).

Each input or model constant is moved to the low and high end of its plausible range while all
others stay at the reference values; the change in plume-axis dT at 1 and 5 km from the campus
centre is reported. The entrainment constant A lives inside dc_plume.convective_mixing_depth,
so it is varied by temporarily wrapping that function.

Run from the repo root:  python paper_figures/fig5_sensitivity.py   -> table on stdout, figures/fig5_sensitivity.png
"""
import dataclasses
import functools
import warnings

import numpy as np
from matplotlib.patches import Patch

from reference_case import NIGHT, air_cooled, fig_path, plt, water_cooled   # also puts the repo root on sys.path
import dc_plume as D
from dc_plume import Ambient, plume

warnings.filterwarnings("ignore")
X = np.array([1000.0, 5000.0])            # receptors on the plume axis, m from campus centre


def dT(amb=NIGHT, Q=137e6, fan_heat=None, **kw):
    """Plume-axis dT (K) at 1 and 5 km for the air-cooled campus."""
    src = air_cooled(amb, Q)
    if fan_heat is not None:
        src = dataclasses.replace(src, fan_heat=fan_heat)
    return plume(src, amb, X, **kw)["dT"]


def with_A(A):
    """dT with entrainment constant A in the convective-layer growth (default 0.2)."""
    orig = D.convective_mixing_depth
    D.convective_mixing_depth = functools.partial(orig, A=A)
    try:
        return dT()
    finally:
        D.convective_mixing_depth = orig


rep = lambda **k: dataclasses.replace(NIGHT, **k)
v_h = lambda C_H: C_H * NIGHT.U10 + 0.001        # heat-loss transfer velocity for a given C_H

# (label as printed in Fig. 5, dT with the first value, dT with the second value)
ROWS = [
    ("Lateral spread σθ (30° / 5°)", dT(rep(sigma_theta=30)), dT(rep(sigma_theta=5))),
    ("Stable gradient dθ/dz (0.02 / 0.05 K m⁻¹)", dT(rep(dtheta_dz=0.02)), dT(rep(dtheta_dz=0.05))),
    ("Entrainment constant A (0.3 / 0.1)", with_A(0.3), with_A(0.1)),
    ("Wind speed U₁₀ (4 / 1.5 m s⁻¹)", dT(rep(U10=4.0)), dT(rep(U10=1.5))),
    ("Blend exponent (2 / 8)", dT(blend_n=2), dT(blend_n=8)),
    ("Along-wind release (strips / edge)", dT(alongwind_strips=2), dT()),
    ("Heat rejected (−25% / +25%)", dT(Q=137e6 * 0.75), dT(Q=137e6 * 1.25)),
    ("Heat-transfer coeff. C_H (0.004 / 0.001)", dT(rep(heat_loss_velocity=v_h(0.004))),
     dT(rep(heat_loss_velocity=v_h(0.001)))),
    ("Mixing height z_i (100 / 300 m)", dT(rep(mixing_height=100.0)), dT(rep(mixing_height=300.0))),
    ("Heat per fan (40 / 160 kW)", dT(fan_heat=40e3), dT(fan_heat=160e3)),
]
BASE = dT()

if __name__ == "__main__":
    print(f"reference dT: {BASE[0]:.3f} K at 1 km, {BASE[1]:.3f} K at 5 km\n")
    print(f"{'input (first / second value)':44s} {'1 km':>15s} {'5 km':>15s}")
    for name, lo, hi in ROWS:
        print(f"{name:44s} {lo[0]:6.3f} / {hi[0]:6.3f}  {lo[1]:6.3f} / {hi[1]:6.3f}")

    # Section 4.2: dT at 1 km by stability class and wind speed, air-cooled / water-cooled
    print("\nstability x U10 at 1 km (air-cooled / water-cooled, K); U10 = 1.5, 2, 3, 4, 6 m/s")
    for stab, zi, T, RH in [("B", 2000, 43, 0.10), ("C", 1500, 40, 0.12), ("D", 800, 36, 0.15),
                            ("E", 300, 33, 0.18), ("F", 150, 32, 0.20)]:
        cells = []
        for U in (1.5, 2, 3, 4, 6):
            a = Ambient(T=T, RH=RH, U10=U, stability=stab, mixing_height=zi, elevation=340.0)
            cells.append(f"{plume(air_cooled(a), a, np.array([1000.]))['dT'][0]:.3f}/"
                         f"{plume(water_cooled(a), a, np.array([1000.]))['dT'][0]:.3f}")
        print(stab, " ".join(cells))

    # Fig. 5: tornado chart, inputs ordered by total swing
    order = sorted(ROWS, key=lambda r: -(abs(r[2][0] - r[1][0]) + abs(r[2][1] - r[1][1])))
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for j, b in enumerate(BASE):
        for i, (_, lo, hi) in enumerate(order):
            ax[j].barh(i, 100 * (lo[j] / b - 1), color="#5d6d7e", height=.6)
            ax[j].barh(i, 100 * (hi[j] / b - 1), color="#c0392b", height=.6)
        ax[j].axvline(0, color="k", lw=.7)
        km = "1" if j == 0 else "5"
        ax[j].set_xlabel(f"Change in ΔT at {km} km (%)")
        ax[j].set_title(f"({'ab'[j]}) baseline {b:.2f} K at {km} km", fontsize=8.5, loc="left")
    ax[0].invert_yaxis()
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels([r[0] for r in order], fontsize=7.5)
    ax[1].legend(handles=[Patch(color="#5d6d7e", label="First value in label"),
                          Patch(color="#c0392b", label="Second value in label")],
                 frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_path("fig5_sensitivity.png"), dpi=300, bbox_inches="tight")
    print("\nwrote", fig_path("fig5_sensitivity.png"))
