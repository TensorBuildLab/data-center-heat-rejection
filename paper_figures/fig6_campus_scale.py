"""
fig6_campus_scale.py - effect of campus size on the downwind temperature rise (Section 4.4, Fig. 6).

IT capacity is swept from 25 to 400 MW on the reference night. Heat rejected scales with IT load
(1.37 W/W air-cooled, 1.25 W/W water-cooled), and the footprint grows at constant heat flux per
unit area (side ~ sqrt(IT)), so a bigger campus is a wider source, not a hotter one.
dT is evaluated on the plume axis 0.5 km and 3 km beyond the downwind fence line, so campuses of
different size are compared at the same distance from their edge. The table also gives values at
fixed distances from the campus centre (1 and 5 km) and the self-induced mixing depth h_mix.

Run from the repo root:  python paper_figures/fig6_campus_scale.py   -> table on stdout, figures/fig6_scale.png
"""
import warnings

import numpy as np
from matplotlib.ticker import NullFormatter

from reference_case import AIR, HEAT_PER_IT, NIGHT, WATER, air_cooled, fig_path, plt, water_cooled   # also puts the repo root on sys.path
from dc_plume import plume

warnings.filterwarnings("ignore")
IT = np.array([25, 50, 100, 200, 400])           # MW


def campuses(it):
    side = 600 * np.sqrt(it / 100)               # air-cooled square footprint, m
    Lt = np.sqrt(300 * 200 * it / 100)           # water-cooled footprint, aspect 1.22^2 as in the reference
    dry = air_cooled(Q=HEAT_PER_IT["air"] * 1e6 * it, side=side)
    tow = water_cooled(Q=HEAT_PER_IT["water"] * 1e6 * it, inlet_flow=16.5 * it, fx=Lt * 1.22, fy=Lt / 1.22)
    return dry, tow, side / 2, Lt * 0.61           # sources and their half-length along the wind


rows = []
for it in IT:
    dry, tow, edge_d, edge_t = campuses(it)
    fence_d = plume(dry, NIGHT, np.array([edge_d + 500, edge_d + 3000]))
    fence_t = plume(tow, NIGHT, np.array([edge_t + 500, edge_t + 3000]))
    centre_d = plume(dry, NIGHT, np.array([1000.0, 5000.0]))
    centre_t = plume(tow, NIGHT, np.array([1000.0, 5000.0]))
    rows.append(dict(IT=it, side=2 * edge_d, h_mix=fence_d["mixing_depth"],
                     air_05=fence_d["dT"][0], air_3=fence_d["dT"][1],
                     water_05=fence_t["dT"][0], water_3=fence_t["dT"][1],
                     air_c1=centre_d["dT"][0], air_c5=centre_d["dT"][1],
                     water_c1=centre_t["dT"][0], water_c5=centre_t["dT"][1]))

if __name__ == "__main__":
    print("IT(MW) side(m) h_mix(m) | beyond fence: air 0.5/3 km, water 0.5/3 km | from centre: air 1/5 km, water 1/5 km")
    for r in rows:
        print(f"{r['IT']:5d} {r['side']:7.0f} {r['h_mix']:7.1f} | {r['air_05']:.3f} {r['air_3']:.3f}  "
              f"{r['water_05']:.3f} {r['water_3']:.3f} | {r['air_c1']:.3f} {r['air_c5']:.3f}  "
              f"{r['water_c1']:.3f} {r['water_c5']:.3f}")

    get = lambda k: [r[k] for r in rows]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(IT, get("air_05"), "o-", color=AIR, label="Air-cooled, 0.5 km")
    ax.plot(IT, get("air_3"), "o--", color=AIR, mfc="w", label="Air-cooled, 3 km")
    ax.plot(IT, get("water_05"), "s-", color=WATER, label="Water-cooled, 0.5 km")
    ax.plot(IT, get("water_3"), "s--", color=WATER, mfc="w", label="Water-cooled, 3 km")
    ax.set(xscale="log", xlabel="IT capacity (MW)", ylabel="ΔT at 2 m on plume axis (K)", ylim=(0, 0.78))
    ax.set_xticks(IT)
    ax.set_xticklabels(IT)
    ax.xaxis.set_minor_formatter(NullFormatter())   # newer matplotlib would also label minor log ticks
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=.3)
    ax.set_title("Distance beyond downwind fence line", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(fig_path("fig6_scale.png"), dpi=200)
    print("\nwrote", fig_path("fig6_scale.png"))
