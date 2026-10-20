"""
exceedance.py - how often, and over how much land, does dc_plume predict near-surface
warming above 0.5 K and 1 K around large air-cooled campuses and clusters?

Hourly Phoenix TMY3 (climate.onebuilding.org) wind, temperature and sky cover; Pasquill
class by the Turner method; mixing height 1500/800/300/150 m (A-C/D/E/F). For each hour
the 2 m dT field on a 100 m grid (+-6 km) is taken from fields precomputed in wind-aligned
coordinates for that hour's class and speed bin (0.5 m/s), rotated to the hour's wind
direction. Calm hours (U < 0.5 m/s) are counted separately and not mapped (the model's
range starts at 1 m/s); 0.5-1 m/s is run at 1 m/s.

Scenarios (air-cooled, heat = 1.37 x IT): footprint density 1 kW IT per m2 for single
campuses; a 3 GW cluster spread over 3 x 3 km (0.33 kW/m2).
Run from the repo root:  python analysis/exceedance.py   (FAN_HEAT=0 for the fully merged plume)
Outputs: results/exceedance/{summary.csv, hours*_*.npy, meandT_*.npy, fig_exceedance_maps.png}
"""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dc_plume as dp

OUT = os.environ.get("EXC_OUT", "results/exceedance")
os.makedirs(OUT, exist_ok=True)
EPW = "data/weather/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw"   # python building_energy/get_weather.py
SCEN = {"100 MW campus": (100, 316), "300 MW campus": (300, 548), "1 GW campus": (1000, 1000),
        "3 GW cluster": (3000, 3000)}
G = np.arange(-6000, 6001, 100.0)
X, Y = np.meshgrid(G, G, indexing="xy")           # east, north (m) from campus centre
ZI = {"A": 1500.0, "B": 1500.0, "C": 1500.0, "D": 800.0, "E": 300.0, "F": 150.0}
UB = np.arange(1.0, 10.01, 0.5)
FAN_HEAT = float(os.environ.get("FAN_HEAT", 80e3)) or None   # W per fan; 0 -> merged plume

# ---- weather and stability (same Turner scheme as energy_demo/make_epw.py)
LAT, LON, TZ = 33.45, -111.98, -7.0
rows = [l.split(",") for l in open(EPW, encoding="latin-1").read().splitlines()[8:] if l.strip()]
w = pd.DataFrame({k: [float(r[i]) for r in rows] for k, i in
                  dict(month=1, day=2, hour=3, T=6, wdir=20, wspd=21, sky=22).items()})
doy = pd.to_datetime(dict(year=2001, month=w.month.astype(int), day=w.day.astype(int))).dt.dayofyear.values
decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
B = 2 * np.pi * (doy - 81) / 364
eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
ha = np.radians(15 * ((w.hour.values - 0.5) + (4 * (LON - 15 * TZ) + eot) / 60 - 12))
elev = np.degrees(np.arcsin(np.sin(np.radians(LAT)) * np.sin(decl) + np.cos(np.radians(LAT)) * np.cos(decl) * np.cos(ha)))


def turner(U, el, sky):
    if el <= 0:
        cloudy = sky >= 5
        if U < 3:
            return "E" if cloudy else "F"
        return "D" if (U >= 5 or cloudy) else "E"
    ins = 3 if el > 60 else 2 if el > 35 else 1 if el > 15 else 0
    if sky >= 8:
        ins = min(ins, 1)
    if ins == 0:
        return "D"
    tab = {3: "ABBCC", 2: "ABCCD", 1: "BCCDD"}
    k = 0 if U < 2 else 1 if U < 3 else 2 if U < 5 else 3 if U < 6 else 4
    return tab[ins][k]


stab = np.array([turner(u, e, s) for u, e, s in zip(w.wspd, elev, w.sky)])
calm = w.wspd.values < 0.5
night = elev <= 0


def fields(it_mw, L):
    """Precomputed dT on a wind-aligned grid (along, cross) for each class x speed bin."""
    A = np.arange(-6000, 8501, 100.0)
    C = np.arange(-8500, 8501, 100.0)
    AA, CC = np.meshgrid(A, C, indexing="ij")
    F = {}
    for cls in "ABCDEF":
        for U in UB:
            amb = dp.Ambient(T=25.0, RH=0.2, U10=U, stability=cls, mixing_height=ZI[cls], elevation=340.0)
            src = dp.dry_cooler_source(1.37 * it_mw * 1e6, 12.0, amb, height=15.0, footprint_x=L,
                                       footprint_y=L, building_height=12.0, exit_velocity=3.0,
                                       fan_heat=FAN_HEAT)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                F[(cls, U)] = np.nan_to_num(dp.plume(src, amb, AA, CC)["dT"])
    return A, C, F


def run(name, it_mw, L):
    A, C, F = fields(it_mw, L)
    h05 = np.zeros_like(X); h1 = np.zeros_like(X); h2 = np.zeros_like(X); tot = np.zeros_like(X)
    hn1 = np.zeros_like(X); hd1 = np.zeros_like(X)
    pt = {d: [] for d in (500, 1000, 2000)}                 # dT on the hour's plume axis beyond the fence
    for i in range(len(w)):
        if calm[i]:
            continue
        U = float(UB[np.abs(UB - max(w.wspd[i], 1.0)).argmin()])
        f = F[(stab[i], U)]
        to = np.radians((w.wdir[i] + 180) % 360)
        along = X * np.sin(to) + Y * np.cos(to)
        cross = -X * np.cos(to) + Y * np.sin(to)
        ia = np.clip(np.round((along - A[0]) / 100).astype(int), 0, len(A) - 1)
        ic = np.clip(np.round((cross - C[0]) / 100).astype(int), 0, len(C) - 1)
        v = f[ia, ic]
        h05 += v > 0.5; h1 += v > 1.0; h2 += v > 2.0; tot += v
        if night[i]:
            hn1 += v > 1.0
        else:
            hd1 += v > 1.0
        for d in pt:
            j = int(round((L / 2 + d - A[0]) / 100)); pt[d].append(f[j, int(round(-C[0] / 100))])
    np.save(f"{OUT}/hours1_{it_mw}.npy", h1)
    np.save(f"{OUT}/hours05_{it_mw}.npy", h05)
    np.save(f"{OUT}/meandT_{it_mw}.npy", tot / len(w))
    cell = 0.01                                                 # km2 per 100 m cell
    inside = (np.abs(X) <= L / 2) & (np.abs(Y) <= L / 2)
    res = dict(scenario=name, IT_MW=it_mw, footprint_km=L / 1000,
               area_km2_ge100h_gt1K=float(((h1 >= 100) & ~inside).sum() * cell),
               area_km2_ge500h_gt1K=float(((h1 >= 500) & ~inside).sum() * cell),
               area_km2_ge500h_gt05K=float(((h05 >= 500) & ~inside).sum() * cell),
               max_hours_gt1K_outside=float(h1[~inside].max()), max_hours_gt2K_outside=float(h2[~inside].max()),
               max_night_hours_gt1K=float(hn1[~inside].max()), max_day_hours_gt1K=float(hd1[~inside].max()),
               max_annual_mean_dT_outside=round(float((tot / len(w))[~inside].max()), 3))
    for d, v in pt.items():
        v = np.array(v)
        res[f"axis_{d}m_hours_gt1K"] = int((v > 1).sum())
        res[f"axis_{d}m_p95"] = round(float(np.percentile(v, 95)), 2)
        res[f"axis_{d}m_max"] = round(float(v.max()), 2)
    print(res, flush=True)
    return res, h1


if __name__ == "__main__":
    out, maps = [], {}
    for name, (it, L) in SCEN.items():
        r, h1 = run(name, it, L)
        out.append(r); maps[name] = (h1, L)
    s = pd.DataFrame(out)
    s.to_csv(f"{OUT}/summary.csv", index=False)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for a, (name, (h1, L)) in zip(ax, maps.items()):
        im = a.pcolormesh(G / 1000, G / 1000, np.where(h1 > 0, h1, np.nan), cmap="magma_r", vmin=0, vmax=1500,
                          shading="auto")
        a.add_patch(plt.Rectangle((-L / 2000, -L / 2000), L / 1000, L / 1000, fill=True, color="0.6"))
        a.contour(G / 1000, G / 1000, h1, levels=[100, 500], colors=["0.3", "k"], linewidths=[0.6, 1.0])
        a.set(title=name, xlabel="East (km)", aspect="equal", xlim=(-6, 6), ylim=(-6, 6))
    ax[0].set_ylabel("North (km)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Hours per year with ΔT > 1 K (Phoenix TMY3)")
    fig.savefig(f"{OUT}/fig_exceedance_maps.png", dpi=200)
    print(s.to_string(index=False))
    print("calm hours (not mapped):", int(calm.sum()), "of 8760; night hours:", int(night.sum()))
