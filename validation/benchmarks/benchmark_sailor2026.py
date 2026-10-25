"""
benchmark_sailor2026.py - dc_plume against the published vehicle-traverse measurements of
Sailor, Samareh Abolhassani & Martin (2026), ASME J. Eng. Sustain. Bldgs. Cities 7(2) 024501,
doi:10.1115/1.4071922 (CC-BY 4.0): differences in 1.6-2.2 m air temperature between averaging
boxes near four Phoenix-area data centers, June-October 2025.

Geometry-resolved set-up (version 2; rules fixed in PROTOCOL.md before it was run):
  - sources: the hatched facility polygons of the article's Fig. 2 (sailor2026_geometry.json),
    heat = IT x load x 1.3 (PUE) shared by area; roof 12 m (one storey) or 20 m (two),
    fans 2 m above the roof, exhaust 11 K above ambient at 3 m/s, 80 kW per fan;
  - each polygon is a wind-aligned rectangle (crosswind width = the polygon's crosswind
    extent, area preserved); a receptor's along-wind distance is measured from the polygon's
    downwind boundary on the receptor's own wind line;
  - receptors every 5 m along the traverse path in each published averaging box (10 m grid
    over the rotated box of case a); fields of all polygons superposed;
  - compared quantity: the same box difference as published (downwind - upwind; for case e
    the near box T_d3 minus the far box T_d2).
Uncertainty range: load fraction 0.5-1.0 and heat per fan 50-150 kW (central 0.75, 80 kW).
Metrics (Chang & Hanna 2004): FAC2, FB, NMSE, MG. The fully merged plume (fan_heat=None) is
shown for comparison.
Run from the repo root:  python validation/benchmarks/benchmark_sailor2026.py
Outputs: results/benchmarks/{benchmark.csv, metrics.json, fig_benchmark.png}
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import dc_plume as dp

OUT = os.path.join(os.getcwd(), "results", "benchmarks")
GEOM = json.load(open(os.path.join(HERE, "sailor2026_geometry.json")))
CASES = pd.read_csv(os.path.join(HERE, "sailor2026_cases.csv"))
PUE, DT_EXH, W_EXIT, FAN_HEAT, LOAD = 1.3, 11.0, 3.0, 80e3, 0.75


def to_m(pts, mpp):
    """Figure pixels (x right, y down) -> metres (x east, y north)."""
    p = np.asarray(pts, float)
    return np.column_stack([p[:, 0] * mpp, -p[:, 1] * mpp])


def receptors(box, mpp, step=5.0):
    pts = to_m(box["points"], mpp)
    if box["type"] == "path":
        out = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            out.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
        return np.vstack(out + [pts[-1:]])
    lo, hi = pts.min(0), pts.max(0)                     # 'area': 10 m grid inside the polygon
    g = np.stack(np.meshgrid(np.arange(lo[0], hi[0], 10.0), np.arange(lo[1], hi[1], 10.0)), -1).reshape(-1, 2)
    return g[Path(pts).contains_points(g)]


def frame(wind_from_deg):
    """Unit vectors along the wind (toward which it blows) and across it (to the left)."""
    t = np.radians((wind_from_deg + 180.0) % 360.0)     # bearing toward which the wind blows
    u = np.array([np.sin(t), np.cos(t)])                # (east, north)
    return u, np.array([-u[1], u[0]])


def polygon_source(poly, u, v):
    """Wind-aligned rectangle for a polygon, and a function giving each receptor's (x, y)
    in that rectangle's frame with x measured from the polygon's downwind boundary."""
    s, n = poly @ u, poly @ v
    Ly = float(n.max() - n.min())
    area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
    Lx = area / Ly
    n0 = 0.5 * (n.max() + n.min())
    path = Path(poly)
    sl = np.linspace(s.min() - 1, s.max() + 1, 400)

    def xy(P):
        ps, pn = P @ u, P @ v
        x = np.full(len(P), -1e6)
        for i, (a, b) in enumerate(zip(ps, pn)):
            if path.contains_point(P[i]):
                continue                                 # receptor on the building: no value
            line = np.outer(sl, u) + b * v               # the receptor's wind line, in (east, north)
            on = path.contains_points(line)
            if on.any():                                 # the polygon lies on this wind line
                s_dn = sl[on].max()
                if a > s_dn:
                    x[i] = Lx / 2 + (a - s_dn)           # distance beyond its downwind boundary
            elif a - 0.5 * (s.max() + s.min()) > Lx / 2:  # beside the polygon, downwind of its centre
                x[i] = a - 0.5 * (s.max() + s.min())
        return x, pn - n0
    return Lx, Ly, area, xy


def setup(case):
    g, c = GEOM[case], CASES.set_index("case").loc[case]
    mpp = g["m_per_px"]
    polys = [to_m(p, mpp) for p in g["polygons"]]
    u, v = frame(c.wind_from_deg)
    srcs = [polygon_source(p, u, v) for p in polys]
    A = sum(s[2] for s in srcs)
    if "facility_area_from" in g:          # panel shows only part of the facility: use the full one
        _, _, _, _, _, A, _ = setup(g["facility_area_from"])
    storeys = int(c.storeys_stated) if pd.notna(c.storeys_stated) else (1 if c.floor_m2 <= 1.1 * A else 2)
    return g, c, mpp, polys, srcs, A, storeys


def box_mean(case, key, load=LOAD, fan_heat=FAN_HEAT, sig_mult=1.0):
    g, c, mpp, polys, srcs, A, storeys = setup(case)
    roof = 12.0 if storeys == 1 else 20.0
    amb = dp.Ambient(T=c.T_C, RH=0.15, U10=c.U_ms, stability=c.stability, mixing_height=c.zi_m, elevation=370.0,
                     sigma_theta=sig_mult * dp._SIGMA_THETA[c.stability])
    P = receptors(g[key], mpp)
    P = P[~np.array([any(Path(p).contains_point(q) for p in polys) for q in P])]
    dT = np.zeros(len(P))
    for Lx, Ly, area, xy in srcs:
        Q = c.IT_MW * 1e6 * load * PUE * area / A
        src = dp.dry_cooler_source(Q, DT_EXH, amb, height=roof + 2.0, footprint_x=Lx, footprint_y=Ly,
                                   building_height=roof, exit_velocity=W_EXIT, fan_heat=fan_heat)
        x, y = xy(P)
        ok = x > -1e5
        if ok.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = dp.plume(src, amb, x[ok], y[ok])["dT"]
            dT[ok] += np.nan_to_num(d)
    return float(dT.mean())


def predict(case, load=LOAD, fan_heat=FAN_HEAT, sig_mult=1.0):
    """Modelled counterpart of the published box difference."""
    hi, lo = ("near", "far") if case == "e" else ("downwind", "upwind")
    return box_mean(case, hi, load, fan_heat, sig_mult) - box_mean(case, lo, load, fan_heat, sig_mult)


def metrics(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    ok = (o > 0) & (p > 0)
    return dict(n=int(len(o)),
                FAC2=float(np.mean((p / o >= 0.5) & (p / o <= 2.0))),
                FB=float((o.mean() - p.mean()) / (0.5 * (o.mean() + p.mean()))),
                NMSE=float(np.mean((o - p) ** 2) / (o.mean() * p.mean())) if p.mean() > 0 else float("inf"),
                MG=float(np.exp(np.mean(np.log(o[ok] / p[ok])))) if ok.all() else float("nan"))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for c in CASES.itertuples():
        _, _, _, _, _, A, storeys = setup(c.case)
        central = predict(c.case)
        rng = [predict(c.case, l, f) for l in (0.5, 1.0) for f in (50e3, 150e3)]
        rows.append(dict(case=c.case, site=c.site, date=c.date_time_local, compared=c.compared,
                         heat_rejection_area_m2=round(A), storeys=storeys, obs=c.obs_dT_C,
                         pred=round(central, 3), pred_lo=round(min(rng), 3), pred_hi=round(max(rng), 3),
                         pred_merged=round(predict(c.case, fan_heat=None), 3)))
    t = pd.DataFrame(rows)
    t["ratio"] = (t.pred / t.obs).round(2)
    t.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)
    m = dict(fan_array=metrics(t.obs, t.pred), merged_plume=metrics(t.obs, t.pred_merged),
             obs_within_range=int(((t.obs >= t.pred_lo) & (t.obs <= t.pred_hi)).sum()),
             mean_obs=float(t.obs.mean()), mean_pred=float(t.pred.mean()))
    json.dump(m, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = 2.3
    ax.fill_between([0, lim], [0, lim / 2], [0, lim * 2], color="0.92", label="Within a factor of 2")
    ax.plot([0, lim], [0, lim], "k-", lw=0.6)
    ax.errorbar(t.obs, t.pred, yerr=[t.pred - t.pred_lo, t.pred_hi - t.pred], fmt="o", color="C3",
                capsize=3, label="dc_plume, fan-array rise")
    ax.plot(t.obs, t.pred_merged, "x", color="0.4", label="dc_plume, fully merged plume")
    for r in t.itertuples():
        ax.annotate(r.case, (r.obs, r.pred), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set(xlim=(0, 1.5), ylim=(-0.05, lim), xlabel="Observed box difference ΔT (°C)",
           ylabel="Predicted ΔT (K)", title="Sailor et al. (2026) traverses")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_benchmark.png"), dpi=200)
    print(t.drop(columns=["site", "date"]).to_string(index=False))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
