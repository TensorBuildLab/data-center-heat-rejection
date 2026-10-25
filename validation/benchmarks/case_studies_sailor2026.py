"""
case_studies_sailor2026.py - per-case diagnosis of the Sailor et al. (2026) benchmark.

Post-hoc analysis (2026-10-25), run after the version-2 results in PROTOCOL.md were seen. It changes no
rule of the benchmark. For each case it reports:
  1. the modelled 2 m dT field around the site with the published averaging areas (map);
  2. the predicted box difference against wind direction (offsets -90..+90 deg) with the observation
     and the airport-direction uncertainty (sigma 35 deg, wind_uncertainty_sailor2026.py);
  3. a downwash upper bound (all heat held at fan height, no plume rise) and the single input change that would reproduce the observation (wind direction, load fraction,
     roof height) and whether it lies inside the input uncertainty (direction within 2 sigma = 70 deg,
     load 0.5-1.0, roof 8-24 m);
  4. the measured min-max spread in each box (Table 1 of the article);
  5. a neighbour check for cases d and e: buildings visible in the article's Fig. 2 that the protocol
     does not model, and the heat each would need to reject to close the gap (a hypothesis, not a
     correction).
Run from the repo root:  python validation/benchmarks/case_studies_sailor2026.py   (about 10 min)
Outputs: results/benchmarks/case_studies.{csv,json}, fig_case_studies.png
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon as Poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark_sailor2026 as B
dp = B.dp

OUT = os.path.join(os.getcwd(), "results", "benchmarks")
os.makedirs(OUT, exist_ok=True)
CASES = B.CASES.set_index("case")
OFFS = np.arange(-90, 91, 10)
SIG_WD = 35.0

# Table 1 of Sailor et al. (2026): min / max / mean (deg C) in each averaging area
TABLE1 = {"a": {"upwind": (42.1, 43.4, 42.7), "downwind": (42.8, 44.3, 43.5)},
          "b": {"upwind": (38.8, 39.5, 39.1), "downwind": (39.4, 40.0, 39.6)},
          "d": {"upwind": (24.0, 24.4, 24.2), "downwind": (25.0, 25.4, 25.2)},
          "e": {"far": (24.4, 25.0, 24.6), "near": (25.4, 25.8, 25.5)}}

# buildings visible in Fig. 2 but not hatched as the studied facility (figure pixels, see geometry json)
NEIGHBOURS = {"d": {"building north of the facility": [[157.0, 457.0], [210.0, 457.0], [210.0, 501.0], [157.0, 501.0]],
                    "building south of the facility": [[88.0, 632.0], [130.0, 632.0], [130.0, 661.0], [88.0, 661.0]]},
              "e": {"buildings west of the facility": [[345.0, 651.0], [428.0, 651.0], [428.0, 724.0], [345.0, 724.0]],
                    "building south of the facility": [[455.0, 741.0], [538.0, 741.0], [538.0, 757.0], [455.0, 757.0]]}}


def keys(case):
    return ("near", "far") if case == "e" else ("downwind", "upwind")


def field(case, P, wind_from=None, load=B.LOAD, roof=None, polys_q=None, rise=True):
    """Modelled 2 m dT at points P (m) from the facility polygons, or from (polygon, heat) pairs."""
    g, c, mpp, polys, srcs, A, storeys = B.setup(case)
    u, v = B.frame(c.wind_from_deg if wind_from is None else wind_from)
    roof = (12.0 if storeys == 1 else 20.0) if roof is None else roof
    if polys_q is None:
        polys_q = [(p, c.IT_MW * 1e6 * load * B.PUE * B.polygon_source(p, u, v)[2] / A) for p in polys]
    amb = dp.Ambient(T=c.T_C, RH=0.15, U10=c.U_ms, stability=c.stability, mixing_height=c.zi_m, elevation=370.0)
    dT = np.zeros(len(P))
    for p, Q in polys_q:
        Lx, Ly, area, xy = B.polygon_source(p, u, v)
        src = dp.dry_cooler_source(Q, B.DT_EXH, amb, height=roof + 2.0, footprint_x=Lx, footprint_y=Ly,
                                   building_height=roof, exit_velocity=B.W_EXIT, fan_heat=B.FAN_HEAT)
        x, y = xy(P)
        ok = x > -1e5
        if ok.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dT[ok] += np.nan_to_num(dp.plume(src, amb, x[ok], y[ok], plume_rise=rise)["dT"])
    return dT


def box_points(case):
    g, c, mpp, polys, *_ = B.setup(case)
    out = {}
    for k in keys(case):
        P = B.receptors(g[k], mpp)
        out[k] = P[~np.array([any(Path(p).contains_point(q) for p in polys) for q in P])]
    return out


def diff(case, **kw):
    bp = box_points(case)
    hi, lo = keys(case)
    return float(field(case, bp[hi], **kw).mean() - field(case, bp[lo], **kw).mean())


def solve(f, lo, hi, target, n=30):
    """Value in [lo, hi] where f crosses target (scan, then bisection); None if it never does."""
    xs = np.linspace(lo, hi, n)
    ys = np.array([f(x) for x in xs]) - target
    for i in range(n - 1):
        if ys[i] == 0:
            return float(xs[i])
        if ys[i] * ys[i + 1] < 0:
            a, b = xs[i], xs[i + 1]
            for _ in range(8):
                m = 0.5 * (a + b)
                if (f(m) - target) * ys[i] > 0:
                    a = m
                else:
                    b = m
            return float(0.5 * (a + b))
    return None


def main():
    if "--figure" in sys.argv:              # redraw from saved results only
        j = json.load(open(os.path.join(OUT, "case_studies.json")))
        return figure({k: np.array(v) for k, v in j["curves"].items()})
    rows, curves, neigh = [], {}, []
    for case in "abde":
        c = CASES.loc[case]
        obs, wd0 = float(c.obs_dT_C), float(c.wind_from_deg)
        central = diff(case)
        curve = np.array([diff(case, wind_from=wd0 + o) for o in OFFS])
        curves[case] = curve
        # direction change that reproduces the observation: crossing nearest to zero offset
        cross = [OFFS[i] + (OFFS[i + 1] - OFFS[i]) * (obs - curve[i]) / (curve[i + 1] - curve[i])
                 for i in range(len(OFFS) - 1) if (curve[i] - obs) * (curve[i + 1] - obs) <= 0 and curve[i] != curve[i + 1]]
        d_dir = min(cross, key=abs) if cross else None
        load_need = solve(lambda l: diff(case, load=l), 0.2, 3.0, obs, n=15)
        roof_need = solve(lambda r: diff(case, roof=r), 0.0, 40.0, obs, n=17)
        t1 = TABLE1[case]
        hi, lo = keys(case)
        downwash = diff(case, rise=False)          # upper bound: all heat held at roof + 2 m, no rise
        rows.append(dict(case=case, observed=obs, central=round(central, 3), downwash_no_rise=round(downwash, 3),
                         direction_change_deg=None if d_dir is None else round(float(d_dir)),
                         direction_inside_2sigma=None if d_dir is None else bool(abs(d_dir) <= 2 * SIG_WD),
                         load_needed=None if load_need is None else round(load_need, 2),
                         load_inside_range=None if load_need is None else bool(0.5 <= load_need <= 1.0),
                         roof_needed_m=None if roof_need is None else round(roof_need, 1),
                         roof_inside_range=None if roof_need is None else bool(8 <= roof_need <= 24),
                         box_hi_min_max=f"{t1[hi][0]}-{t1[hi][1]}", box_lo_min_max=f"{t1[lo][0]}-{t1[lo][1]}",
                         point_diff_range=f"{t1[hi][0] - t1[lo][1]:+.1f} to {t1[hi][1] - t1[lo][0]:+.1f}"))
        if case in NEIGHBOURS:
            g, cc, mpp, *_ = B.setup(case)
            bp = box_points(case)
            gap = obs - central
            for name, px in NEIGHBOURS[case].items():
                p = B.to_m(px, mpp)
                unit = 10e6
                contrib = float(field(case, bp[hi], polys_q=[(p, unit)]).mean()
                                - field(case, bp[lo], polys_q=[(p, unit)]).mean())
                neigh.append(dict(case=case, neighbour=name, area_m2=round(float(B.polygon_source(p, *B.frame(wd0))[2])),
                                  dT_diff_per_10MW=round(contrib, 3),
                                  heat_needed_MW=None if contrib <= 1e-4 else round(10 * gap / contrib, 0)))
        print(rows[-1])
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "case_studies.csv"), index=False)
    nb = pd.DataFrame(neigh)
    nb.to_csv(os.path.join(OUT, "neighbour_check.csv"), index=False)
    json.dump(dict(offsets_deg=OFFS.tolist(), curves={k: v.round(4).tolist() for k, v in curves.items()},
                   cases=rows, neighbours=neigh), open(os.path.join(OUT, "case_studies.json"), "w"), indent=1)
    print(nb.to_string(index=False))
    figure(curves)


def figure(curves):
    plt.rcParams.update({"font.size": 7.5})
    fig = plt.figure(figsize=(7.2, 11.0), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[1.15, 1, 1, 1, 1])
    wu = pd.read_csv(os.path.join(OUT, "wind_uncertainty.csv"))
    wu = wu[wu.sigma_wd_deg == 35].set_index("case")
    bench = pd.read_csv(os.path.join(OUT, "benchmark.csv")).set_index("case")
    # (a) predicted vs observed with the 5-95 % range from wind direction and load
    ax = fig.add_subplot(gs[0, 0])
    lim = 2.1
    ax.fill_between([0, lim], [0, lim / 2], [0, lim * 2], color="0.92", label="within a factor of 2")
    ax.plot([0, lim], [0, lim], "k-", lw=0.6)
    for case in "abde":
        o, r = CASES.loc[case].obs_dT_C, wu.loc[case]
        ax.plot([o, o], [r.p05, r.p95], color="C3", lw=1.2, alpha=0.5)
        ax.plot(o, bench.pred[case], "o", color="C3", ms=4)
        ax.plot(o, bench.pred_merged[case], "x", color="0.35", ms=4)
        ax.annotate(case, (o, bench.pred[case]), xytext=(4, 2), textcoords="offset points")
    ax.plot([], [], "o", color="C3", ms=4, label="dc_plume, central inputs")
    ax.plot([], [], color="C3", alpha=0.5, label="5–95 % (wind direction, load)")
    ax.plot([], [], "x", color="0.35", label="one merged plume")
    ax.set(xlim=(0, 1.4), ylim=(-0.2, lim), xlabel="Observed ΔT (°C)", ylabel="Predicted ΔT (K)")
    ax.set_title("(a) Predicted vs observed", loc="left")
    ax.legend(fontsize=6, loc="upper right")
    # (b) measured point-to-point spread (Table 1) vs observed mean and model range
    ax = fig.add_subplot(gs[0, 1])
    for k, case in enumerate("abde"):
        hi, lo = keys(case)
        t = TABLE1[case]
        y = 3 - k
        ax.plot([t[hi][0] - t[lo][1], t[hi][1] - t[lo][0]], [y + 0.12] * 2, color="k", lw=3, alpha=0.35,
                solid_capstyle="butt", label="measured spread (Table 1)" if k == 0 else None)
        ax.plot(CASES.loc[case].obs_dT_C, y + 0.12, "ks", ms=4, label="observed mean" if k == 0 else None)
        r = wu.loc[case]
        ax.plot([r.p05, r.p95], [y - 0.12] * 2, color="C3", lw=3, alpha=0.5, solid_capstyle="butt",
                label="model 5–95 %" if k == 0 else None)
        ax.plot(bench.pred[case], y - 0.12, "o", color="C3", ms=4, label="model central" if k == 0 else None)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set(yticks=[3, 2, 1, 0], yticklabels=["a", "b", "d", "e"], ylim=(-0.6, 3.6), xlim=(-0.8, 4.2), xlabel="ΔT (K)")
    ax.set_title("(b) Measured spread and model range", loc="left")
    ax.legend(fontsize=6, loc="lower right")
    letters = iter("cdefghij")
    for row, case in enumerate("abde", start=1):
        g, c, mpp, polys, *_ = B.setup(case)
        allp = np.vstack(polys + [B.receptors(g[k], mpp) for k in keys(case)])
        lo, hi = allp.min(0) - 120, allp.max(0) + 120
        X, Y = np.meshgrid(np.arange(lo[0], hi[0], 15.0), np.arange(lo[1], hi[1], 15.0))
        P = np.column_stack([X.ravel(), Y.ravel()])
        Z = field(case, P).reshape(X.shape)
        inside = np.array([any(Path(p).contains_point(q) for p in polys) for q in P]).reshape(X.shape)
        Z[inside] = np.nan
        ax = fig.add_subplot(gs[row, 0])
        pc = ax.pcolormesh(X - lo[0], Y - lo[1], Z, cmap="Reds", vmin=0, vmax=max(0.5, np.nanpercentile(Z, 99)),
                           shading="auto")
        for p in polys:
            ax.add_patch(Poly(p - lo, fc="0.55", ec="k", lw=0.4))
        for k in keys(case):
            pts = B.receptors(g[k], mpp) - lo
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=1.2, color="b" if k in ("downwind", "near") else "k",
                    label={"downwind": "downwind area", "upwind": "upwind area", "near": "near area", "far": "far area"}[k])
        u, _ = B.frame(c.wind_from_deg)
        ctr = np.vstack(polys).mean(0) - lo
        ax.annotate("", ctr + 160 * u, ctr, arrowprops=dict(arrowstyle="->", lw=1.2))
        ax.set(aspect="equal", xlabel="m", ylabel="m")
        ax.set_title(f"({next(letters)}) Case {case}: modelled 2 m ΔT", loc="left")
        ax.legend(fontsize=5.5, loc="best", markerscale=4)
        fig.colorbar(pc, ax=ax, shrink=0.85, label="ΔT (K)")
        ax = fig.add_subplot(gs[row, 1])
        ax.axvspan(-SIG_WD, SIG_WD, color="0.92", label="±1σ airport direction error")
        ax.plot(OFFS, curves[case], "-o", color="C3", ms=2.5, label="model (load 0.75)")
        ax.axhline(CASES.loc[case].obs_dT_C, color="k", ls="--", lw=0.8, label="observed")
        ax.set(xlim=(-90, 90), ylim=(min(-0.15, curves[case].min() - 0.05), max(1.1, curves[case].max() + 0.1)),
               xlabel="Wind-direction offset (deg)", ylabel="ΔT (K)")
        ax.set_title(f"({next(letters)}) Case {case}: sensitivity to wind direction", loc="left")
        if row == 1:
            ax.legend(fontsize=6, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_case_studies.png"), dpi=300)


if __name__ == "__main__":
    main()
