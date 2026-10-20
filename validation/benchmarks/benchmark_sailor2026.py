"""
benchmark_sailor2026.py - dc_plume against the published vehicle-traverse measurements of
Sailor, Samareh Abolhassani & Martin (2026), ASME J. Eng. Sustain. Bldgs. Cities 7(2) 024501,
doi:10.1115/1.4071922 (CC-BY 4.0): downwind-minus-upwind 1.6-2.2 m air temperature at four
Phoenix-area data centers, June-October 2025 (cases in sailor2026_cases.csv; case (c), Aligned,
is omitted because the paper gives no capacity for it).

For each case the model is run with the published wind speed and air temperature, a Turner
daytime stability class, and the facility's critical IT load; the predicted dT is averaged over
the downwind distances of the published averaging boxes. Inputs the paper does not give are
varied over plausible ranges:
  load fraction (actual / rated IT)   0.5 - 1.0  (central 0.75)
  heat per fan (fan_heat)             50 - 150 kW (central 80 kW)
  heat rejected = 1.3 x IT x load     (PUE 1.3, as in the paper)
Metrics (Chang & Hanna 2004): FAC2, fractional bias FB, NMSE, geometric mean bias MG.
The default fully-merged plume (fan_heat=None) is shown for comparison.
Run from the repo root:  python validation/benchmarks/benchmark_sailor2026.py
Outputs: results/benchmarks/{benchmark.csv, metrics.json, fig_benchmark.png}
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import dc_plume as dp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.getcwd(), "results", "benchmarks")
os.makedirs(OUT, exist_ok=True)
ZI_NOTE = "mixing height from season and time of day (Phoenix): 1500-3000 m"


def predict(c, load, fan_heat):
    Q = 1.3 * c.IT_MW * load * 1e6
    L = float(np.sqrt(c.footprint_m2))
    amb = dp.Ambient(T=c.T_C, RH=0.15, U10=c.U_ms, stability=c.stability, mixing_height=c.zi_m,
                     elevation=370.0)
    src = dp.dry_cooler_source(Q, 11.0, amb, height=c.height_m, footprint_x=L, footprint_y=L,
                               building_height=c.height_m - 2.0, exit_velocity=3.0, fan_heat=fan_heat)
    x = L / 2 + np.linspace(c.dist_min_m, c.dist_max_m, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(dp.plume(src, amb, x)["dT"]))


def metrics(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    ok = (o > 0) & (p > 0)
    return dict(n=int(len(o)),
                FAC2=float(np.mean((p / o >= 0.5) & (p / o <= 2.0))),
                FB=float((o.mean() - p.mean()) / (0.5 * (o.mean() + p.mean()))),
                NMSE=float(np.mean((o - p) ** 2) / (o.mean() * p.mean())) if p.mean() > 0 else float("inf"),
                MG=float(np.exp(np.mean(np.log(o[ok] / p[ok])))) if ok.any() else float("nan"))


def main():
    cases = pd.read_csv(os.path.join(HERE, "sailor2026_cases.csv"))
    rows = []
    for c in cases.itertuples():
        central = predict(c, 0.75, 80e3)
        rng = [predict(c, l, f) for l in (0.5, 1.0) for f in (50e3, 150e3)]
        merged = predict(c, 0.75, None)
        rows.append(dict(case=c.case, site=c.site, date=c.date_time_local, obs=c.obs_dT_mean_C,
                         obs_max=c.obs_dT_max_C, pred=round(central, 3), pred_lo=round(min(rng), 3),
                         pred_hi=round(max(rng), 3), pred_merged_default=round(merged, 3)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)
    m = dict(fan_array=metrics(t.obs, t.pred), merged_default=metrics(t.obs, t.pred_merged_default),
             within_range=int(((t.obs >= t.pred_lo) & (t.obs <= t.pred_hi)).sum()))
    json.dump(m, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = 3.2
    ax.fill_between([0, lim], [0, lim / 2], [0, lim * 2], color="0.92", label="Within a factor of 2")
    ax.plot([0, lim], [0, lim], "k-", lw=0.6)
    ax.errorbar(t.obs, t.pred, yerr=[t.pred - t.pred_lo, t.pred_hi - t.pred], fmt="o", color="C3",
                capsize=3, label="dc_plume, fan-array rise")
    ax.plot(t.obs, t.pred_merged_default, "x", color="0.4", label="dc_plume, fully merged plume")
    for r in t.itertuples():
        ax.annotate(r.case, (r.obs, r.pred), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set(xlim=(0, 1.5), ylim=(-0.05, lim), xlabel="Observed downwind − upwind ΔT (°C)",
           ylabel="Predicted ΔT (K)", title="Sailor et al. (2026) traverses")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_benchmark.png"), dpi=200)
    print(t.to_string(index=False))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
