"""
wind_uncertainty_sailor2026.py - how much of the Sailor et al. (2026) benchmark error can be explained
by the uncertainty of the inputs, rather than by the model.

Post-hoc analysis (2026-10-21), run after the version-2 results in PROTOCOL.md were seen. It changes
no rule of the benchmark; it only asks what range of predictions the published inputs allow.

1. Wind-direction error. The traverse winds come from airport stations several kilometres from the
   sites. The error of such a direction at a site is estimated from data: daytime (10-15 MST) hours
   with 1-3 m/s at both stations of each pair of Phoenix-area ASOS stations (KCHD, KIWA, KFFZ, KPHX;
   IEM archive, 2019-2025). The median absolute direction difference between two airports
   10-20 km apart is 40-60 deg, i.e. about 35 deg for one airport vs. the site (pair / sqrt 2);
   50 deg is run as a sensitivity case.
2. Prediction distribution. For each case the geometry-resolved prediction (benchmark_sailor2026.predict)
   is evaluated for wind offsets of -90..+90 deg (10 deg steps, Gaussian weights) and load fractions
   0.5 / 0.75 / 1.0 (equal weights). Reported: weighted 5th / 50th / 95th percentiles, the percentile
   at which the observation falls, and the probability that a prediction lies within 15 % of the
   observation (what a 15 % agreement target would require).

Run from the repo root:  python validation/benchmarks/wind_uncertainty_sailor2026.py   (about 15 min)
Outputs: results/benchmarks/wind_uncertainty.{csv,json}
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import benchmark_sailor2026 as B
import asos_wind_signature as aws

OUT = os.path.join(os.getcwd(), "results", "benchmarks")
os.makedirs(OUT, exist_ok=True)
OBS = B.CASES.set_index("case").obs_dT_C.to_dict()
OFFS = np.arange(-90, 91, 10)
LOADS = (0.5, 0.75, 1.0)


def ang(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def airport_direction_error():
    st = {s: aws.fetch(s, "2019-01-01", "2026-01-01") for s in ("CHD", "IWA", "FFZ", "PHX")}
    rows = []
    for a, b in (("CHD", "IWA"), ("CHD", "FFZ"), ("IWA", "FFZ"), ("CHD", "PHX")):
        d = st[a][["drct", "U"]].join(st[b][["drct", "U"]], lsuffix="_a", rsuffix="_b", how="inner").dropna()
        loc = d.index + pd.Timedelta(hours=-7)
        m = ((loc.hour >= 10) & (loc.hour <= 15) & d.U_a.between(1, 3) & d.U_b.between(1, 3)
             & (d.drct_a > 0) & (d.drct_b > 0))
        e = ang(d.drct_a[m], d.drct_b[m])
        rows.append(dict(pair=f"K{a}-K{b}", hours=int(m.sum()), frac_within_30=float(np.mean(np.abs(e) <= 30)),
                         median_abs_diff=float(np.median(np.abs(e)))))
    return rows


def weighted_quantile(v, w, q):
    i = np.argsort(v)
    cw = np.cumsum(w[i]) / w.sum()
    return float(np.interp(q, cw, v[i]))


def prediction_grid(case):
    base = float(B.CASES.set_index("case").loc[case, "wind_from_deg"])
    g = np.zeros((len(OFFS), len(LOADS)))
    try:
        for i, o in enumerate(OFFS):
            B.CASES.loc[B.CASES.case == case, "wind_from_deg"] = base + o
            for j, load in enumerate(LOADS):
                g[i, j] = B.predict(case, load=load)
    finally:
        B.CASES.loc[B.CASES.case == case, "wind_from_deg"] = base
    return g


if __name__ == "__main__":
    pairs = airport_direction_error()
    print("airport pairs, daytime 10-15 MST, 1-3 m/s at both:")
    for p in pairs:
        print(f"  {p['pair']:10s} {p['hours']:5d} h  within 30 deg: {p['frac_within_30']:.2f}  "
              f"median |diff|: {p['median_abs_diff']:.0f} deg")
    grids = {c: prediction_grid(c) for c in "abde"}
    rows = []
    for sig in (35, 50):
        w = np.repeat(np.exp(-0.5 * (OFFS / sig) ** 2)[:, None], len(LOADS), axis=1).ravel()
        for c, g in grids.items():
            v, o = g.ravel(), OBS[c]
            rows.append(dict(sigma_wd_deg=sig, case=c, observed=o,
                             central=float(g[list(OFFS).index(0), 1]),
                             p05=weighted_quantile(v, w, .05), p50=weighted_quantile(v, w, .5),
                             p95=weighted_quantile(v, w, .95),
                             obs_percentile=float(w[v <= o].sum() / w.sum()),
                             prob_within_15pct=float(w[np.abs(v - o) <= 0.15 * o].sum() / w.sum())))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "wind_uncertainty.csv"), index=False, float_format="%.3f")
    json.dump(dict(airport_pairs=pairs, offsets_deg=OFFS.tolist(), loads=LOADS,
                   grids={c: g.round(4).tolist() for c, g in grids.items()}),
              open(os.path.join(OUT, "wind_uncertainty.json"), "w"), indent=1)
    print(tab.round(2).to_string(index=False))
