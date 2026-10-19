"""
station_analysis.py - full KGYR station test (validation V6) for the manuscript, Section 4.6.

Uses asos_wind_signature.py for download/parsing/screening and adds:
  - reporting coverage by local hour for target and reference stations
  - sector frequency (share of screened hours, number of nights)
  - a vectorized night-block bootstrap (same statistic as asos_wind_signature.sector_effect)
  - difference in differences before/after commissioning, bootstrapped jointly
  - sensitivity runs (sector half-width, sector centre, reference, hour window,
    seasonal adjustment) and a by-year sector effect
  - the dc_plume prediction for the station under evening conditions
Writes everything to results/station/.   Run from the repo root:  python validation/station_analysis.py
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path[:0] = [_HERE, _os.path.dirname(_HERE)]   # this folder and the repo root (dc_plume.py)
import asos_wind_signature as aws
import dc_plume as dp

OUT = "results/station"
START, END = "2015-01-01", "2026-01-01"
COMMISSIONED = "2020-10-14"          # Stream Phoenix I (PHXA1) commissioning announced, 4.5 MW
CAMPUS_FROM, HW = 111.0, 15.0
N_BOOT = 2000
os.makedirs(OUT, exist_ok=True)

data = {s: aws.fetch(s, START, END) for s in ("GYR", "LUF", "BXK")}


# ---------------------------------------------------------------- coverage
def coverage():
    rows = {}
    for s, df in data.items():
        loc = df.dropna(subset=["T"]).index + pd.Timedelta(hours=-7)
        days = (pd.Timestamp(END) - pd.Timestamp(START)).days
        rows[s] = pd.Series(loc.hour).value_counts().reindex(range(24), fill_value=0) / days
    cov = pd.DataFrame(rows)
    cov.index.name = "hour_MST (rounded obs time)"
    cov.to_csv(f"{OUT}/coverage_by_local_hour.csv", float_format="%.3f")
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for s, c in zip(cov, ("C3", "C0", "C2")):
        ax.plot(cov.index, cov[s], "o-", color=c, label=f"K{s}", ms=3)
    ax.axvspan(21.5, 23.5, color="0.85"); ax.axvspan(-0.5, 5.5, color="0.85",
                                                       label="planned 22-05 window")
    ax.set(xlabel="Local standard time (MST), hour of rounded report", ylabel="Fraction of days with a report",
           xlim=(-0.5, 23.5), ylim=(0, 1.05), xticks=range(0, 24, 3),
           title="Routine METAR coverage, 2015-2025")
    ax.legend(fontsize=7, loc="center left"); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_coverage.png", dpi=200); plt.close(fig)
    return cov


# ---------------------------------------------------------------- statistics
def night_table(d, campus_from, hw):
    ins = np.abs(aws.angdiff(d["drct"], campus_from)) <= hw
    g = d.assign(ins=ins)
    a = g[g.ins].groupby("night")["dT"].mean()
    b = g[~g.ins].groupby("night")["dT"].mean()
    t = pd.DataFrame({"a": a, "b": b})
    return t, int(ins.sum())


def _stat(t, W):
    """Sector-minus-other statistic for bootstrap weight matrix W (n_boot x n_nights)."""
    ha, hb = t["a"].notna().to_numpy(), t["b"].notna().to_numpy()
    a, b = np.nan_to_num(t["a"].to_numpy()), np.nan_to_num(t["b"].to_numpy())
    return (W @ (a * ha)) / (W @ ha) - (W @ (b * hb)) / (W @ hb)


def weights(n, rng, n_boot=N_BOOT):
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.stack([np.bincount(r, minlength=n) for r in idx]).astype(float)


def sector_test(d, campus_from=CAMPUS_FROM, hw=HW, commissioned=COMMISSIONED, seed=0):
    rng = np.random.default_rng(seed)
    t, n_sec_hours = night_table(d, campus_from, hw)
    res = dict(hours=len(d), nights=int(d["night"].nunique()), sector_hours=n_sec_hours,
               sector_pct_hours=100 * n_sec_hours / max(len(d), 1),
               sector_nights=int(t["a"].notna().sum()), other_nights=int(t["b"].notna().sum()))
    if res["sector_nights"] < 3:
        return res
    one = np.ones((1, len(t)))
    res["effect"] = float(_stat(t, one)[0])
    bs = _stat(t, weights(len(t), rng))
    res["ci"] = [float(v) for v in np.nanpercentile(bs, [2.5, 97.5])]
    res["p_one_sided"] = float(np.mean(bs <= 0))
    if commissioned:
        c = pd.Timestamp(commissioned, tz="UTC").normalize()
        pre, post = t[t.index < c], t[t.index >= c]
        if pre["a"].notna().sum() >= 3 and post["a"].notna().sum() >= 3:
            e_pre, e_post = _stat(pre, np.ones((1, len(pre))))[0], _stat(post, np.ones((1, len(post))))[0]
            b_pre, b_post = _stat(pre, weights(len(pre), rng)), _stat(post, weights(len(post), rng))
            did = b_post - b_pre
            res.update(pre=dict(effect=float(e_pre), ci=[float(v) for v in np.nanpercentile(b_pre, [2.5, 97.5])],
                                sector_nights=int(pre["a"].notna().sum())),
                       post=dict(effect=float(e_post), ci=[float(v) for v in np.nanpercentile(b_post, [2.5, 97.5])],
                                 sector_nights=int(post["a"].notna().sum())),
                       did=float(e_post - e_pre), did_ci=[float(v) for v in np.nanpercentile(did, [2.5, 97.5])])
    return res


def screened(tgt, ref, window, months=None, deseason=False, umin=1.0, umax=4.0):
    d = aws.prepare(data[tgt], data[ref], night=window, umin=umin, umax=umax)
    if months:
        d = d[(d.index + pd.Timedelta(hours=-7)).month.isin(months)]
    if deseason and len(d):
        ym = (d.index + pd.Timedelta(hours=-7)).strftime("%Y-%m")
        d = d.assign(dT=d["dT"] - d.groupby(ym)["dT"].transform("mean"))
    return d


def fmt(r):
    if "effect" not in r:
        return f"n/a ({r['hours']} h, {r['sector_nights']} sector nights)"
    return f"{r['effect']:+.3f} K [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"


def harmonic_test(d, campus_from=CAMPUS_FROM, hw=HW, commissioned=COMMISSIONED, seed=0, n_boot=N_BOOT):
    """Hour-level OLS: dT ~ 1 + post + cos/sin(theta) + cos/sin(2 theta) [each also x post]
    + sector + sector x post. The two harmonics absorb the smooth direction-dependent
    site difference (Validation Plan 6.3); 'sector' is the local excess in the campus
    sector over that background, 'sector x post' its change after commissioning.
    CIs: bootstrap over nights."""
    th = np.radians(d["drct"].to_numpy())
    sec = (np.abs(aws.angdiff(d["drct"], campus_from)) <= hw).astype(float)
    post = (d.index >= pd.Timestamp(commissioned, tz="UTC")).astype(float)
    H = np.column_stack([np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)])
    X = np.column_stack([np.ones_like(th), post, H, H * post[:, None], sec, sec * post])
    y = d["dT"].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    nights = d["night"].to_numpy()
    u, inv = np.unique(nights, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(u))]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(u), len(u))])
        bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][-2:])
    bs = np.array(bs)
    pre_eff, did = beta[-2], beta[-1]
    post_eff = bs[:, 0] + bs[:, 1]
    return dict(sector_pre=float(pre_eff), sector_pre_ci=[float(v) for v in np.percentile(bs[:, 0], [2.5, 97.5])],
                sector_post=float(pre_eff + did), sector_post_ci=[float(v) for v in np.percentile(post_eff, [2.5, 97.5])],
                did=float(did), did_ci=[float(v) for v in np.percentile(bs[:, 1], [2.5, 97.5])],
                sector_nights_pre=int(d[(sec > 0) & (post == 0)]["night"].nunique()),
                sector_nights_post=int(d[(sec > 0) & (post > 0)]["night"].nunique()))


# ---------------------------------------------------------------- model prediction
def prediction():
    """dc_plume plume-axis dT at KGYR (1.3 km from the PHXA1 building centre) for an
    air-cooled building (heat = 1.37 x IT, as in the reference case) on a 200 x 200 m
    footprint (418,000 ft2), chillers at 15 m. Evening vs late-night conditions."""
    rows = []
    for label, kw in {"class E, U10 2.5 m/s, z_i 300 m": dict(stability="E", U10=2.5, mixing_height=300.0),
                      "class F, U10 2 m/s, z_i 150 m": dict(stability="F", U10=2.0, mixing_height=150.0)}.items():
        amb = dp.Ambient(T=30.0, RH=0.20, elevation=330.0, **kw)
        for it in (4.5, 20.0, 40.0):
            src = dp.dry_cooler_source(1.37 * it * 1e6, 12.0, amb, height=15.0, footprint_x=200.0,
                                       footprint_y=200.0, building_height=12.0)
            r = dp.plume(src, amb, np.array([1300.0]))
            rows.append(dict(conditions=label, IT_MW=it, dT_K=round(float(r["dT"][0]), 3),
                             area_weight=round(r["area_weight"], 2)))
    p = pd.DataFrame(rows); p.to_csv(f"{OUT}/model_prediction_KGYR.csv", index=False)
    return p


# ---------------------------------------------------------------- runs
def signature_plot(d, tag, title):
    tab = aws.binned(d)
    tab.to_csv(f"{OUT}/signature_{tag}.csv")
    # night-block bootstrap CI per 10-degree bin
    rng = np.random.default_rng(1)
    b = (np.floor(d["drct"] / 10) * 10 + 5) % 360
    per = d.assign(b=b).groupby(["b", "night"])["dT"].mean()
    lo, hi = [], []
    for k in tab.index:
        v = per.loc[k].to_numpy()
        m = v[rng.integers(0, len(v), size=(1000, len(v)))].mean(1)
        lo.append(np.percentile(m, 2.5)); hi.append(np.percentile(m, 97.5))
    nm = per.groupby(level=0).mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(nm.index, nm.values - nm.mean(), width=8.5, color="0.6",
           yerr=[nm.values - np.array(lo), np.array(hi) - nm.values], error_kw=dict(lw=0.6, ecolor="0.3"))
    ax.axvspan(CAMPUS_FROM - HW, CAMPUS_FROM + HW, color="C3", alpha=0.2, label="campus sector (111 ± 15°)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="Wind direction at KGYR (deg, from)", ylabel="ΔT − mean over bins (K)",
           xlim=(0, 360), xticks=range(0, 361, 45), title=title)
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_signature_{tag}.png", dpi=200); plt.close(fig)


def by_year(d, tag):
    rows = []
    for y in range(2015, 2026):
        x = d[(d.index + pd.Timedelta(hours=-7)).year == y]
        r = sector_test(x, commissioned=None, seed=y) if len(x) else dict(sector_nights=0)
        rows.append(dict(year=y, sector_nights=r.get("sector_nights", 0), effect=r.get("effect", np.nan),
                         ci_lo=r.get("ci", [np.nan])[0], ci_hi=r.get("ci", [np.nan, np.nan])[1]))
    t = pd.DataFrame(rows); t.to_csv(f"{OUT}/by_year_{tag}.csv", index=False)
    return t


def main():
    cov = coverage()
    runs = {}
    windows = {"spec 22-05": (22, 5), "evening 20-21": (20, 21), "evening 21": (21, 21)}
    for ref in ("LUF", "BXK"):
        for wl, w in windows.items():
            runs[f"{ref} | {wl} | hw15 from111"] = sector_test(screened("GYR", ref, w))
        ev = (20, 21)
        for hw in (10, 20):
            runs[f"{ref} | evening 20-21 | hw{hw} from111"] = sector_test(screened("GYR", ref, ev), hw=hw)
        for cf in (100, 120):
            runs[f"{ref} | evening 20-21 | hw15 from{cf}"] = sector_test(screened("GYR", ref, ev), campus_from=cf)
        runs[f"{ref} | evening 20-21 | hw15 from111 | deseasonalized"] = sector_test(screened("GYR", ref, ev, deseason=True))
        runs[f"{ref} | evening 20-21 | hw15 from111 | U 0.5-4"] = sector_test(screened("GYR", ref, ev, umin=0.5))
        runs[f"{ref} | morning 06 (05:47 obs), Nov-Feb | hw15 from111"] = sector_test(
            screened("GYR", ref, (6, 6), months=[11, 12, 1, 2]))
        d = screened("GYR", ref, ev)
        signature_plot(d, f"GYR_{ref}_evening",
                       f"KGYR − K{ref}, 20–21 MST, wind 1–4 m/s, 2015–2025")
        by_year(d, f"GYR_{ref}_evening")
    harm = {}
    for ref in ("LUF", "BXK"):
        d = screened("GYR", ref, (20, 21))
        for hw, cf in ((15, 111), (10, 111), (20, 111), (15, 100), (15, 120)):
            harm[f"{ref} | evening 20-21 | hw{hw} from{cf}"] = harmonic_test(d, campus_from=cf, hw=hw)
    with open(f"{OUT}/harmonic_runs.json", "w") as f:
        json.dump(harm, f, indent=1)
    ht = pd.DataFrame([dict(run=k, sector_nights_pre=v["sector_nights_pre"], sector_nights_post=v["sector_nights_post"],
                            pre=f"{v['sector_pre']:+.3f} [{v['sector_pre_ci'][0]:+.3f}, {v['sector_pre_ci'][1]:+.3f}]",
                            post=f"{v['sector_post']:+.3f} [{v['sector_post_ci'][0]:+.3f}, {v['sector_post_ci'][1]:+.3f}]",
                            did=f"{v['did']:+.3f} [{v['did_ci'][0]:+.3f}, {v['did_ci'][1]:+.3f}]")
                       for k, v in harm.items()])
    ht.to_csv(f"{OUT}/harmonic_runs.csv", index=False)
    with open(f"{OUT}/station_runs.json", "w") as f:
        json.dump(runs, f, indent=1)
    rows = []
    for k, r in runs.items():
        row = dict(run=k, hours=r["hours"], nights=r["nights"], sector_hours=r["sector_hours"],
                   sector_pct_hours=round(r["sector_pct_hours"], 2), sector_nights=r["sector_nights"],
                   effect_all=fmt(r))
        if "did" in r:
            row.update(pre=f"{r['pre']['effect']:+.3f} [{r['pre']['ci'][0]:+.3f}, {r['pre']['ci'][1]:+.3f}] (n={r['pre']['sector_nights']})",
                       post=f"{r['post']['effect']:+.3f} [{r['post']['ci'][0]:+.3f}, {r['post']['ci'][1]:+.3f}] (n={r['post']['sector_nights']})",
                       did=f"{r['did']:+.3f} [{r['did_ci'][0]:+.3f}, {r['did_ci'][1]:+.3f}]")
        rows.append(row)
    tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/station_runs.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(cov.round(2).T.to_string())
    print(tab.to_string(index=False))
    print(ht.to_string(index=False))
    print(prediction().to_string(index=False))


if __name__ == "__main__":
    main()
