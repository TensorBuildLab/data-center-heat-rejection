"""
asos_wind_signature.py  -  wind-direction signature of data-center warming at public ASOS stations
(Section 3.4 of the manuscript; feasibility test with public data only).

What it does
  1. Downloads routine hourly METARs (temperature, wind) for a target station near a data center
     and a reference station outside its influence, from the Iowa Environmental Mesonet (IEM).
  2. Keeps nighttime hours with light, steady-enough wind at the target (the regime where the model
     predicts a signal).
  3. Computes dT = T_target - T_reference, binned by the target's wind direction, with block-bootstrap
     (by night) confidence intervals.
  4. Tests the campus sector (wind blowing from the campus toward the station) against all other
     directions, optionally before vs after commissioning (difference in differences).

Usage (needs internet; IEM is public, no login):
  python asos_wind_signature.py --target GYR --ref LUF --campus-from 111 --halfwidth 15 \
         --start 2015-01-01 --end 2026-01-01 --commissioned 2021-06-01
  --campus-from : wind direction (deg, meteorological "from") that carries the plume onto the target
  --halfwidth   : sector half-width (deg); the model predicts ~10-15 deg at 1-2 km
Requires: pandas, numpy, matplotlib, requests
"""
import argparse, io
import numpy as np, pandas as pd

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

def fetch(station, start, end, cache_dir="data/asos"):
    """Download routine METARs from IEM one calendar year at a time (large single
    requests can time out), caching each year as CSV under cache_dir."""
    import os, requests
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    os.makedirs(cache_dir, exist_ok=True)
    parts = []
    for yr in range(s.year, e.year + 1):
        a, b = max(s, pd.Timestamp(yr, 1, 1)), min(e, pd.Timestamp(yr + 1, 1, 1))
        if a >= b:
            continue
        fn = os.path.join(cache_dir, f"{station}_{a:%Y%m%d}_{b:%Y%m%d}.csv")
        if not os.path.exists(fn):
            params = [("station", station), ("data", "tmpf"), ("data", "drct"), ("data", "sknt"),
                      ("year1", a.year), ("month1", a.month), ("day1", a.day),
                      ("year2", b.year), ("month2", b.month), ("day2", b.day),
                      ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
                      ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3")]
            r = requests.get(IEM, params=params, timeout=300)
            r.raise_for_status()
            if not r.text.startswith("station"):
                raise RuntimeError(f"IEM returned unexpected content for {station} {yr}: {r.text[:200]}")
            with open(fn, "w", encoding="utf-8") as f:
                f.write(r.text)
        with open(fn, encoding="utf-8") as f:
            parts.append(parse(f.read()))
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()

def parse(text):
    df = pd.read_csv(io.StringIO(text), na_values=["M"])
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["hour"] = df["valid"].dt.round("h")          # routine METAR at hh:5x -> nearest hour
    df = df.groupby("hour").last()
    df["T"] = (df["tmpf"] - 32) / 1.8
    df["U"] = df["sknt"] * 0.5144
    return df[["T", "drct", "U"]]

def prepare(tgt, ref, utc_offset=-7, night=(22, 5), umin=1.0, umax=4.0):
    d = tgt.join(ref[["T"]].rename(columns={"T": "Tref"}), how="inner").dropna()
    loc = d.index + pd.Timedelta(hours=utc_offset)
    h = loc.hour
    # local-hour window [night[0], night[1]], wrapping past midnight when night[0] > night[1]
    keep = ((h >= night[0]) & (h <= night[1])) if night[0] <= night[1] else            ((h >= night[0]) | (h <= night[1]))
    d = d[keep].copy()
    d["night"] = (loc[keep] - pd.Timedelta(hours=12)).normalize()
    d = d[(d["U"] >= umin) & (d["U"] <= umax) & (d["drct"] > 0)]
    d["dT"] = d["T"] - d["Tref"]
    return d

def angdiff(a, b):
    return (np.asarray(a) - b + 180) % 360 - 180

def block_boot(d, stat, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    nights = d["night"].unique()
    g = {k: v for k, v in d.groupby("night")}
    vals = []
    for _ in range(n):
        pick = rng.choice(nights, size=len(nights), replace=True)
        vals.append(stat(pd.concat([g[k] for k in pick])))
    return np.nanpercentile(vals, [2.5, 97.5])

def sector_effect(d, campus_from, hw):
    ins = np.abs(angdiff(d["drct"], campus_from)) <= hw
    def stat(x):
        i = np.abs(angdiff(x["drct"], campus_from)) <= hw
        # night means first, so each night counts once
        a = x[i].groupby("night")["dT"].mean().mean()
        b = x[~i].groupby("night")["dT"].mean().mean()
        return a - b
    est = stat(d)
    ci = block_boot(d, stat)
    return est, ci, int(d[ins]["night"].nunique()), int(d[~ins]["night"].nunique())

def binned(d, width=10):
    b = (np.floor(d["drct"] / width) * width + width / 2) % 360
    return d.groupby(b)["dT"].agg(["mean", "count"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True); p.add_argument("--ref", required=True)
    p.add_argument("--campus-from", type=float, required=True)
    p.add_argument("--halfwidth", type=float, default=15)
    p.add_argument("--start", default="2015-01-01"); p.add_argument("--end", default="2026-01-01")
    p.add_argument("--commissioned", default=None)
    p.add_argument("--night", default="22-5",
                   help="local-standard-time hour window, start-end (default 22-5)")
    p.add_argument("--outdir", default=".")
    p.add_argument("--tag", default="", help="suffix for output file names")
    a = p.parse_args()
    import os
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, f"signature_{a.target}_{a.ref}{a.tag}")
    nw = tuple(int(v) for v in a.night.split("-"))
    d = prepare(fetch(a.target, a.start, a.end), fetch(a.ref, a.start, a.end), night=nw)
    print(f"{len(d)} night hours, {d['night'].nunique()} nights after screening")
    if d.empty or d["night"].nunique() < 2:
        print("too few screened hours for a sector test (check the target's reporting hours)")
        return
    if a.commissioned:
        c = pd.Timestamp(a.commissioned, tz="UTC")
        pre, post = d[d.index < c], d[d.index >= c]
        for lab, x in (("before", pre), ("after", post)):
            if len(x):
                e, ci, n1, n2 = sector_effect(x, a.campus_from, a.halfwidth)
                print(f"{lab:6s}: sector minus other = {e:+.3f} K (95% CI {ci[0]:+.3f} to {ci[1]:+.3f}); "
                      f"{n1} sector nights, {n2} other nights")
    e, ci, n1, n2 = sector_effect(d, a.campus_from, a.halfwidth)
    print(f"all   : sector minus other = {e:+.3f} K (95% CI {ci[0]:+.3f} to {ci[1]:+.3f}); {n1} / {n2} nights")
    tab = binned(d)
    tab.to_csv(stem + ".csv")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(tab.index, tab["mean"], width=9, color="0.6")
        ax.axvspan(a.campus_from - a.halfwidth, a.campus_from + a.halfwidth, color="C3", alpha=0.2,
                   label="campus sector")
        ax.set(xlabel="Wind direction at target (deg, from)", ylabel="T_target - T_ref (K)",
               title=f"{a.target} minus {a.ref}, stable-night hours"); ax.legend()
        fig.tight_layout(); fig.savefig(stem + ".png", dpi=200)
    except Exception as ex:
        print("plot skipped:", ex)

if __name__ == "__main__":
    main()
