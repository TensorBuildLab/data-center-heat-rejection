"""
satellite_analysis.py - ECOSTRESS night LST test (validation V5) for the Goodyear cluster.

Inputs : data/appeears/goodyear_{A,B,C}_*/ (or $SAT_DATA) AppEEARS ECO_L2T_LSTE.002 GeoTIFFs
         (LST, QC, cloud; time in the file name, UTC) + lookup CSVs;
         hourly winds from IEM ASOS (asos_wind_signature.fetch cache).
Method : Validation Plan 6.2.
  1. night scenes 21:00-06:00 MST; drop 2025-05-15..2025-07-01; mask cloud and bad QC;
     keep scenes with enough clear pixels near the campus.
  2. wind at overpass time from KLUF (vector-interpolated between hourly reports);
     1-4 m/s and direction steady within 30 deg over the bracketing reports.
  3. per-scene anomaly = LST - scene median over the box; fixed geographic pattern
     (per-pixel mean anomaly within each period) removed; rotate to wind coordinates
     centred on Stream PHXA1 (+x downwind); downwind-minus-upwind sector contrast
     (+-22.5 deg) in rings 0.5-1, 1-2, 2-3, 3-5 km from the footprint edge;
     bootstrap over scenes; before vs after commissioning (2020-10-14);
     random wind directions at the campus give the empirical null (placebo).
  4. dc_plume for each kept scene's meteorology, sampled at the same pixels, for
     the SHAPE (decay with distance, lateral width), not the magnitude.
Outputs: results/satellite/
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform as warp_transform
import sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path[:0] = [_HERE, _os.path.dirname(_HERE)]   # this folder and the repo root (dc_plume.py)
import asos_wind_signature as aws
import dc_plume as dp

OUT = os.environ.get("SAT_OUT", "results/satellite")
DATA = os.environ.get("SAT_DATA", "data/appeears")
os.makedirs(OUT, exist_ok=True)

# Stream PHXA1 building (OpenStreetMap way 468707940, "Stream Data Centers PHX-1",
# 2950 S Litchfield Rd): centre and E-W / N-S half-sizes in metres
CAMPUS = dict(lat=33.41960, lon=-112.36032, hx=154.0, hy=67.0, name="Stream PHXA1")
# Microsoft Goodyear (OSM PHX-10..14, 14250 W Broadway Rd): 5 halls of 393 x 58 m
MSFT = dict(lat=33.41047, lon=-112.36472, hx=197.0, hy=145.0, name="Microsoft PHX-10..14")
COMMISSIONED = pd.Timestamp("2020-10-14", tz="UTC")
DROP = (pd.Timestamp("2025-05-15", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"))
RINGS = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]     # km from footprint edge
SECTOR = 22.5
UMIN, UMAX = 1.0, 4.0
MIN_CLEAR = float(os.environ.get("SAT_MIN_CLEAR", 0.5))           # fraction of valid pixels required within 3 km of the campus
MIN_SECTOR_PIX = 5
N_BOOT = 2000
PERIODS = {"pre (2018-08 to 2020-10)": (None, COMMISSIONED),
           "post (2020-10 to 2026-10)": (COMMISSIONED, None)}
RX = re.compile(r"_(LST|QC|cloud)_doy(\d{13})_aid\d+", re.I)


# ---------------------------------------------------------------- scenes
def list_scenes():
    files = glob.glob(os.path.join(DATA, "*", "**", "*.tif"), recursive=True)
    sc = {}
    for f in files:
        m = RX.search(os.path.basename(f))
        if m:
            sc.setdefault(m.group(2), {})[m.group(1).upper()] = f
    rows = []
    for key, layers in sc.items():
        t = pd.to_datetime(key[:7], format="%Y%j").tz_localize("UTC") + pd.Timedelta(
            hours=int(key[7:9]), minutes=int(key[9:11]), seconds=int(key[11:13]))
        rows.append(dict(key=key, time=t, **{k.lower(): v for k, v in layers.items()}))
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def lookup_good_values():
    """Values of the QC and cloud layers that mark usable pixels, from the AppEEARS
    lookup CSVs. QC: mandatory QA 'best' or 'nominal' quality. cloud: 'not cloudy'."""
    good = {"QC": None, "CLOUD": None}
    for f in glob.glob(os.path.join(DATA, "*", "**", "*lookup*.csv"), recursive=True):
        t = pd.read_csv(f)
        name = os.path.basename(f).lower()
        cols = {c.lower(): c for c in t.columns}
        if "qc" in name and "lookup" in name:
            mq = [c for c in t.columns if "mandatory" in c.lower()][0]
            ok = t[mq].astype(str).str.strip().str.startswith("Pixel produced")   # best + nominal (degraded)
            good["QC"] = (good["QC"] or set()) | set(t.loc[ok, "Value"].astype(int))
        elif "cloud" in name and "lookup" in name:
            ok = t.iloc[:, 1].astype(str).str.strip().str.lower().eq("clear")
            good["CLOUD"] = (good["CLOUD"] or set()) | set(t.loc[ok, "Value"].astype(int))
    return good


_GRID = {}
def grid_xy(src):
    """East/north offsets (m) of every pixel centre from the campus, cached per grid."""
    key = (src.crs.to_string(), src.transform, src.shape)
    if key not in _GRID:
        rows, cols = np.indices(src.shape)
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
        lon, lat = warp_transform(src.crs, "EPSG:4326", xs, ys) if src.crs.is_projected else (xs, ys)
        lon, lat = np.asarray(lon).reshape(src.shape), np.asarray(lat).reshape(src.shape)
        _GRID[key] = (lon, lat)
    return _GRID[key]


def offsets(lon, lat, c):
    e = (lon - c["lon"]) * 111320.0 * np.cos(np.radians(c["lat"]))
    n = (lat - c["lat"]) * 111320.0
    return e, n


def _to_ref(arr, src_tr, ref, fill):
    """Nearest-pixel resampling of a geographic raster onto the reference grid."""
    if ref is None or (arr.shape == ref["shape"] and src_tr.almost_equals(ref["transform"])):
        return arr
    lon, lat = ref["lon"], ref["lat"]
    c = np.floor((lon - src_tr.c) / src_tr.a).astype(int)
    r = np.floor((lat - src_tr.f) / src_tr.e).astype(int)
    inside = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.full(ref["shape"], fill, dtype=float)
    out[inside] = arr[r[inside], c[inside]]
    return out


def read_scene(row, good, ref=None):
    with rasterio.open(row["lst"]) as s:
        lst = s.read(1).astype(float)
        nod, tr = s.nodata, s.transform
        lon, lat = grid_xy(s)
        scale = s.scales[0] if s.scales else 1.0
        offset = s.offsets[0] if s.offsets else 0.0
    bad = ~np.isfinite(lst) | (lst == nod if nod is not None and np.isfinite(nod) else False) | (lst <= 0)
    lst = lst * scale + offset
    for layer, ok in (("qc", good["QC"]), ("cloud", good["CLOUD"])):
        if isinstance(row.get(layer), str) and ok is not None:
            with rasterio.open(row[layer]) as s:
                v = s.read(1)
            if v.shape != lst.shape:
                bad |= True
                continue
            bad |= ~np.isin(v, list(ok))
    lst[bad] = np.nan
    lst[(lst < 200) | (lst > 350)] = np.nan
    if ref is not None:
        return _to_ref(lst, tr, ref, np.nan), ref["lon"], ref["lat"]
    return lst, lon, lat


# ---------------------------------------------------------------- wind at overpass
def wind_at(stations, times):
    """Vector-interpolated wind (speed m/s, from-direction deg) at each time from the
    hourly reports bracketing it; NaN when either bracket is missing or more than
    1.5 h away. 'steady' = bracketing directions within 30 deg."""
    out = {}
    for s in stations:
        w = aws.fetch(s, "2018-01-01", "2026-10-19")
        w = w[np.isfinite(w["U"])].copy()
        calm = (w["U"] == 0) | (w["drct"] == 0)
        th = np.radians(w["drct"].fillna(0))
        w["u"] = np.where(calm, 0.0, -w["U"] * np.sin(th))
        w["v"] = np.where(calm, 0.0, -w["U"] * np.cos(th))
        w = w[np.isfinite(w["drct"]) | calm]
        idx = w.index.values.astype("datetime64[s]").astype(np.int64)
        T = np.array([t.value // 10**9 for t in times])
        j = np.clip(np.searchsorted(idx, T), 1, len(idx) - 1)
        t0, t1 = idx[j - 1], idx[j]
        f = (T - t0) / np.maximum(t1 - t0, 1)
        ok = (t1 - t0 <= 5400) & (T >= t0) & (T <= t1)
        u = w["u"].values[j - 1] * (1 - f) + w["u"].values[j] * f
        v = w["v"].values[j - 1] * (1 - f) + w["v"].values[j] * f
        d0, d1 = w["drct"].values[j - 1], w["drct"].values[j]
        spd = np.hypot(u, v)
        dirn = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
        steady = (np.abs(aws.angdiff(d0, d1)) <= 30) & (d0 > 0) & (d1 > 0)
        out[s] = pd.DataFrame(dict(U=np.where(ok, spd, np.nan), dir=np.where(ok, dirn, np.nan),
                                   steady=ok & steady,
                                   T=np.where(ok, w["T"].values[j - 1] * (1 - f) + w["T"].values[j] * f, np.nan)),
                              index=times)
    return out


# ---------------------------------------------------------------- statistics
def edge_distance(e, n, c):
    return np.hypot(np.maximum(np.abs(e) - c["hx"], 0), np.maximum(np.abs(n) - c["hy"], 0)) / 1000.0


def ring_contrast(anom, e, n, c, wind_from, rings=RINGS):
    """Downwind-minus-upwind mean anomaly per ring; NaN where a sector has too few pixels."""
    dist = edge_distance(e, n, c)
    bearing_to = (np.degrees(np.arctan2(e, n)) + 360) % 360       # bearing of the pixel from the centre
    downwind = (wind_from + 180) % 360
    dd = np.abs(aws.angdiff(bearing_to, downwind))
    du = np.abs(aws.angdiff(bearing_to, wind_from))
    res = []
    for r0, r1 in rings:
        ring = (dist >= r0) & (dist < r1) & np.isfinite(anom)
        a, b = anom[ring & (dd <= SECTOR)], anom[ring & (du <= SECTOR)]
        res.append(a.mean() - b.mean() if a.size >= MIN_SECTOR_PIX and b.size >= MIN_SECTOR_PIX else np.nan)
    return np.array(res)


def rotated(e, n, wind_from):
    """Along-wind (+ downwind) and crosswind (+ left of downwind) coordinates, m."""
    th = np.radians((wind_from + 180) % 360)
    de, dn = np.sin(th), np.cos(th)
    return e * de + n * dn, -e * dn + n * de


def boot_mean(x, n=N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, [np.nan, np.nan], x.size
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, (n, x.size))].mean(1)
    return float(x.mean()), [float(v) for v in np.percentile(m, [2.5, 97.5])], int(x.size)


def boot_diff(x, y, n=N_BOOT, seed=1):
    x, y = [np.asarray(v, float)[np.isfinite(v)] for v in (x, y)]
    if x.size < 3 or y.size < 3:
        return np.nan, [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    d = x[rng.integers(0, x.size, (n, x.size))].mean(1) - y[rng.integers(0, y.size, (n, y.size))].mean(1)
    return float(x.mean() - y.mean()), [float(v) for v in np.percentile(d, [2.5, 97.5])]


# ---------------------------------------------------------------- model
def model_field(e, n, wind_from, U, T, it_mw=40.0):
    """dc_plume 2 m dT (K) at the pixel offsets for one scene's meteorology:
    class F (z_i 150 m) for U <= 2.5 m/s, else class E (z_i 300 m); air-cooled
    building, heat 1.37 x IT, 205 m square footprint (PHXA1 area), height 12.5 m."""
    import warnings
    stab, zi = ("F", 150.0) if U <= 2.5 else ("E", 300.0)
    amb = dp.Ambient(T=float(T) if np.isfinite(T) else 25.0, RH=0.2, U10=float(U), stability=stab,
                     mixing_height=zi, elevation=300.0)
    src = dp.dry_cooler_source(1.37 * it_mw * 1e6, 12.0, amb, height=12.5, footprint_x=205.0,
                               footprint_y=205.0, building_height=12.5)
    xa, yc = rotated(e, n, wind_from)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        r = dp.plume(src, amb, xa, yc)
    return np.where(np.isfinite(r["dT"]), r["dT"], np.nan)


# ---------------------------------------------------------------- composite map
BIN = 140.0
EDGES = np.arange(-5600.0, 5600.0 + BIN, BIN)


def composite(stack, E, N, winds):
    s = np.zeros((EDGES.size - 1,) * 2)
    c = np.zeros_like(s)
    for a, w in zip(stack, winds):
        xa, yc = rotated(E, N, w)
        ok = np.isfinite(a)
        h, _, _ = np.histogram2d(xa[ok], yc[ok], bins=[EDGES, EDGES], weights=a[ok])
        k, _, _ = np.histogram2d(xa[ok], yc[ok], bins=[EDGES, EDGES])
        s += h
        c += k
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c, c


# ---------------------------------------------------------------- main
def load(log):
    sc = list_scenes()
    log["files_scenes"] = len(sc)
    log["granules_without_LST"] = int(sc["lst"].isna().sum())
    sc = sc[sc["lst"].notna()].reset_index(drop=True)
    log["layers_present"] = {k: int(sc[k].notna().sum()) for k in ("lst", "qc", "cloud") if k in sc}
    loc = sc["time"] - pd.Timedelta(hours=7)
    night = (loc.dt.hour >= 21) | (loc.dt.hour < 6)
    drop = (sc["time"] >= DROP[0]) & (sc["time"] < DROP[1])
    log["night_scenes"] = int(night.sum())
    log["dropped_2025_window"] = int((night & drop).sum())
    sc = sc[night & ~drop].reset_index(drop=True)
    good = lookup_good_values()
    log["good_values"] = {k: sorted(map(int, v)) if v else None for k, v in good.items()}
    # reference grid = the most common LST grid
    import collections
    grids = collections.Counter()
    for f in sc["lst"]:
        with rasterio.open(f) as s:
            grids[(s.shape, tuple(s.transform)[:6])] += 1
    (shape, trt), n_ref = grids.most_common(1)[0]
    with rasterio.open(next(f for f in sc["lst"] if rasterio.open(f).shape == shape
                            and tuple(rasterio.open(f).transform)[:6] == trt)) as s:
        LON, LAT = grid_xy(s)
        ref = dict(shape=shape, transform=s.transform, lon=LON, lat=LAT)
    log["grids"] = {"reference": [list(shape), list(trt)], "scenes_on_reference": n_ref,
                    "scenes_resampled": int(sum(grids.values()) - n_ref)}
    E0, N0 = offsets(LON, LAT, CAMPUS)
    near = edge_distance(E0, N0, CAMPUS) <= 3.0
    stack, keep = [], []
    for i, row in sc.iterrows():
        lst, _, _ = read_scene(row, good, ref)
        frac = float(np.isfinite(lst[near]).mean())
        sc.loc[i, "clear_frac_3km"] = frac
        if frac < MIN_CLEAR:
            keep.append(False)
            continue
        stack.append(lst - np.nanmedian(lst))
        keep.append(True)
    sc = sc[np.array(keep)].reset_index(drop=True)
    log["scenes_clear"] = len(sc)
    return sc, np.array(stack), LON, LAT


def main():
    log = {}
    sc, stack, LON, LAT = load(log)
    E, N = offsets(LON, LAT, CAMPUS)
    Em, Nm = offsets(LON, LAT, MSFT)
    log["grid"] = dict(shape=list(LON.shape), pixel_m=float(np.nanmedian(np.abs(np.diff(E, axis=1)))))

    # fixed geographic pattern per period (pre / post commissioning), removed
    post = (sc["time"] >= COMMISSIONED).to_numpy()
    fixed = np.empty_like(stack)
    for m in (post, ~post):
        with np.errstate(all="ignore"):
            fixed[m] = np.nanmean(stack[m], axis=0)
    resid = stack - fixed

    winds = wind_at(["LUF", "PHX", "BXK"], list(sc["time"]))
    for s, w in winds.items():
        sc[f"U_{s}"], sc[f"dir_{s}"] = w["U"].values, w["dir"].values
        sc[f"steady_{s}"], sc[f"T_{s}"] = w["steady"].values, w["T"].values
    agree = np.abs(aws.angdiff(sc["dir_LUF"], sc["dir_PHX"])) <= 45
    rl, rp = np.radians(sc["dir_LUF"]), np.radians(sc["dir_PHX"])
    vecmean = (np.degrees(np.arctan2(np.sin(rl) + np.sin(rp), np.cos(rl) + np.cos(rp))) + 360) % 360
    spd = {st: sc[f"U_{st}"].between(UMIN, UMAX) for st in ("LUF", "PHX", "BXK")}
    std = {st: sc[f"steady_{st}"].astype(bool) for st in ("LUF", "PHX", "BXK")}
    # primary screen: speed only; the steadiness check (bracketing METAR directions
    # within 30 deg) is a sensitivity case because it removes ~70% of light-wind scenes
    sources = {"KLUF": (sc["dir_LUF"], spd["LUF"]),
               "KPHX": (sc["dir_PHX"], spd["PHX"]),
               "KBXK": (sc["dir_BXK"], spd["BXK"]),
               "KLUF steady": (sc["dir_LUF"], spd["LUF"] & std["LUF"]),
               "KPHX steady": (sc["dir_PHX"], spd["PHX"] & std["PHX"]),
               "KLUF+KPHX agree": (vecmean, spd["LUF"] & spd["PHX"] & agree)}

    rows, per_scene = [], {}
    for sname, (wdir, ok) in sources.items():
        ok = ok.fillna(False).to_numpy(bool)
        for cname, (e, n, c) in {"PHXA1": (E, N, CAMPUS), "Microsoft": (Em, Nm, MSFT)}.items():
            for field, arr in (("fixed pattern removed", resid), ("raw anomaly", stack)):
                con = np.full((len(sc), len(RINGS)), np.nan)
                for i in np.flatnonzero(ok):
                    con[i] = ring_contrast(arr[i], e, n, c, wdir.iloc[i])
                per_scene[(sname, cname, field)] = con
                for k, (r0, r1) in enumerate(RINGS):
                    for pname, m in (("all", ok), ("pre", ok & ~post), ("post", ok & post)):
                        est, ci, nn = boot_mean(con[m, k], seed=k)
                        rows.append(dict(wind=sname, centre=cname, field=field, ring=f"{r0}-{r1} km",
                                         period=pname, mean_K=est, lo=ci[0], hi=ci[1], scenes=nn))
                    d, dci = boot_diff(con[ok & post, k], con[ok & ~post, k], seed=10 + k)
                    rows.append(dict(wind=sname, centre=cname, field=field, ring=f"{r0}-{r1} km",
                                     period="post minus pre", mean_K=d, lo=dci[0], hi=dci[1],
                                     scenes=int(np.isfinite(con[ok, k]).sum())))
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT}/contrast_summary.csv", index=False, float_format="%.4f")

    # placebo: random wind directions at the real campus (null distribution of the
    # contrast when the wind label carries no information)
    rng = np.random.default_rng(42)
    plac_rows = []
    for sname, (wdir, okk) in sources.items():
        okk = okk.fillna(False).to_numpy(bool)
        idx = np.flatnonzero(okk)
        if idx.size < 3:
            continue
        obs_s = per_scene[(sname, "PHXA1", "fixed pattern removed")]
        null = []
        for _ in range(1000):
            th = rng.uniform(0, 360, idx.size)
            c = np.array([ring_contrast(resid[i], E, N, CAMPUS, t, rings=RINGS[:2]) for i, t in zip(idx, th)])
            pm = post[idx]
            null.append([np.nanmean(c[:, 0]), np.nanmean(c[:, 1]),
                         np.nanmean(c[pm, 0]) - np.nanmean(c[~pm, 0]) if pm.any() and (~pm).any() else np.nan])
        null = np.array(null)
        o = [np.nanmean(obs_s[okk, 0]), np.nanmean(obs_s[okk, 1]),
             np.nanmean(obs_s[okk & post, 0]) - np.nanmean(obs_s[okk & ~post, 0])]
        for j, lab in enumerate(("ring 0.5-1 km", "ring 1-2 km", "ring 0.5-1 km post minus pre")):
            nn = null[:, j][np.isfinite(null[:, j])]
            plac_rows.append(dict(wind=sname, statistic=lab, observed=o[j], null_mean=nn.mean() if nn.size else np.nan,
                                  null_p2_5=np.percentile(nn, 2.5) if nn.size else np.nan,
                                  null_p97_5=np.percentile(nn, 97.5) if nn.size else np.nan,
                                  p_one_sided=float((nn >= o[j]).mean()) if nn.size else np.nan))
    plac = pd.DataFrame(plac_rows)
    plac.to_csv(f"{OUT}/placebo_random_direction.csv", index=False, float_format="%.4f")
    wdir, ok = sources["KLUF"]
    ok = ok.fillna(False).to_numpy(bool)
    obs = per_scene[("KLUF", "PHXA1", "fixed pattern removed")]

    # model shape for the same post-commissioning scenes
    mcon, mstack, mw = [], [], []
    for i in np.flatnonzero(ok & post):
        f = model_field(E, N, wdir.iloc[i], sc["U_LUF"].iloc[i], sc["T_LUF"].iloc[i])
        f = np.where(np.isfinite(stack[i]), f, np.nan)
        mcon.append(ring_contrast(f, E, N, CAMPUS, wdir.iloc[i]))
        mstack.append(f)
        mw.append(wdir.iloc[i])
    model_ring = np.nanmean(np.array(mcon), axis=0) if mcon else np.full(len(RINGS), np.nan)
    obs_post = [boot_mean(obs[ok & post, k], seed=20 + k) for k in range(len(RINGS))]
    shape = pd.DataFrame(dict(ring=[f"{a}-{b} km" for a, b in RINGS],
                              obs_LST_contrast_K=[o[0] for o in obs_post],
                              obs_lo=[o[1][0] for o in obs_post], obs_hi=[o[1][1] for o in obs_post],
                              model_2m_dT_contrast_K=model_ring,
                              model_normalised=model_ring / model_ring[0]))
    shape.to_csv(f"{OUT}/shape_vs_model.csv", index=False, float_format="%.4f")

    # composites and lateral profile 0.5-2 km downwind
    pre_ok, post_ok = ok & ~post, ok & post
    comp_post, cnt_post = composite(resid[post_ok], E, N, wdir[post_ok])
    comp_pre, cnt_pre = composite(resid[pre_ok], E, N, wdir[pre_ok])
    comp_mod = composite(np.array(mstack), E, N, mw)[0] if mstack else np.full_like(comp_post, np.nan)
    np.savez_compressed(f"{OUT}/composites.npz", edges=EDGES, post=comp_post, pre=comp_pre,
                        model=comp_mod, n_post=cnt_post, n_pre=cnt_pre)
    xc = 0.5 * (EDGES[1:] + EDGES[:-1])
    band = (xc >= 500) & (xc <= 2000)
    with np.errstate(all="ignore"):
        prof = pd.DataFrame(dict(y_m=xc, obs_post=np.nanmean(comp_post[band], axis=0),
                                 obs_pre=np.nanmean(comp_pre[band], axis=0),
                                 model=np.nanmean(comp_mod[band], axis=0)))
    prof.to_csv(f"{OUT}/lateral_profile_0.5-2km.csv", index=False, float_format="%.4f")

    # figures
    lim = 5.0
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    allv = np.r_[comp_post[np.isfinite(comp_post)], comp_pre[np.isfinite(comp_pre)]]
    v = np.nanpercentile(np.abs(allv), 98) if allv.size else 1.0
    for a, data, ttl in ((ax[0], comp_pre, f"Before commissioning ({pre_ok.sum()} scenes)"),
                         (ax[1], comp_post, f"After commissioning ({post_ok.sum()} scenes)")):
        im = a.pcolormesh(EDGES / 1000, EDGES / 1000, data.T, cmap="RdBu_r", vmin=-v, vmax=v, shading="flat")
        a.set(title=ttl, xlabel="Downwind distance (km)", ylabel="Crosswind distance (km)",
              xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
        a.add_patch(plt.Rectangle((-0.15, -0.15), 0.3, 0.3, fill=False, ec="k"))
    fig.colorbar(im, ax=ax[:2], shrink=0.8, label="Night LST anomaly, fixed pattern removed (K)")
    im2 = ax[2].pcolormesh(EDGES / 1000, EDGES / 1000, comp_mod.T, cmap="Reds", shading="flat")
    ax[2].set(title="dc_plume 2 m ΔT, same scenes (40 MW IT)", xlabel="Downwind distance (km)",
              xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    fig.colorbar(im2, ax=ax[2], shrink=0.8, label="ΔT (K)")
    fig.savefig(f"{OUT}/fig_composite.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    mid = np.array([0.5 * (a + b) for a, b in RINGS])
    sel = summ[(summ.wind == "KLUF") & (summ.centre == "PHXA1") & (summ.field == "fixed pattern removed")]
    for p, off, col in (("pre", -0.06, "0.5"), ("post", 0.06, "C3")):
        s = sel[sel.period == p]
        ax.errorbar(mid + off, s.mean_K, yerr=[s.mean_K - s.lo, s.hi - s.mean_K], fmt="o", color=col,
                    capsize=3, label=f"LST, {p}-commissioning")
    ax2 = ax.twinx()
    ax2.plot(mid, model_ring, "k--", label="dc_plume 2 m ΔT (right axis)")
    ax2.set_ylabel("Model downwind − upwind ΔT (K)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="Distance from footprint edge (km)", ylabel="Downwind − upwind LST (K)",
           title="Stream PHXA1, night ECOSTRESS, KLUF wind")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ring_contrast.png", dpi=200)
    plt.close(fig)

    out = sc.drop(columns=[c for c in ("lst", "qc", "cloud") if c in sc])
    for k in range(len(RINGS)):
        out[f"contrast_ring{k + 1}_KLUF"] = obs[:, k]
    out.to_csv(f"{OUT}/scenes.csv", index=False)
    log["screened_scenes"] = {k: dict(all=int(v[1].fillna(False).sum()),
                                      pre=int((v[1].fillna(False) & ~post).sum()),
                                      post=int((v[1].fillna(False) & post).sum())) for k, v in sources.items()}
    h = pd.cut(sc.loc[ok, "dir_LUF"], np.arange(0, 361, 45)).value_counts().sort_index()
    log["wind_dir_hist_KLUF"] = {str(k): int(v) for k, v in h.items()}
    with open(f"{OUT}/run_log.json", "w") as f:
        json.dump(log, f, indent=1, default=float)
    pd.set_option("display.width", 220)
    print(json.dumps(log, indent=1, default=float))
    print(summ[summ.field == "fixed pattern removed"].round(3).to_string(index=False))
    print(shape.round(3).to_string(index=False))
    print(plac.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
