"""
make_epw.py - hourly dc_plume dT at fixed buildings near the reference campus, written
into modified Phoenix TMY3 weather files for the Section 5.1 building-energy demonstration.

Campus: Section 2.8 reference (100 MW IT air-cooled, 137 MW heat, 600 m x 600 m, fans at 12 m).
Buildings: 0.5 and 1 km beyond the downwind fence (centre distance 0.8 and 1.3 km), at the
bearing toward which the night wind most often blows in the TMY3 year.
Each hour: wind from the EPW; Pasquill class by the Turner method (wind speed, solar
elevation, total sky cover); mixing height 1500 (A-C), 800 (D), 300 (E), 150 m (F);
calm hours (U < 0.5 m/s): direction-averaged ('radial') value at U = 1 m/s.
Humidity ratio is unchanged (dry coolers add no vapour): dew point kept, RH recomputed.
Outputs: PHX_TMY3_dc{500,1000}m.epw, dT_hourly.csv, summary in dT_summary.json

Input PHX_TMY3.epw is the Phoenix Sky Harbor TMY3 file shipped with openstudio-standards. Only
dT_hourly.csv is used downstream: rewrite_epw.py applies it to the climate.onebuilding.org copy of
the same TMY3 year, whose .ddy/.stat set the design days. The two copies have identical dry-bulb,
wind and sky cover (the dT inputs) and differ only in dew point/RH (4 % of hours) and radiation,
so the PHX_TMY3_dc*.epw files written here are superseded and not used in the paper.
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dc_plume as dp

HERE = os.path.dirname(os.path.abspath(__file__))
LAT, LON, TZ = 33.45, -111.98, -7.0
lines = open(os.path.join(HERE, "PHX_TMY3.epw"), encoding="latin-1").read().splitlines()
head, data = lines[:8], [l.split(",") for l in lines[8:] if l.strip()]
cols = dict(month=1, day=2, hour=3, T=6, Td=7, RH=8, P=9, wdir=20, wspd=21, sky=22)
w = pd.DataFrame({k: [float(r[i]) for r in data] for k, i in cols.items()})

# solar elevation (approximate, local standard time, mid-hour)
doy = pd.to_datetime(dict(year=2001, month=w.month.astype(int), day=w.day.astype(int))).dt.dayofyear.values
decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
B = 2 * np.pi * (doy - 81) / 364
eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
solar_time = (w.hour.values - 0.5) + (4 * (LON - 15 * TZ) + eot) / 60
ha = np.radians(15 * (solar_time - 12))
elev = np.degrees(np.arcsin(np.sin(np.radians(LAT)) * np.sin(decl) + np.cos(np.radians(LAT)) * np.cos(decl) * np.cos(ha)))


def turner(U, el, sky):
    if el <= 0:                                   # night
        cloudy = sky >= 5
        if U < 3:
            return "E" if cloudy else "F"
        return "D" if (U >= 5 or cloudy) else "E"
    ins = 3 if el > 60 else 2 if el > 35 else 1 if el > 15 else 0     # strong..slight, 0 = low sun
    if sky >= 8:
        ins = min(ins, 1)
    if ins == 0:
        return "D"
    table = {3: ["A", "B", "B", "C", "C"], 2: ["A", "B", "C", "C", "D"], 1: ["B", "C", "C", "D", "D"]}
    k = 0 if U < 2 else 1 if U < 3 else 2 if U < 5 else 3 if U < 6 else 4
    return table[ins][k]


ZI = {"A": 1500.0, "B": 1500.0, "C": 1500.0, "D": 800.0, "E": 300.0, "F": 150.0}
stab = [turner(u, e, s) for u, e, s in zip(w.wspd, elev, w.sky)]
night = elev <= 0
calm = w.wspd.values < 0.5
# bearing toward which the night wind most often blows (wind 'to' = from + 180), 10-deg bins
to = (w.wdir[night & ~calm] + 180) % 360
bearing = float(pd.Series((np.round(to / 10) * 10) % 360).value_counts().idxmax())

res = {}
for dist in (500.0, 1000.0):
    r_c = 300.0 + dist                          # from campus centre
    dT = np.zeros(len(w))
    for i in range(len(w)):
        U = max(float(w.wspd[i]), 0.5)
        amb = dp.Ambient(T=float(w["T"][i]), RH=min(max(w.RH[i] / 100, 0.01), 1.0), U10=max(U, 1.0),
                         stability=stab[i], mixing_height=ZI[stab[i]], elevation=340.0)
        src = dp.dry_cooler_source(137e6, 12.0, amb, height=12.0, footprint_x=600.0,
                                   footprint_y=600.0, building_height=10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if calm[i]:
                v = dp.plume(src, amb, np.array([r_c]), mode="radial")["dT"][0]
            else:
                downwind = (w.wdir[i] + 180) % 360
                a = np.radians(bearing - downwind)
                v = dp.plume(src, amb, np.array([r_c * np.cos(a)]), np.array([r_c * np.sin(a)]))["dT"][0]
        dT[i] = v if np.isfinite(v) else 0.0
    res[int(dist)] = dT
    out = []
    for r, d in zip(data, dT):
        r = list(r)
        T_new = float(r[6]) + d
        Td = float(r[7])
        rh = 100 * dp.p_sat(Td) / dp.p_sat(T_new)
        r[6] = f"{T_new:.1f}"
        r[8] = f"{min(rh, 100):.0f}"
        out.append(",".join(r))
    with open(os.path.join(HERE, f"PHX_TMY3_dc{int(dist)}m.epw"), "w", encoding="latin-1", newline="\r\n") as f:
        f.write("\n".join(head + out) + "\n")

hourly = pd.DataFrame(dict(month=w.month, day=w.day, hour=w.hour, T=w["T"], wdir=w.wdir, wspd=w.wspd,
                           stability=stab, night=night, dT_500m=res[500], dT_1000m=res[1000]))
hourly.to_csv(os.path.join(HERE, "dT_hourly.csv"), index=False, float_format="%.4f")
summ = {"bearing_deg": bearing,
        "stability_counts": pd.Series(stab).value_counts().to_dict()}
for d in (500, 1000):
    x = res[d]
    summ[f"{d}m"] = dict(mean_all=float(x.mean()), mean_night=float(x[night].mean()), mean_day=float(x[~night].mean()),
                         max=float(x.max()), p95=float(np.percentile(x, 95)),
                         hours_gt_0p1=int((x > 0.1).sum()), hours_gt_0p3=int((x > 0.3).sum()),
                         degree_hours_K=float(x.sum()))
json.dump(summ, open(os.path.join(HERE, "dT_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
