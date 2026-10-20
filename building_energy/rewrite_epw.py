"""rewrite_epw.py - apply dT_hourly.csv to the climate.onebuilding.org Phoenix TMY3 EPW (the file whose
.ddy/.stat set the design days), changing only dry-bulb (field 6) and RH (field 8, recomputed at constant
dew point). A zero-dT control file checks that the rewrite itself changes nothing."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dc_plume as dp

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "dl", "phx", "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw")
if not os.path.exists(BASE):  # same file as downloaded by python building_energy/get_weather.py
    BASE = os.path.join(HERE, "..", "data", "weather", "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw")
lines = open(BASE, encoding="latin-1").read().splitlines()
head, data = lines[:8], lines[8:]
TAG = os.environ.get("DC_TAG", "")
dT = pd.read_csv(os.path.join(HERE, f"dT_hourly{TAG}.csv"))
assert len(dT) == len(data) == 8760
for name, col in (("ctrl0", None), ("dc500m", "dT_500m"), ("dc1000m", "dT_1000m")):
    out = []
    for i, l in enumerate(data):
        r = l.split(",")
        assert int(r[1]) == dT.month[i] and int(r[2]) == dT.day[i] and int(r[3]) == dT.hour[i]
        d = 0.0 if col is None else float(dT[col][i])
        T = float(r[6]) + d
        rh = 100 * dp.p_sat(float(r[7])) / dp.p_sat(T)
        r[6] = f"{T:.2f}"
        if col is not None:
            r[8] = f"{min(rh, 100):.0f}"
        out.append(",".join(r))
    with open(os.path.join(HERE, f"PHX2{TAG}_{name}.epw"), "w", encoding="latin-1", newline="\r\n") as f:
        f.write("\n".join(head + out) + "\n")
print("written")
