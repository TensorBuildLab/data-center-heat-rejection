"""analyze.py - compare the baseline and dT-perturbed MidriseApartment runs (hourly meters from eplusout.sql)."""
import os, sqlite3, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def meter(sql, name):
    con = sqlite3.connect(sql)
    q = """SELECT t.Month, t.Day, t.Hour, d.Value FROM ReportData d
           JOIN ReportDataDictionary k ON d.ReportDataDictionaryIndex = k.ReportDataDictionaryIndex
           JOIN Time t ON d.TimeIndex = t.TimeIndex
           WHERE k.Name = ? AND k.ReportingFrequency = 'Hourly' AND (t.WarmupFlag IS NULL OR t.WarmupFlag = 0)
           AND t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay') ORDER BY d.TimeIndex"""
    df = pd.read_sql_query(q, con, params=(name,))
    con.close()
    return df


dT = pd.read_csv(os.path.join(HERE, "dT_hourly.csv"))
rows, series = [], {}
for case in ("base", "ctrl0", "dc500m", "dc1000m"):
    sql = os.path.join(HERE, "sim", f"{case}.sql")
    ce = meter(sql, "Cooling:Electricity")
    cc = meter(sql, "CoolingCoils:EnergyTransfer")
    fe = meter(sql, "Electricity:Facility")
    assert len(ce) == 8760, (case, len(ce))
    hrs = ce["Hour"].values
    night = (hrs >= 21) | (hrs <= 6)          # hour ending 21..6 local
    series[case] = dict(ce=ce.Value.values / 3.6e6, cc=cc.Value.values / 3.6e6, fe=fe.Value.values / 3.6e6)
    s = series[case]
    rows.append(dict(case=case,
                     cooling_elec_MWh=s["ce"].sum() / 1000, cooling_elec_night_MWh=s["ce"][night].sum() / 1000,
                     coil_load_MWh=s["cc"].sum() / 1000,
                     peak_cooling_elec_kW=s["ce"].max(), peak_coil_load_kW=s["cc"].max(),
                     peak_facility_kW=s["fe"].max(), facility_MWh=s["fe"].sum() / 1000))
tab = pd.DataFrame(rows).set_index("case")
for c in ("ctrl0", "dc500m", "dc1000m"):
    for k in tab.columns:
        tab.loc[f"{c} change %", k] = 100 * (tab.loc[c, k] / tab.loc["base", k] - 1)
# hour-of-peak, and the largest hourly increase in cooling electricity
for c in ("ctrl0", "dc500m", "dc1000m"):
    d = series[c]["ce"] - series["base"]["ce"]
    tab.loc[c, "max_hourly_increase_kW"] = d.max()
tab.round(3).to_csv(os.path.join(HERE, "energy_demo_results.csv"))
print(tab.round(3).T.to_string())
