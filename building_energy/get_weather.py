"""
get_weather.py - download the Phoenix Sky Harbor TMY3 weather file (EPW + DDY + STAT) from
Climate.OneBuilding.Org into data/weather/. Used by analysis/exceedance.py and building_energy/.
TMY3 data: Wilcox & Marion (2008), NREL/TP-581-43156, doi:10.2172/928611.

Run from the repo root:  python building_energy/get_weather.py
"""
import io, os, zipfile
import requests

URL = ("https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/"
       "USA_United_States_of_America/AZ_Arizona/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.zip")
OUT = os.path.join("data", "weather")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            if name.endswith((".epw", ".ddy", ".stat")):
                z.extract(name, OUT)
                print("wrote", os.path.join(OUT, name))
