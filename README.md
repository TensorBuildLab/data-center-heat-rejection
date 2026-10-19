# data-center-heat-rejection

Code for the manuscript *Neighborhood-scale air temperature and humidity perturbations downwind of data center heat rejection* (Liang Zhang, University of Arizona; in preparation for *Energy and Buildings*).

The core is **`dc_plume.py`**, a reduced-order model of the hourly 2 m air-temperature rise (ΔT) and humidity change (Δq, ΔRH) downwind of a data center's heat-rejection plant: dry coolers (air-cooled) or cooling towers (water-cooled). The other scripts produce the paper's figures, test the model against public data (weather stations and ECOSTRESS satellite land-surface temperature), estimate what each data source can detect, and feed the hourly ΔT into an EnergyPlus building model.

```
dc_plume.py                 the model (single file, NumPy only)
test_dc_plume.py            code-verification tests T1–T12 (pytest)
docs/technical_reference.md full equations, assumptions, validity range, verification ledger
paper_figures/              Figures 3–6 and the sensitivity / stability / campus-size tables
validation/                 station test, satellite test, detectability analysis
building_energy/            hourly ΔT → modified weather file → EnergyPlus comparison
```

## Install

Python 3.9 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

`dc_plume.py` itself needs only NumPy (SciPy is used for `erf` when installed). Run every script **from the repository root**; outputs go to `figures/`, `results/` and `data/`, which are git-ignored.

## The model: `dc_plume.py`

A steady-state (1-hour average) Gaussian plume adapted to large, low-velocity, buoyant heat sources. For each hour it:

1. **Computes the source terms from the exhaust state** (`source_terms`): sensible heat Q_s, latent heat Q_l, vapor flux E and Briggs buoyancy flux follow from the exhaust airflow, temperature and humidity, so no separate power input is needed. Helper constructors build the exhaust state from the heat rejected:
   - `dry_cooler_source(Q_W, dT_air, amb, ...)`: dry coolers. The airflow carries Q_W at a fixed air-side temperature rise, and no moisture is added.
   - `cooling_tower_source(Q_W, inlet_flow, amb, ...)`: cooling towers. The exhaust temperature is solved from the tower energy balance with a near-saturated exhaust (RH_exit 0.95), so it responds to the ambient wet-bulb.
2. **Treats the campus as an area source, a point source, or a blend of the two** (`regime="auto"`):
   - *Area treatment.* The rejected heat mixes a convective layer over the footprint whose depth h_mix is the smallest of three limits: encroachment against the stable gradient, the convective-velocity limit, and the mixing height (`convective_mixing_depth`). It is released as a crosswind strip at the downwind edge.
   - *Point treatment.* Briggs buoyant or momentum plume rise from the footprint centre, with partial penetration of the capping inversion.
   - *Blending.* The weight is w = R⁴/(1+R⁴), with R = footprint width / max(h_mix, fan height, point rise). Large campuses behave as area sources, small ones as point sources.
3. **Disperses the heat** using:
   - lateral spread from the hourly wind-direction fluctuation σθ with Draxler's travel-time factor, so hourly meander is included;
   - exact crosswind integration of the source width (erf profile);
   - Pasquill–Gifford/Briggs vertical spread with image reflections at the ground and the mixing height;
   - on stable nights, heat loss to the surface and by longwave radiation, applied as Chamberlain–Horst source depletion with transfer velocity v_h = C_H·U₁₀ + v_rad.
4. **Returns ΔT, Δq, T, q and RH** at the receptor height, along a line (plume axis or wind-direction average) or over a 2-D field, plus diagnostics (h_mix, area weight, rise, σy, σz, heat-loss factor, source terms). It warns when the exhaust/ambient mixing line or the receptor exceeds saturation, which indicates a visible plume or fog.

```python
import numpy as np
from dc_plume import Ambient, dry_cooler_source, cooling_tower_source, plume

night = Ambient(T=32.0, RH=0.20, U10=2.0, stability="F", mixing_height=150.0, elevation=340.0)
air   = dry_cooler_source(137e6, 12.0, night, height=12.0, footprint_x=600.0,
                          footprint_y=600.0, building_height=10.0)      # 100 MW IT, air-cooled
water = cooling_tower_source(125e6, 1650.0, night, height=15.0, footprint_x=300.0,
                             footprint_y=200.0, building_height=10.0)  # 100 MW IT, water-cooled

x = np.linspace(50, 5000, 400)                       # m downwind of the campus centre
r = plume(air, night, x)                             # plume axis, one wind direction
print(np.interp([1000, 5000], x, r["dT"]))           # -> [0.486 0.161] K
m = plume(air, night, x[:, None], np.linspace(-2000, 2000, 201)[None, :])   # 2-D map
radial = plume(air, night, x, mode="radial")         # average over all wind directions
```

Main options of `plume()`:

| Option | Values |
| --- | --- |
| `mode` | `"centerline"` (one wind direction; x along-wind, y crosswind) or `"radial"` (average over uniformly distributed wind directions) |
| `regime` | `"auto"` (blend), `"area"` or `"point"` |
| `lateral` | `"sigma_theta"` (hourly σθ, default; set `Ambient.sigma_theta` from measurements when available) or `"pg"` (Pasquill–Gifford 10-min σy) |
| `alongwind_strips` | 1 = release at the downwind edge (upper near-field estimate); 2–5 = Gauss–Legendre strips over the footprint (lower estimate) |
| `z` | receptor height (default 2 m, weather-station height) |

Key assumptions: steady state within the hour; horizontally homogeneous atmosphere; flat terrain; excess heat and vapor are passive after rise and mixing; wind below 1 m/s is clipped to 1 m/s. `docs/technical_reference.md` gives every equation (Section 5), the inputs and outputs (Sections 3–4), the validity range (Section 6), a worked Phoenix example (Section 7), and a ledger of every defect found and fixed during development and review (Section 8).

## Tests

```bash
python -m pytest -q          # 21 tests, about 1–2 minutes
```

| Test | What it checks | Criterion |
| --- | --- | --- |
| T1 | Heat-flux conservation through a cross-section at 2 km, no heat loss (area and point; classes B, D, F) | relative error < 1e-6 |
| T2 | Heat budget with loss on: transported fraction equals the reported depletion factor | < 1e-6 |
| T3 | Crosswind integral of the centerline field / 2πx equals the radial (direction-averaged) value | < 1e-6 |
| T4 | Far-field well-mixed limit Q_s/(ρ c_p U z_i) × lateral factor | < 1e-6 |
| T5 | Continuity between the image sum and the well-mixed limit at σz = 1.6 z_i | < 1e-4 |
| T6 | Cooling-tower energy balance: Q_s + Q_l equals the requested heat | < 0.5 % |
| T7 | Area/point blend for footprints of 60–600 m: ΔT at 1 km is monotonic, with no jumps | local log-log slope within [0, 2.5] |
| T8 | Along-wind strips converge (n = 2…5) | identical to 1e-3 K |
| T9 | Momentum-rise branch is selected for a fast, weakly buoyant jet | momentum rise used |
| T10 | Robustness: NaN inputs, negative x, lowercase class, out-of-range RH, impossible tower load | NaN / warning / ValueError as documented |
| T11 | Mixing-line saturation check for tower exhaust: summer max RH 0.950, winter 1.002 | warning only in winter |
| T12 | Reference case: air-cooled 0.486 / 0.161 K and water-cooled 0.052 / 0.009 K at 1 / 5 km | within 1e-3 K |

T12 guards against unintended changes in results. If a deliberate model change alters it, update the expected values and record the change in the technical reference's verification ledger.

## Paper figures: `paper_figures/`

All four scripts share `reference_case.py`: the reference night (Phoenix, class F, U₁₀ = 2 m/s, z_i = 150 m, 32 °C, 20 % RH) and the 100 MW (IT) air-cooled and water-cooled campuses.

| Script | Produces |
| --- | --- |
| `fig3_fig4_profiles_maps.py` | Fig. 3: plume-axis ΔT and ΔRH profiles, including the edge-release-to-strips band and the point-plume lower bound. Fig. 4: 2-D ΔT and ΔRH maps |
| `fig5_sensitivity.py` | One-at-a-time sensitivity of ΔT at 1 and 5 km to σθ, dθ/dz, the entrainment constant, U₁₀, the blend exponent, along-wind treatment, heat load, C_H and z_i (Fig. 5, tornado chart). Also prints the stability × wind-speed matrix at 1 km for both technologies |
| `fig6_campus_scale.py` | ΔT 0.5 and 3 km beyond the fence for 25–400 MW campuses at constant heat flux per area (Fig. 6). Also prints h_mix and fixed-distance values |

```bash
python paper_figures/fig3_fig4_profiles_maps.py
python paper_figures/fig5_sensitivity.py
python paper_figures/fig6_campus_scale.py
```

The manuscript figures were rendered with matplotlib 3.6. With that version these scripts reproduce them to within anti-aliasing (under 0.6 % of pixels differ). Newer matplotlib versions shift layout by a few pixels, but the data are unchanged.

## Validation with public data: `validation/`

These scripts test whether the predicted warming can be seen around a real campus, Stream Data Centers PHXA1 in Goodyear, Arizona (air-cooled, commissioned 2020-10-14), using only public data.

### Station test: `asos_wind_signature.py`, `station_analysis.py`

`asos_wind_signature.py` is a general tool:
1. It downloads hourly METARs for a target station near a campus and a reference station from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml), with no login. Each year is cached in `data/asos/`.
2. It keeps light-wind night hours.
3. It compares the target-minus-reference temperature when the wind blows from the campus sector against all other directions, using a block bootstrap by night. It can also compare before and after commissioning (difference in differences).

```bash
python validation/asos_wind_signature.py --target GYR --ref LUF --campus-from 111 --halfwidth 15 \
       --start 2015-01-01 --end 2026-01-01 --commissioned 2020-10-14 --outdir results/station
```

`station_analysis.py` runs the full test for KGYR (Phoenix Goodyear Airport, 1.3 km WNW of PHXA1) against KLUF and KBXK and writes everything to `results/station/`:
- reporting coverage by local hour (KGYR does not report overnight, so an evening window is used);
- sector frequency, sector effects and difference in differences;
- sensitivity to sector width and centre, reference station, hour window and deseasonalization;
- a harmonic-regression variant;
- the `dc_plume` prediction for the station.

### Satellite test: `satellite_analysis.py`

This tests whether night ECOSTRESS land-surface temperature shows a warm band that follows the wind downwind of the campus.

**Data:** ECOSTRESS `ECO_L2T_LSTE.002` (layers LST, QC and cloud) for the 12 × 12 km box in `validation/areas/area_goodyear.geojson`, ordered through [AppEEARS](https://appeears.earthdatacloud.nasa.gov/) (free NASA Earthdata login):
- choose an Area request, upload the GeoJSON, select GeoTIFF output in Geographic projection;
- split long periods into several requests, because AppEEARS caps the request size;
- unzip each request into `data/appeears/<name>/`, or point `SAT_DATA` at the folder.

**Method:**
1. Keep night scenes (21:00–06:00 MST), drop 2025-05-15 to 2025-07-01 (a known noisy period), and apply the cloud and QC masks.
2. Take the wind at overpass time from KLUF (also KPHX and KBXK), interpolated by vector.
3. Remove each scene's median and the fixed geographic pattern.
4. Rotate to wind-relative coordinates.
5. Compute the downwind-minus-upwind contrast in rings 0.5–1, 1–2, 2–3 and 3–5 km from the footprint edge, before and after commissioning, with a scene bootstrap and a random-direction placebo.
6. Compare the shape (decay and width) with `dc_plume` run for each scene's weather.

The script also repeats the analysis for the neighbouring Microsoft halls. Outputs go to `results/satellite/`.

```bash
SAT_DATA=path/to/appeears python validation/satellite_analysis.py
```

### Detectability: `detectability.py`

This compares the predicted signal with the smallest effect each data source can detect (95 % two-sided, 80 % power): the KGYR station record, ECOSTRESS night scenes, and a planned on-site campaign with assumed night-to-night scatter. The station and satellite scatter values are back-calculated from the confidence intervals of the two analyses above and entered at the top of the script. Outputs: `results/detectability/`.

## Building-energy demonstration: `building_energy/`

The model's hourly ΔT is fed into an EnergyPlus run of a DOE prototype building, to show what the warming means for cooling energy downwind.

1. **Weather:** download the Phoenix Sky Harbor TMY3 package (`USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3`, `.epw`/`.ddy`/`.stat`) from [climate.onebuilding.org](https://climate.onebuilding.org) into `building_energy/dl/phx/`.
2. **`make_epw.py`** computes the hourly ΔT (`dT_hourly.csv`) for buildings 0.5 and 1 km beyond the downwind fence of the 100 MW air-cooled reference campus. They sit at the bearing the night wind most often blows toward (west). Stability class comes from the Turner method; calm hours use the direction-averaged value. It reads `building_energy/PHX_TMY3.epw`, the same TMY3 year as shipped with [openstudio-standards](https://github.com/NREL/openstudio-standards); only `dT_hourly.csv` is used downstream (see the script header).
3. **`rewrite_epw.py`** writes `PHX2_dc500m.epw`, `PHX2_dc1000m.epw` and a zero-ΔT control `PHX2_ctrl0.epw` from the onebuilding file. It changes only dry-bulb and RH, recomputed at constant dew point because dry coolers add no vapor. The control run must reproduce the baseline exactly.
4. **Building model:** a DOE MidriseApartment from the ASHRAE 90.1-2019 template, climate zone 2B, built with OpenStudio 3.11 (`create_typical` / openstudio-standards). It has 4 stories, 27 zones, and PTACs with DX cooling and gas heating. Add hourly output meters `Cooling:Electricity`, `CoolingCoils:EnergyTransfer` and `Electricity:Facility`. Use the onebuilding `.ddy` design days in every run so sizing is identical. Simulate it with EnergyPlus 25.2 under the baseline EPW and the three `PHX2_*.epw` files, and save the `eplusout.sql` files as `building_energy/sim/{base,ctrl0,dc500m,dc1000m}.sql`.
5. **`analyze.py`** compares annual and night-time cooling electricity, coil load, and peak cooling and facility demand, and writes `energy_demo_results.csv`.

## Reproducing the paper's numbers

Every script here was rerun from a clean checkout against the inputs used for the paper. These outputs came out byte-identical to the manuscript's:
- the reference case and the sensitivity, stability and campus-size tables;
- all station-test tables;
- all satellite-test tables (`contrast_summary.csv`, `scenes.csv`, `placebo_random_direction.csv`, `shape_vs_model.csv`, `lateral_profile_0.5-2km.csv`);
- `detectability.csv`;
- `dT_hourly.csv`;
- the three modified weather files;
- `energy_demo_results.csv`.

A fresh download from IEM or AppEEARS can differ slightly if those archives are revised.

## Data sources

| Source | Used for | Terms |
| --- | --- | --- |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) ASOS/AWOS archive | Station test, wind at satellite overpass | Please acknowledge: "ASOS data were obtained from the Iowa Environmental Mesonet of Iowa State University." |
| NASA ECOSTRESS `ECO_L2T_LSTE.002` via LP DAAC AppEEARS | Satellite test | NASA Earthdata open data policy; cite the dataset DOI |
| [Climate.OneBuilding.Org](https://climate.onebuilding.org) TMY3 | Building-energy demonstration | See the site's terms of use |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | Campus footprints (coordinates in `satellite_analysis.py`) | ODbL |

No data are redistributed in this repository.

## License

MIT; see [LICENSE](LICENSE).
