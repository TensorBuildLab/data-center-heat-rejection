# data-center-heat-rejection

Code for the manuscript *Neighborhood-scale air temperature and humidity perturbations downwind of data center heat rejection* (Liang Zhang, University of Arizona; in preparation for *Energy and Buildings*).

The core is **`dc_plume.py`**, a reduced-order model of the hourly 2 m air-temperature rise (ΔT) and humidity change (Δq, ΔRH) downwind of a data center's heat-rejection plant: dry coolers (air-cooled) or cooling towers (water-cooled). The other scripts benchmark the model against published field measurements, produce the paper's figures, count the hours per year above 1 K around large campuses, and feed the hourly ΔT into an EnergyPlus building model.

**New here? Start with the tutorials** in [`tutorial/`](tutorial/): `01_quickstart.ipynb` runs the model, and `02_validation_sailor2026.ipynb` reproduces the comparison with published measurements downwind of Phoenix data centers.

```
dc_plume.py                 the model (single file, NumPy only)
test_dc_plume.py            code-verification tests T1–T13 (pytest)
tutorial/                   Jupyter tutorials: quickstart and the validation case study
docs/technical_reference.md full equations, assumptions, validity range, verification ledger
validation/benchmarks/      benchmark against the Sailor et al. (2026) traverses (Fig. 2, Table 4)
paper_figures/              Figures 3–6 and the sensitivity / stability / campus-size tables
analysis/                   hourly TMY3 exceedance of 0.5 / 1 / 2 K (Fig. 7, Table 7)
building_energy/            hourly ΔT → modified weather file → EnergyPlus comparison (Table 8)
validation/                 archive station and ECOSTRESS tests near Goodyear, AZ (Section 4.6)
```

## Install

Python 3.9 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python building_energy/get_weather.py   # Phoenix TMY3 into data/weather/ (for analysis/ and building_energy/)
```

`dc_plume.py` itself needs only NumPy (SciPy is used for `erf` when installed). Run every script **from the repository root**; outputs go to `figures/`, `results/` and `data/`, which are git-ignored.

## The model: `dc_plume.py`

A steady-state (1-hour average) Gaussian plume adapted to large, low-velocity, buoyant heat sources. For each hour it:

1. **Computes the source terms from the exhaust state** (`source_terms`): sensible heat Q_s, latent heat Q_l, vapor flux E and Briggs buoyancy flux follow from the exhaust airflow, temperature and humidity, so no separate power input is needed. Helper constructors build the exhaust state from the heat rejected:
   - `dry_cooler_source(Q_W, dT_air, amb, ...)`: dry coolers. The airflow carries Q_W at a fixed air-side temperature rise, and no moisture is added.
   - `cooling_tower_source(Q_W, inlet_flow, amb, ...)`: cooling towers. The exhaust temperature is solved from the tower energy balance with a near-saturated exhaust (RH_exit 0.95), so it responds to the ambient wet-bulb.
2. **Treats the campus as an area source, a point source, or a blend of the two** (`regime="auto"`):
   - *Area treatment.* The rejected heat mixes a convective layer over the footprint whose depth h_mix is the smallest of three limits: encroachment against the stable gradient, the convective-velocity limit, and the mixing height (`convective_mixing_depth`). It is released as a crosswind strip at the downwind edge.
   - *Point treatment.* Briggs buoyant or momentum plume rise from the footprint centre, with partial penetration of the capping inversion. With `fan_heat` set (W per fan or tower cell), the plant is treated as an array of n = Q/fan_heat fan plumes on a square grid, and the rise of one fan plume is enhanced for merging with Briggs' (1974) multiple-source rule. The paper uses this setting (80 kW per dry-cooler fan, 8 MW per tower cell). With `fan_heat=None` the whole array is one merged plume. That plume rises hundreds of metres on hot days and predicts no daytime warming, contrary to the measurements in `validation/benchmarks/`.
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
air   = dry_cooler_source(137e6, 12.0, night, height=12.0, footprint_x=600.0, footprint_y=600.0,
                          building_height=10.0, exit_velocity=3.0, fan_heat=80e3)    # 100 MW IT, air-cooled
water = cooling_tower_source(125e6, 1650.0, night, height=15.0, footprint_x=300.0,
                             footprint_y=200.0, building_height=10.0, fan_heat=8e6)  # 100 MW IT, water-cooled

x = np.linspace(50, 5000, 400)                       # m downwind of the campus centre
r = plume(air, night, x)                             # plume axis, one wind direction
print(np.interp([1000, 5000], x, r["dT"]))           # -> [0.493 0.163] K
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

Main options of the source helpers: `fan_heat` (W per fan or tower cell; `None` = one merged plume), `exit_velocity` (m/s), `height`, `footprint_x`, `footprint_y`, `building_height`.

Key assumptions: steady state within the hour; horizontally homogeneous atmosphere; flat terrain; excess heat and vapor are passive after rise and mixing; wind below 1 m/s is clipped to 1 m/s. `docs/technical_reference.md` gives every equation (Section 5), the inputs and outputs (Sections 3–4), the validity range (Section 6), a worked Phoenix example (Section 7), and a ledger of every defect found and fixed during development and review (Section 8).

## Tests

```bash
python -m pytest -q          # 22 test cases (T1–T13, some parametrized), about 1–2 minutes
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
| T12 | Reference case without `fan_heat`: air-cooled 0.486 / 0.161 K and water-cooled 0.052 / 0.009 K at 1 / 5 km | within 1e-3 K |
| T13 | Fan-array rise: `fan_heat=None` unchanged; one fan equals the merged plume; smaller fans rise less; n merging fans rise at least as high as independent ones; invalid `fan_heat` raises | as stated |

T12 guards against unintended changes in results. If a deliberate model change alters it, update the expected values and record the change in the technical reference's verification ledger.

## Validation benchmark: `validation/benchmarks/`

The model is evaluated the way dispersion models usually are: it is run for the conditions of a published field study, and predicted values are compared with observed ones using standard statistics (Chang & Hanna 2004). The benchmark is the only published set of air-temperature measurements downwind of operating data centers:

> Sailor, D.J., Samareh Abolhassani, S., Martin, E.P. (2026). Data center waste heat as an emerging urban thermal hazard: first field measurements of neighborhood-scale air temperature impacts. *ASME Journal of Engineering for Sustainable Buildings and Cities* 7(2), 024501. https://doi.org/10.1115/1.4071922. Open access, CC-BY 4.0.

The benchmark uses the four traverses with a reported facility capacity: CyrusOne PHX1–8 on two days, Digital Realty, and NTT PH1.
- `sailor2026_cases.csv`: for each traverse, the published temperature difference between two averaging areas, the weather, the facility capacity, and the section or table of the article each number comes from.
- `sailor2026_geometry.json`: the facility buildings and averaging areas, digitised from the article's Fig. 2 maps.
- `PROTOCOL.md`: the input rules, written down before the comparison was run, and the history of the comparison.

The values are reused under CC-BY with attribution. Please cite the paper if you use them.

```bash
python validation/benchmarks/benchmark_sailor2026.py     # -> results/benchmarks/{benchmark.csv, metrics.json, fig_benchmark.png}
```

Each building is a separate source, and the model is evaluated along the traverse path inside the same two averaging areas as the published difference.

| Case | Observed (°C) | Fan array, central (range) (K) | One merged plume (K) |
| --- | --- | --- | --- |
| a, CyrusOne, 18 Jun | 0.8 | 0.50 (0.36–0.64) | 0.006 |
| b, CyrusOne, 8 Aug | 0.5 | 1.32 (0.67–2.02) | 0.006 |
| d, Digital Realty, 25 Oct | 1.0 | 0.26 (0.16–0.41) | 0.005 |
| e, NTT PH1, 25 Oct (near − far area) | 0.9 | 0.08 (0.06–0.12) | 0.000 |
| Mean | 0.80 | 0.54 | 0.004 |

What this shows:
- **Mechanism and magnitude.** The fan-array model predicts warming of the observed order, about two-thirds of the observed mean. The merged plume predicts less than 0.01 K in every case, two orders of magnitude too low.
- **Individual cases are not reproduced.** FAC2 is 0.25, FB 0.39, NMSE 1.14 and MG 2.25, against Chang & Hanna's acceptance of FAC2 ≥ 0.5 and magnitude of FB ≤ 0.3.
- **Why single traverses are limited.** Each value is one 5–20 min pass, three of the four winds come from airports several kilometres away, and the compared areas differ in land cover. Case e in particular needs a wind closer to due south than the one recorded.

**Post-hoc uncertainty analysis and per-case diagnosis** (labelled as such in `PROTOCOL.md`):

```bash
python validation/benchmarks/wind_uncertainty_sailor2026.py   # about 15 min -> results/benchmarks/wind_uncertainty.{csv,json}
python validation/benchmarks/case_studies_sailor2026.py       # about 10 min -> case_studies.{csv,json}, neighbour_check.csv, fig_case_studies.png
```

- `wind_uncertainty_sailor2026.py` estimates the airport wind-direction error from Phoenix-area ASOS pairs (median daytime disagreement 40–60°, so about 35° between one airport and a site). It then propagates that error and load fractions of 0.5–1.0 through the benchmark. Cases a and b fall inside the 5–95 % prediction range, at the 92nd and 20th percentiles. Cases d and e lie above it.
- `case_studies_sailor2026.py` writes, for each case:
  - a map of the modelled 2 m ΔT with the published areas;
  - the prediction against wind direction;
  - the single input change that would reproduce the observation;
  - a downwash upper bound;
  - the measured spread in each area (Table 1 of the article);
  - a check of unmodelled neighbouring buildings (a hypothesis, not a correction).
  It produces Fig. 2 of the paper.

The benchmark covers four daytime cases at 15–570 m. It does not test the night-time mixed layer or the decay beyond 0.5 km. An earlier version used square-equivalent sites and generic distances (repository commit f577c8e: FAC2 0.75); `PROTOCOL.md` explains why it was replaced. `tutorial/02_validation_sailor2026.ipynb` walks through the comparison step by step and shows how to add new cases.

## Tutorials: `tutorial/`

```bash
pip install notebook          # or jupyterlab
jupyter notebook tutorial/
```

| Notebook | Contents |
| --- | --- |
| `01_quickstart.ipynb` | Weather and campus inputs; ΔT along the plume axis and as a map; air- versus water-cooled; night versus day; the effect of `fan_heat`; the wind-direction average |
| `02_validation_sailor2026.ipynb` | The Sailor et al. (2026) benchmark: inputs, uncertainty ranges, FAC2 / FB / NMSE / MG, scatter plot, and how to add your own measurements |

Both notebooks are stored with their outputs and run in about a minute.

## Paper figures: `paper_figures/`

All scripts share `reference_case.py`. It defines the reference night (Phoenix, class F, U₁₀ = 2 m/s, z_i = 150 m, 32 °C, 20 % RH) and the 100 MW (IT) air-cooled and water-cooled campuses, with the fan-array rise (80 kW per fan, 8 MW per tower cell). Set `DC_MERGED=1` to rerun them with the fully merged plume.

| Script | Produces |
| --- | --- |
| `fig3_fig4_profiles_maps.py` | Fig. 3: plume-axis ΔT and ΔRH profiles, including the edge-release-to-strips band and the point-plume lower bound. Fig. 4: 2-D ΔT and ΔRH maps |
| `fig5_sensitivity.py` | One-at-a-time sensitivity of ΔT at 1 and 5 km to σθ, dθ/dz, the entrainment constant, U₁₀, the blend exponent, along-wind treatment, heat load, C_H, z_i and heat per fan (Fig. 5, tornado chart). Also prints the day and night stability × wind-speed matrix at 1 km for both technologies (Table 6) |
| `fig6_campus_scale.py` | ΔT 0.5 and 3 km beyond the fence for 25 MW–3 GW campuses at constant heat flux per area (Fig. 6). Also prints h_mix and fixed-distance values |
| `validation/benchmarks/benchmark_sailor2026.py`, `wind_uncertainty_sailor2026.py`, `case_studies_sailor2026.py` | Table 4 and Fig. 2 (see above) |
| `analysis/exceedance.py`, `analysis/plot_exceedance.py` | Table 7 and Fig. 7 (see below) |

```bash
python paper_figures/fig3_fig4_profiles_maps.py
python paper_figures/fig5_sensitivity.py
python paper_figures/fig6_campus_scale.py
```

The manuscript figures were rendered with matplotlib 3.6. With that version these scripts reproduce them to within anti-aliasing (under 0.6 % of pixels differ). Newer matplotlib versions shift layout by a few pixels, but the data are unchanged.

## Hours per year above 1 K: `analysis/`

`exceedance.py` runs the model for every hour of the Phoenix TMY3 year. It covers air-cooled campuses of 100 MW, 300 MW and 1 GW (IT; 1 kW IT per m² of heat-rejection footprint) and a 3 GW cluster over 3 km × 3 km. Each hour gets its Turner stability class and wind direction. On a 100 m grid, the script counts the hours with ΔT above 0.5, 1 and 2 K. It takes about 20 minutes.

```bash
python building_energy/get_weather.py      # once
python analysis/exceedance.py              # -> results/exceedance/summary.csv and *.npy grids
python analysis/plot_exceedance.py         # -> figures/fig7_exceedance_maps.png
```

| Campus | Max hours/yr > 1 K outside the fence (night / day) | Area with ≥ 100 h/yr > 1 K |
| --- | --- | --- |
| 100 MW | 609 (9 / 600) | 0.9 km² |
| 300 MW | 1,144 (108 / 1,039) | 4.1 km² |
| 1 GW | 2,380 (973 / 1,554) | 22 km² |
| 3 GW cluster | 816 (816 / 0) | 55 km² |

`FAN_HEAT=0` reruns the analysis with the merged plume.

## Archive station and satellite tests: `validation/`

These scripts test whether the predicted warming shows up in archive records around a small real campus: Stream Data Centers PHXA1 in Goodyear, Arizona (air-cooled, 4.5–40 MW, commissioned 2020-10-14). The predicted signal there (0.06–0.16 K) is below what the records can resolve, so the paper uses them only to bound the warming (Section 4.6). They are kept as general tools for sites with larger campuses.

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

1. **Weather:** `python building_energy/get_weather.py` downloads the Phoenix Sky Harbor TMY3 package (`.epw`/`.ddy`/`.stat`) from [climate.onebuilding.org](https://climate.onebuilding.org) into `data/weather/`. The scripts also accept a copy in `building_energy/dl/phx/`.
2. **`make_epw.py`** computes the hourly ΔT (`dT_hourly{TAG}.csv`) for buildings 0.5 and 1 km beyond the downwind fence of an air-cooled campus. The buildings sit at the bearing the night wind most often blows toward (west). Stability class comes from the Turner method, and calm hours use the direction-averaged value. Environment variables select the campus; the paper's Table 8 uses:
   ```bash
   DC_IT_MW=100  DC_L=600  DC_TAG=_100MW_fan python building_energy/make_epw.py
   DC_IT_MW=1000 DC_L=1000 DC_TAG=_1GW_fan   python building_energy/make_epw.py
   ```
   It reads `building_energy/PHX_TMY3.epw` when present (the same TMY3 year as shipped with [openstudio-standards](https://github.com/NREL/openstudio-standards)) and the onebuilding copy otherwise. Only `dT_hourly{TAG}.csv` is used downstream (see the script header).
3. **`rewrite_epw.py`** (same `DC_TAG`) writes `PHX2{TAG}_dc500m.epw`, `PHX2{TAG}_dc1000m.epw` and a zero-ΔT control `PHX2{TAG}_ctrl0.epw` from the onebuilding file. It changes only dry-bulb and RH; RH is recomputed at constant dew point because dry coolers add no vapor. The control run must reproduce the baseline exactly.
4. **Building model:** a DOE MidriseApartment from the ASHRAE 90.1-2019 template, climate zone 2B, built with OpenStudio 3.11 (`create_typical` / openstudio-standards). It has 4 stories, 27 zones, and PTACs with DX cooling and gas heating. Add hourly output meters `Cooling:Electricity`, `CoolingCoils:EnergyTransfer` and `Electricity:Facility`. Use the onebuilding `.ddy` design days in every run so sizing is identical. Simulate it with EnergyPlus 25.2 under the baseline EPW and the three `PHX2_*.epw` files, and save the `eplusout.sql` files as `building_energy/sim/base.sql` and `building_energy/sim/<TAG without the leading _>_{ctrl0,dc500m,dc1000m}.sql` (for example `1GW_fan_dc500m.sql`).
5. **`analyze.py`** (same `DC_TAG`) compares annual and night-time cooling electricity, coil load, and peak cooling and facility demand, and writes `energy_demo_results{TAG}.csv`. In the paper's runs, annual cooling electricity rises as follows (building 0.5 / 1 km beyond the fence):
   - 100 MW campus: 0.70 % / 0.44 %;
   - 1 GW campus: 3.8 % / 2.2 %.

## Reproducing the paper's numbers

The commands above reproduce the benchmark (`benchmark.csv`, `metrics.json`), the exceedance summary and the Table 8 changes in cooling electricity. For the archive tests, every script was rerun from a clean checkout against the inputs used for the paper, and these outputs came out byte-identical:
- the reference case and the sensitivity, stability and campus-size tables;
- all station-test tables;
- all satellite-test tables (`contrast_summary.csv`, `scenes.csv`, `placebo_random_direction.csv`, `shape_vs_model.csv`, `lateral_profile_0.5-2km.csv`);
- `detectability.csv`;

A fresh download from IEM or AppEEARS can differ slightly if those archives are revised.

## Data sources

| Source | Used for | Terms |
| --- | --- | --- |
| Sailor et al. (2026), doi:10.1115/1.4071922 | Validation benchmark (values in `validation/benchmarks/sailor2026_cases.csv`) | CC-BY 4.0; cite the paper |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) ASOS/AWOS archive | Station test, wind at satellite overpass | Please acknowledge: "ASOS data were obtained from the Iowa Environmental Mesonet of Iowa State University." |
| NASA ECOSTRESS `ECO_L2T_LSTE.002` via LP DAAC AppEEARS | Satellite test | NASA Earthdata open data policy; cite the dataset DOI |
| [Climate.OneBuilding.Org](https://climate.onebuilding.org) TMY3 | Exceedance analysis, building-energy demonstration | See the site's terms of use |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | Campus footprints (coordinates in `satellite_analysis.py`) | ODbL |

Apart from the published benchmark values in `validation/benchmarks/sailor2026_cases.csv`, no data are redistributed in this repository.

## License

MIT; see [LICENSE](LICENSE).
