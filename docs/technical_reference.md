# dc\_plume.py — Technical Reference

Oct 10, 2026 · @Liang

## 1. Purpose and scope

`dc_plume.py` estimates the steady-state rise in near-surface air temperature and humidity downwind of a data center's heat-rejection equipment (dry coolers or cooling towers). It returns ΔT and Δq along a line (1-D: the plume axis, or a wind-direction average versus distance) or over a horizontal field at receptor height (2-D: ΔT(x, y), where x is distance along the wind and y is horizontal distance perpendicular to it). Because σy grows with x, the 2-D field is a widening plume, not the product of two independent 1-D curves.

The model is a buoyant Gaussian plume: the exhaust state sets the heat, moisture, and buoyancy fluxes; for a campus-scale source the heat flux itself mixes a convective layer over the footprint (area regime), while a small source rises as a Briggs plume (point regime); the 1-hour wind-direction fluctuation spreads it laterally and Pasquill–Gifford/Briggs coefficients vertically; images reflect it between the ground and the mixing height; and a penetration fraction removes the part that escapes through the capping inversion. Excess heat and vapor are then treated as passive scalars.

Intended uses:

- **Preliminary study.** Fast, transparent estimates of the magnitude and spatial extent of DC-induced warming for different cooling technologies and atmospheric conditions.
- **Measurement design.** Predicting where and under which conditions a signal should be detectable, to place upwind/downwind sensors.
- **Envelope bounds.** Running the area regime (best estimate) and the merged point plume (lower bound), so that field measurements can discriminate between a heat layer trapped near the ground and a plume that escapes the inversion.
- **Coupling to building energy models.** The hourly ΔT and Δq at a receptor can be added to EnergyPlus weather data or imposed through EMS/Python plugins.

Out of scope: transient and weak-wind (U < 1 m/s) conditions, terrain and building-resolved flow, condensation and fog, surface energy balance feedback, and accumulation over time. These require LES (e.g., PALM-4U) or coupled models and are deliberately left to the funded research.

## 2. Quick start

One call to `plume()` returns the full downwind distribution; the same function produces a 1-D centerline profile, a 1-D wind-direction-averaged (radial) profile, or a 2-D map depending on the shape of `x`, `y` and on `mode`. Requires Python 3.9+ and NumPy; SciPy is used for erf if installed, and Matplotlib only for the example.

```python
import numpy as np
from dc_plume import HeatSource, Ambient, plume

night = Ambient(T=32.0, RH=0.20, U10=2.0, stability="F",
                mixing_height=150.0, elevation=340.0,
                sigma_theta=15.0)      # measured 1-h σθ if available

dry = HeatSource(height=12.0, T_exit=44.0, flow=10500.0, exit_area=1750.0,
                 footprint_x=600.0, footprint_y=600.0,
                 RH_exit=None,          # dry cooler: no moisture added
                 n_plumes=1,            # point regime only: 1 = fans merge
                 building_height=10.0)

x = np.linspace(50, 5000, 400)                       # m from campus center

# 1-D: plume axis for one wind direction (worst case)
r = plume(dry, night, x)

# 1-D: average over uniformly distributed wind directions
r_rad = plume(dry, night, x, mode="radial")

# lower bound: force the merged point plume
r_low = plume(dry, night, x, regime="point")
print(r["area_weight"], r["mixing_depth"])        # 0.986, 85.5 m

# 2-D ground-level map (x downwind, y crosswind)
X = np.linspace(-1000, 5000, 300)[:, None]
Y = np.linspace(-2000, 2000, 200)[None, :]
r_map = plume(dry, night, X, Y, z=2.0)               # r_map["dT"].shape == (300, 200)

print(r["source"]["Q_s"] / 1e6, "MW sensible")
print(np.interp(1000, x, r["dT"]), "K at 1 km")
```

Running `python dc_plume.py` executes the Phoenix summer-night example of Section 7 and writes `dc_plume_example.png`.

To compare technologies on the same heat load, build the sources from heat rejection instead (Section 3.4):

```python
from dc_plume import dry_cooler_source, cooling_tower_source

dry   = dry_cooler_source(137e6, 12.0, night, height=12.0, footprint_x=600.0,
                          footprint_y=600.0, building_height=10.0)
tower = cooling_tower_source(125e6, 1650.0, night, height=15.0, footprint_x=300.0,
                             footprint_y=200.0, building_height=10.0)
```

## 3. Inputs

All inputs are grouped into two dataclasses, `HeatSource` (the equipment) and `Ambient` (the atmosphere), plus the call arguments of `plume()`. A HeatSource built directly takes the exhaust state, and the heat rejected follows from airflow, exhaust state, and ambient state (Section 5.2). The helpers of Section 3.4 work the other way round, from the heat rejected to a consistent exhaust state, and are the recommended way to compare cooling technologies.

### 3.1 HeatSource

| Field | Unit | Meaning | Typical values / guidance |
| --- | --- | --- | --- |
| `height` | m | Fan outlet height above ground | Rooftop dry coolers 10–20 m; ground-level arrays 3–5 m; towers 10–20 m |
| `T_exit` | °C | Exhaust air temperature | Prefer the Section 3.4 helpers. Dry cooler: ambient + 8–15 K. Tower: near-saturated; below ambient dry-bulb in arid climates with cool process water, above it with warm-water cooling. Prefer values from a tower model |
| `RH_exit` | 0–1 | Exhaust relative humidity | `None` for dry coolers (exhaust keeps ambient specific humidity). 0.9–1.0 for cooling towers |
| `flow` | m³/s | Total exhaust airflow at exit conditions, all units combined | From equipment schedules, or flow = Q / (ρ cp ΔT) for dry coolers |
| `exit_area` | m² | Total fan outlet area, all units combined | Sets exit velocity w0 = flow / exit\_area; typically 5–10 m/s |
| `footprint_x` | m | Along-wind size of the heat-rejection area | Masks receptors inside the source; sets residence time, footprint area, and the downwind-edge release position in the area treatment |
| `footprint_y` | m | Crosswind size of the heat-rejection area | Crosswind width of the uniform source, integrated exactly (Section 5.10) |
| `n_plumes` | – | Number of independent plumes sharing the buoyancy | Point regime only. 1 = all fans merge (maximum rise). N\_fans = no merging (minimum rise). Default 1. Ignored when `fan_heat` is set |
| `fan_heat` | W | Heat rejected per fan or tower cell (sensible + latent) | Point regime only. When set, the plant is a fan array whose individual plumes merge by Briggs' multiple-source rule (Section 5.5). Dry-cooler and condenser fans 50–150 kW (80 kW used in the paper); tower cells 5–10 MW (8 MW). `None` (default) keeps the `n_plumes` behaviour |
| `building_height` | m | Height of the structure the equipment sits on | Sets initial vertical spread σz0 = building\_height / 2.15 to mimic wake mixing. 0 to disable |
| `downwash` | – | ISC3 building downwash of the point plume | Default `False`. When `True` and the plume is in the building wake, Section 5.5 (building downwash) applies. Used only as a sensitivity case in the paper |

### 3.2 Ambient

| Field | Unit | Meaning | Typical values / guidance |
| --- | --- | --- | --- |
| `T` | °C | Ambient dry-bulb temperature | Measured upwind or from a reference station |
| `RH` | 0–1 | Ambient relative humidity | Needed for latent flux, buoyancy, and receptor RH |
| `U10` | m/s | Equivalent wind speed at 10 m | Values below 1 m/s are clipped to 1 m/s with a warning |
| `stability` | A–F | Pasquill stability class | Night, clear, light wind: E–F. Overcast or windy: D. Sunny afternoon: A–C. Derive from Turner's method, ΔT between two heights, or Monin–Obukhov length |
| `mixing_height` | m | Depth of the boundary layer that caps vertical spread | Stable night: 50–300 m. Convective day in Phoenix: 1,500–4,000 m. ERA5 boundary-layer height or soundings |
| `elevation` | m | Site elevation above sea level | Phoenix ≈ 340 m, Tucson ≈ 730 m; sets pressure |
| `urban` | bool | Urban or rural dispersion coefficients | `False` for open desert edges, `True` inside built-up areas |
| `dtheta_dz` | K/m | Potential temperature gradient for stable plume rise | `None` uses the ISC defaults: 0.020 (E), 0.035 (F). Ignored for A–D |
| sigma\_theta | deg | 1-hour standard deviation of horizontal wind direction; sets lateral spread | Compute from on-site wind-direction records. None uses class defaults: A 25°, B 20°, C 15°, D 10°, E–F 15°. Light-wind nights often show 15–30° |
| heat\_loss\_velocity | m/s | Transfer velocity of excess heat to the surface and by longwave radiation (Section 5.14) | None: 0.004 × U10 + 0.001 for E–F, 0 for A–D. 0 disables heat loss. Calibrate against measured decay of ΔT with distance |

### 3.3 Call arguments of `plume()`

| Argument | Unit | Meaning |
| --- | --- | --- |
| `x` | m | Downwind distance from the center of the heat-rejection area; scalar or array. Negative values are upwind |
| `y` | m | Crosswind offset; scalar or array broadcastable with `x`. Ignored in radial mode |
| `z` | m | Receptor height; 2 m matches standard air-temperature sensors |
| `mode` | – | `"centerline"`: field for one wind direction. `"radial"`: average over uniformly distributed wind directions |
| `plume_rise` | – | `False` keeps the plume at fan height with ambient turbulence. Diagnostic only: physically inconsistent for strong heat sources (Section 5.13) |
| `n_images` | – | Number of image pairs for reflection at ground and mixing height; default 5 |
| regime | – | "auto" (default) blends the area and point treatments smoothly according to source width relative to the height its heat reaches; "area" or "point" forces one (Section 5.13) |
| lateral | – | "sigma\_theta" (default): σy from the 1-hour wind-direction fluctuation, giving hourly-average plumes. "pg": short-term Pasquill–Gifford σy (Section 5.8) |
| blend\_n | – | Exponent of the area/point blend weight; default 4 (Section 5.13) |
| alongwind\_strips | – | 1 (default): area release at the downwind edge, upper near-field estimate. 2–5: Gauss–Legendre strips over the footprint length, lower near-field estimate (Section 5.13) |

In radial mode a non-zero y raises a warning that it is ignored. Input checks raise `ValueError` for a stability class outside A–F, RH outside 0–1, non-positive flow, exit area or mixing height, `n_plumes` < 1, and `fan_heat` ≤ 0.

### 3.4 Building sources from heat rejection

Two helpers return a `HeatSource` whose exhaust state is computed from the heat rejected, so that technologies are compared on the same load and the exhaust responds to the weather. Both accept the geometry fields of `HeatSource` (height, footprint\_x, footprint\_y, building\_height, n\_plumes) plus `exit_velocity` (defaults 6 m/s for dry coolers, 8 m/s for towers), from which the exit area is set.

`dry_cooler_source(Q_W, dT_air, amb, ...)`: the exhaust is the ambient air heated by the coil's air-side temperature rise, with no moisture added. The airflow is whatever carries Q\_W at that rise:

```latex
T_e = T_a + \Delta T_{air}, \qquad \dot m_{da} = \frac{Q_W}{(c_{pd} + w_a c_{pv})\,\Delta T_{air}}, \qquad q_e = q_a
```

`cooling_tower_source(Q_W, inlet_flow, amb, RH_exit=0.95, ...)`: the dry-air mass flow follows from the inlet airflow at ambient conditions, and the exhaust temperature solves the tower energy balance at relative humidity RH\_exit, with make-up water entering at ambient temperature:

```latex
Q_W = \dot m_{da}\,\big(h(T_e, q_e) - h_a\big) - \dot m_{da}\,(w_e - w_a)\,c_w\,T_a, \qquad h(T, q) = c_{pd}\,T + w\,(2.501\times10^6 + c_{pv}\,T)
```

solved for T\_e by bisection (c\_w = 4186 J/(kg·K)). With this make-up term, the Q\_s + Q\_l computed by `source_terms()` reproduces Q\_W to within 0.1% (Section 8). Less airflow for the same load gives a hotter, more humid exhaust; in hot, dry air the exhaust can be cooler than ambient while still carrying the full heat load as vapour.

If no exhaust temperature between −20 and 80 °C satisfies the balance, the helper raises ValueError instead of returning the search bound. Below 0 °C saturation is computed over supercooled water (Buck), which is adequate for engineering use; freezing fog in the plume is outside the model.

## 4. Outputs

`plume()` returns a dictionary. Field arrays have the broadcast shape of `x` and `y`; receptors inside the source footprint, and receptors whose x or y is NaN, are `NaN`; receptors upwind of the footprint are 0.

| Key | Unit | Meaning |
| --- | --- | --- |
| `x` | m | Downwind distances, as passed |
| `dT` | K | Air temperature rise above ambient at the receptor |
| `T` | °C | Receptor air temperature, T\_amb + ΔT |
| `dq` | kg/kg | Specific humidity rise above ambient |
| `q` | kg/kg | Receptor specific humidity |
| `RH` | 0–1 | Receptor relative humidity; a warning is raised where it exceeds 1 |
| `h_eff` | m | Release height: fan height + rise, capped at 0.98 z\_i (point regime); h\_mix / 2 (area regime) |
| `final_rise` | m | Final rise of one plume (point regime); 0 in the area regime |
| `penetration_fraction` | 0–1 | Share of the plume that escapes through the capping inversion and is excluded from ground-level fields |
| `sigma_y`, `sigma_z` | m | Lateral spread of a point release (source width handled separately) and total vertical spread |
| `U_rise` | m/s | Wind at fan height, used for plume rise |
| `U_dilution` | m/s | Wind at effective plume height, used for dilution |
| `stability_param` | 1/s² | s = (g/T) dθ/dz |
| `source` | dict | Source terms from `source_terms()`, listed below |
| regime | – | "area" (w ≥ 0.99), "point" (w ≤ 0.01), or "blend" |
| mixing\_depth | m | Depth of the source-induced convective layer (area regime); 0 in the point regime |
| buoyancy\_flux\_per\_area | K·m/s | Kinematic virtual heat flux B\_k over the footprint |
| area\_weight | 0–1 | Weight w of the area treatment in the blend (Section 5.13) |
| heat\_loss\_factor | 0–1 | Fraction of excess heat remaining after loss to the surface and by radiation, of the dominant treatment (Section 5.14) |
| heat\_loss\_velocity | m/s | v\_h actually used |
| components | dict | dT and dq of the area and point treatments before blending |
| source\["mixing\_line\_max\_RH"\] | 0–1+ | Highest RH on the exhaust–ambient mixing line; above 1 the plume condenses (Section 5.2) |

`source` keys:

| Key | Unit | Meaning |
| --- | --- | --- |
| `Q_s` | W | Sensible heat flux relative to ambient (negative if the exhaust is cooler than ambient) |
| `Q_l` | W | Latent heat flux, E × L |
| `Q_total` | W | Q\_s + Q\_l; compare with facility power as a sanity check |
| `E` | kg/s | Water vapor added to the atmosphere |
| `F_b` | m⁴/s³ | Briggs buoyancy flux of all plumes combined (virtual-temperature based) |
| `m_da` | kg/s | Dry-air mass flow through the equipment |
| `w0` | m/s | Exit velocity |
| `q_a`, `q_e` | kg/kg | Ambient and exhaust specific humidity |
| `rho_a`, `rho_e` | kg/m³ | Ambient and exhaust moist-air density |
| `cp_a` | J/(kg·K) | Ambient moist-air specific heat |
| `P` | Pa | Atmospheric pressure at site elevation |

For coupling to a building energy model, use `T` and `RH` (or `dT` and `dq`) at the building's location and height; the model gives one steady value per hour of meteorological input.

## 5. Computational procedure

The model runs in twelve steps: moist-air properties, source fluxes, wind, stability, plume rise or area-source mixing (Section 5.13), inversion penetration, dispersion coefficients, added spread, vertical reflection, the Gaussian kernel, conversion to ΔT and Δq, heat loss (Section 5.14), and receptor state. Temperatures in equations are in kelvin unless marked °C; the code converts internally.

### 5.1 Moist-air properties

Pressure at site elevation Z (m), from the standard atmosphere \[ASHRAE 2021\]:

```latex
P = 101325\,\left(1 - 2.25577\times10^{-5}\,Z\right)^{5.25588}
```

Saturation vapor pressure over water (Pa), T in °C, Buck equation with the 1996 coefficients \[Buck 1981\]:

```latex
e_s(T) = 611.21\,\exp\!\left[\left(18.678 - \frac{T}{234.5}\right)\frac{T}{257.14 + T}\right]
```

Specific humidity from relative humidity, and its inverse \[Stull 1988\]:

```latex
e = RH\,e_s(T), \qquad q = \frac{0.622\,e}{P - 0.378\,e}, \qquad e = \frac{q\,P}{0.622 + 0.378\,q}
```

Virtual temperature, moist-air density, specific heat, and latent heat of vaporization \[Stull 1988; Rogers and Yau 1989\]:

```latex
T_v = T\,(1 + 0.608\,q), \qquad \rho = \frac{P}{R_d\,T_v}, \qquad c_p = c_{pd}(1-q) + c_{pv}\,q, \qquad L(T) = (2.501 - 0.00237\,T_{^\circ C})\times10^{6}
```

with R\_d = 287.05 J/(kg·K), c\_pd = 1004.67 J/(kg·K), c\_pv = 1860 J/(kg·K).

### 5.2 Source terms

The exhaust state and the ambient state together define the fluxes, so no power input is needed. With mixing ratio w = q/(1 − q) and total volumetric flow V̇ at exit conditions:

```latex
\dot m_{da} = \frac{\rho_e\,\dot V}{1 + w_e}, \qquad Q_s = \dot m_{da}\,(c_{pd} + w_e\,c_{pv})\,(T_e - T_a), \qquad E = \dot m_{da}\,(w_e - w_a), \qquad Q_l = E\,L(T_a)
```

For a dry cooler (`RH_exit=None`) q\_e = q\_a, so E = 0 and Q\_l = 0. For a cooling tower in dry air, Q\_s can be negative while Q\_l is large. Exit velocity is w0 = V̇ / A\_exit.

The code also checks whether the plume condenses while it dilutes: it evaluates RH along the straight mixing line between the exhaust state and the ambient state and reports the maximum as `mixing_line_max_RH`. A value above 1 means a visible, condensing plume, for which the dry-plume rise and passive-vapour assumptions fail near the source; a warning is raised. For a saturated 31 °C exhaust into a 32 °C, 20% RH summer night the maximum stays at the exhaust's 95%; into a 5 °C, 40% RH winter night it reaches 104%.

Buoyancy flux of all plumes combined, using virtual temperature so that moisture contributes buoyancy \[Briggs 1975; U.S. EPA 1995\]:

```latex
F_b = \frac{g\,\dot V}{\pi}\,\frac{T_{v,e} - T_{v,a}}{T_{v,e}}
```

This follows from Briggs' F = g w0 r0² ΔT/T\_s with π r0² w0 = V̇.

### 5.3 Wind profile

A power law extrapolates the 10 m wind to height z \[U.S. EPA 1995\]:

```latex
U(z) = \max\!\left[U_{10}\left(\frac{\max(z,10)}{10}\right)^{p},\; 1\ \mathrm{m\,s^{-1}}\right]
```

| Class | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- |
| p, rural | 0.07 | 0.07 | 0.10 | 0.15 | 0.35 | 0.55 |
| p, urban | 0.15 | 0.15 | 0.20 | 0.25 | 0.30 | 0.30 |

Plume rise uses U\_rise = U(h\_s) at fan height, as in ISC3. Dilution uses U\_dil = U(h\_eff) at the effective plume height.

### 5.4 Stability parameter

```latex
s = \frac{g}{T_a}\,\frac{\partial \theta}{\partial z}
```

Defaults are ∂θ/∂z = 0.020 K/m for class E and 0.035 K/m for class F \[U.S. EPA 1995\]. If a stable class is given with ∂θ/∂z ≤ 0, the neutral rise formula is used and a warning is raised.

### 5.5 Plume rise

Sections 5.5 and 5.6 apply in the point regime only; the area regime of Section 5.13 replaces them. The buoyancy flux per plume is F = F\_b / n\_plumes. If F ≤ 0 (exhaust denser than ambient) there is no rise; sinking dense plumes are not modeled.

Neutral and unstable classes (A–D), final rise \[Briggs 1975; U.S. EPA 1995\]:

```latex
x^* = \begin{cases} 14\,F^{5/8} & F < 55 \\ 34\,F^{2/5} & F \ge 55 \end{cases}, \qquad \Delta h_f = \frac{1.6\,F^{1/3}\,(3.5\,x^*)^{2/3}}{U_{rise}}
```

This is algebraically identical to ISC3's 21.425 F^(3/4)/U for F < 55 and 38.71 F^(3/5)/U for F ≥ 55.

Stable classes (E–F), the lesser of the windy and calm limits \[Briggs 1975\]:

```latex
\Delta h_f = \min\!\left[2.6\left(\frac{F}{U_{rise}\,s}\right)^{1/3},\; 4\,F^{1/4}\,s^{-3/8}\right]
```

Gradual rise ("two-thirds law"), capped at the final rise \[Briggs 1975\]:

```latex
\Delta h(x) = \min\!\left[\frac{1.6\,F^{1/3}\,x^{2/3}}{U_{rise}},\; \Delta h_f\right]
```

**Plume merging.** Without `fan_heat`, n\_plumes brackets the merging problem: n = 1 assumes all fan plumes merge into one (maximum rise), n = number of fans assumes none merge (minimum rise).

With `fan_heat` = q\_f set, the plant is an array of n = |Q\_W| / q\_f fans on a square grid of spacing Δx = (L\_x L\_y / n)^(1/2). The rise Δh\_1 of one fan plume is computed from F = F\_b / n with the formulas above. It is then enhanced for merging along a crosswind row of N = n^(1/2) fans with Briggs' multiple-source rule \[Briggs 1974; Briggs 1984\]:

```latex
\Delta h = \Delta h_1\,M, \qquad M = \left(\frac{N + S}{1 + S}\right)^{1/3}, \qquad S = 6\left[\frac{(N-1)\,\Delta x}{N^{1/3}\,\Delta h_1}\right]^{3/2}
```

The same factor M multiplies the gradual rise. Closely spaced fans (S → 0) approach the rise of one plume carrying N times the buoyancy; widely spaced fans (S → ∞) rise independently (M → 1). The momentum rise uses the diameter of one fan, d = (4 A\_exit / (π n))^(1/2). The rule was derived for rows of stacks; applying it to one row of a two-dimensional array is the main assumption, and q\_f is an equipment-level input that operators rarely publish.

This matters by day. On hot, light-wind days the fully merged plume of a 300–600 m array rises 1–2.5 km and leaves the surface layer, so it predicts essentially no ground-level warming. Measured traverses near four Phoenix data centers show 0.5–1.0 °C differences between areas 15–570 m from the buildings \[Sailor et al. 2026\]. With q\_f = 80 kW the model predicts warming of that order, 0.54 K against 0.80 °C on average, but does not reproduce individual cases (FAC2 0.25, FB 0.39; see `validation/benchmarks/`). At night the area treatment carries the heat, so q\_f changes the reference-case ΔT by less than 0.1%.

Momentum (jet) rise is also computed and the larger of the two final rises is used, following ISC3's buoyancy/momentum crossover logic \[Briggs 1969; Briggs 1975; U.S. EPA 1995\]. With d the diameter of one plume, d = (4 A\_exit / (π n\_plumes))^(1/2), and momentum flux F\_m = w0² (d/2)² T\_a/T\_e:

```latex
\Delta h_m = \begin{cases} 3\,d\,w_0 / U_{rise} & \text{A--D} \\ \min\!\left[3\,d\,w_0 / U_{rise},\; 1.5\left(\dfrac{F_m}{U_{rise}\sqrt{s}}\right)^{1/3}\right] & \text{E--F} \end{cases}
```

When Δh\_m exceeds the buoyant final rise, Δh(x) = Δh\_m at all distances, since jet rise is completed within a few tens of metres.

**Building downwash (optional, `downwash=True`).** This follows ISC3 \[U.S. EPA 1995, Sections 1.1.4.11 and 1.1.5.3\], with h\_b = building\_height and L\_B = min(h\_b, crosswind width).
- **Wake check.** The plume is in the wake when h\_e ≤ h\_b + 1.5 L\_B. Here h\_e is the release height plus the gradual momentum rise of one fan at 2 h\_b.
- **Schulman–Scire treatment** (release height < h\_b + 0.5 L\_B):
  - The rise of one fan plume is reduced by its initial dilution in the wake: Z³ + (3R₀/β)Z² + (3R₀²/β²)Z = 3F x²/(2β²U³), with R₀ = √2 A σz(3L\_B) and β = 0.6. The stable and momentum forms follow ISC3, and the larger of buoyant and momentum rise is used.
  - The fan-merging enhancement M is then applied to this reduced rise.
  - A = 1 − (h\_e − h\_b)/(2L\_B), clipped to [0, 1], is the linear decay factor.
- **Wake spreads (Huber–Snyder).** σz = A[0.7 h\_b + 0.067(x − 3h\_b)] for 3h\_b ≤ x < 10h\_b, and σz(x + x\_z) beyond, never below the curve value. σy is at least 0.35 h\_b + 0.067(x − 3h\_b). The initial σz0 is then not added.
- **Checks.** Heat flux through a cross-section is conserved (test T14).
- **Effect on the benchmark.** Predictions rise by 0.006–0.026 K for cases a, d and e and by 0.20 K for case b. FB changes from 0.39 to 0.29, and FAC2 stays at 0.25 (`benchmark_sailor2026.py --downwash`).

### 5.6 Penetration of the capping inversion

The fraction of the plume that escapes above the mixing height z\_i is estimated with a heuristic adapted from AERMOD's convective penetration factor \[Cimorelli et al. 2005\]:

```latex
P_{pen} = \min\!\left[1,\; \max\!\left(0,\; 1.5 - \frac{z_i - h_s}{\Delta h_f}\right)\right]
```

P\_pen = 1 if h\_s ≥ z\_i, and 0 when there is no rise. The penetrated fraction is assumed not to reach the ground (it is removed from ground-level fields). The trapped fraction is released at

```latex
h_{eff}(x) = \min\left[h_s + \Delta h(x),\; 0.98\,z_i\right]
```

### 5.7 Dispersion coefficients

Briggs' fits to the Pasquill–Gifford curves, x in m, valid from about 100 m to 10 km \[Briggs 1973; Hanna, Briggs and Hosker 1982\]:

| Class | σy, rural | σz, rural | σy, urban | σz, urban |
| --- | --- | --- | --- | --- |
| A | 0.22x(1+0.0001x)^-1/2 | 0.20x | 0.32x(1+0.0004x)^-1/2 | 0.24x(1+0.001x)^1/2 |
| B | 0.16x(1+0.0001x)^-1/2 | 0.12x | 0.32x(1+0.0004x)^-1/2 | 0.24x(1+0.001x)^1/2 |
| C | 0.11x(1+0.0001x)^-1/2 | 0.08x(1+0.0002x)^-1/2 | 0.22x(1+0.0004x)^-1/2 | 0.20x |
| D | 0.08x(1+0.0001x)^-1/2 | 0.06x(1+0.0015x)^-1/2 | 0.16x(1+0.0004x)^-1/2 | 0.14x(1+0.0003x)^-1/2 |
| E | 0.06x(1+0.0001x)^-1/2 | 0.03x(1+0.0003x)^-1 | 0.11x(1+0.0004x)^-1/2 | 0.08x(1+0.0015x)^-1/2 |
| F | 0.04x(1+0.0001x)^-1/2 | 0.016x(1+0.0003x)^-1 | 0.11x(1+0.0004x)^-1/2 | 0.08x(1+0.0015x)^-1/2 |

σz always comes from this table; σy comes from it only with lateral="pg" (Section 5.8). The coefficients are evaluated at max(x, 1 m) to avoid division by zero; results closer than about 100 m are not reliable.

### 5.8 Lateral spread, source width, and vertical initial spread

Lateral spread is computed from the 1-hour standard deviation of horizontal wind direction, σθ, with Draxler's time-dependent factor \[Draxler 1976; Irwin 1983; U.S. EPA 2000\]:

```latex
\sigma_{y,turb} = \sigma_\theta\,x\,f_y(t), \qquad f_y = \frac{1}{1 + 0.9\sqrt{t / 1000\ \mathrm{s}}}, \qquad t = \frac{x}{U_{dil}}
```

This captures wind-direction meander, which dominates hourly plume width on light-wind nights. The short-term Pasquill–Gifford σy of Section 5.7 (sampling times of about 10 minutes) misses it and keeps a nighttime plume almost as narrow as its source for kilometres; it remains available with `lateral="pg"`. Use measured σθ (`Ambient.sigma_theta`) whenever possible. Defaults are 25°, 20°, 15°, 10° for classes A–D (mid-points of the EPA σθ classification ranges) and 15° for E–F, a placeholder for light-wind meander rather than the much smaller short-term E–F ranges.

Buoyancy-induced dispersion is added in quadrature \[Pasquill 1976\], and the vertical spread includes the initial spread of the release layer:

```latex
\sigma_y^2 = \sigma_{y,turb}^2 + \left(\frac{\Delta h}{3.5}\right)^2, \qquad \sigma_z^2 = \sigma_{z,PG}^2(x) + \sigma_{z0}^2 + \left(\frac{\Delta h}{3.5}\right)^2
```

σz0 = H\_b / 2.15 in the point regime and max(H\_b, h\_mix) / 2.15 in the area regime, following ISC3's surface-based volume-source convention \[U.S. EPA 1995\], with H\_b the `building_height`.

The crosswind width W of the source (`footprint_y`) is not folded into σy; it is integrated exactly in Section 5.10 as a uniform crosswind line of width W. An earlier version added σy0 = W/4.3 in quadrature, which squeezed a 600 m campus into a Gaussian about 340 m wide at half maximum and overstated the centerline ΔT by about 65% (Section 8, defect 10).

### 5.9 Vertical term with reflections

The ground and the mixing height are treated as perfect reflectors using images \[Turner 1994; Seinfeld and Pandis 2016\]:

```latex
V(z) = \sum_{n=-N}^{N}\left\{\exp\!\left[-\frac{(z - h_{eff} + 2n z_i)^2}{2\sigma_z^2}\right] + \exp\!\left[-\frac{(z + h_{eff} + 2n z_i)^2}{2\sigma_z^2}\right]\right\}
```

When σz > 1.6 z\_i the plume is well mixed through the layer and the sum is replaced by its limit \[Turner 1994\]:

```latex
V = \frac{\sqrt{2\pi}\,\sigma_z}{z_i}
```

### 5.10 Gaussian kernel

Centerline mode: a uniform crosswind line of width W, convolved with the Gaussian spread of each element, gives the plume per unit emission (units s/m³) \[Turner 1994; Seinfeld and Pandis 2016\]:

```latex
K_c(x,y,z) = \underbrace{\frac{V(z)}{\sqrt{2\pi}\,U_{dil}\,\sigma_z}}_{\text{vertical}} \cdot \underbrace{\frac{1}{2W}\left[\operatorname{erf}\!\left(\frac{y + W/2}{\sqrt{2}\,\sigma_y}\right) - \operatorname{erf}\!\left(\frac{y - W/2}{\sqrt{2}\,\sigma_y}\right)\right]}_{\text{lateral}}
```

The lateral factor integrates to 1 over y, so heat is conserved. Near the source (σy ≪ W) it is a flat band of height 1/W, matching a well-mixed strip; far downwind (σy ≫ W) it tends to the familiar exp(−y²/2σy²)/(√(2π) σy).

Radial mode averages the plume over wind directions uniformly distributed around the compass. Integrating K\_c across the plume and spreading it over the circumference 2πr gives the sector-average form used in long-term dispersion models \[U.S. EPA 1995\]:

```latex
K_r(r,z) = \min\!\left[\frac{V(z)}{\sqrt{2\pi}\,U_{dil}\,\sigma_z\,2\pi r},\; K_c(r,0,z)\right]
```

The cap at the centerline value prevents the radial form from exceeding the plume axis close to the source, where the plume is as wide as the circle. For a real wind rose, weight one centerline run per direction sector by its frequency.

### 5.11 Conversion to ΔT and Δq

Excess heat and vapor are treated as conserved passive scalars after rise, so the kernel converts an emission rate into a concentration. The heat "emission" Q\_s / (ρ\_a c\_pa) has units K·m³/s, and the vapor emission E / ρ\_a has units (kg/kg)·m³/s:

```latex
\Delta T = (1 - P_{pen})\,\frac{Q_s}{\rho_a\,c_{pa}}\,K, \qquad \Delta q = (1 - P_{pen})\,\frac{E}{\rho_a}\,K
```

Receptors inside the source footprint are set to NaN. In centerline mode, receptors more than half a footprint upwind are set to zero.

### 5.12 Receptor state

```latex
T = T_a + \Delta T, \qquad q = q_a + \Delta q, \qquad RH = \frac{e(q,P)}{e_s(T)}
```

If RH > 1 anywhere, a warning flags likely condensation; the passive-scalar assumption is invalid there.

### 5.13 Area regime: source-induced convective mixing

A campus-scale heat source changes the stability it is dispersed in. The example dry-cooler campus releases 376 W/m² of sensible heat over 36 ha, comparable to a sunny afternoon, so the air over it is convectively mixed even on a class F night. Using class F dispersion with a ground-level release (the no-rise case) ignores this and confines the heat to a layer about 10 m deep; using a merged point-plume rise lets a 600 m-wide array rise like a single stack. Both are physically inconsistent for wide arrays of closely spaced fans.

The area regime instead computes the depth of the convective layer the source builds while air crosses it, then releases the excess heat and vapor uniformly through that layer. The kinematic buoyancy (virtual heat) flux per unit footprint area is

```latex
B_k = \frac{1}{L_x L_y}\left[\frac{Q_s}{\rho_a c_{pa}} + 0.608\,T_a\,\frac{E}{\rho_a}\right]
```

so cooling-tower moisture contributes buoyancy. The residence time over the source is t = L\_x / U\_rise. The mixing depth is the smallest of three limits:

```latex
h_{mix} = \min\!\left[\sqrt{\frac{2(1+2A)\,B_k\,t}{\gamma}},\;\; \left(\frac{g\,B_k}{T_a}\right)^{1/2} t^{3/2},\;\; z_i\right], \qquad \gamma = \frac{\partial\theta}{\partial z},\; A = 0.2
```

The first term is encroachment of a convective layer into a stable gradient with entrainment factor (1 + 2A) \[Stull 1988; Batchvarova and Gryning 1991\]. The second is the distance thermals rising at the convective velocity w\* = (g B\_k h / T\_a)^(1/3) can travel in time t \[Deardorff 1970\]; it limits growth when γ is small (classes A–D). The third caps the layer at the mixing height. h\_mix is at least the fan height.

The mixed layer leaves the footprint at its downwind edge, so in the area treatment the release is a crosswind strip of width W at x = L\_x/2 and every distance-dependent quantity (σy, σz, travel time, heat loss) uses the distance from that edge, d = x − L\_x/2. The strip is released at h\_eff = h\_mix / 2 with initial vertical spread σz0 = max(H\_b, h\_mix) / 2.15 (ISC3 surface-based volume-source convention \[U.S. EPA 1995\]), no further rise, and no penetration. Receptors beside the footprint but upwind of its downwind edge receive no heat. The point treatment keeps its release at the footprint centre.

The initial vertical spread is combined with the ambient σz in quadrature rather than through a virtual upwind distance. Both conventions are used in regulatory models (ISC3 used virtual distances); for class F, whose σz levels off near 53 m, a virtual distance would make far-field vertical growth even slower, raising ground-level ΔT by roughly 15% at 5 km.

The encroachment depth uses the travel time over the footprint, t = L\_x / U. This has the same form, h ∝ (2 H x / (ρ c\_p U γ))^(1/2), as thermal internal boundary-layer models used for coastal fumigation \[Venkatram 1977\]. With w\* ≈ 1 m/s and h\_mix ≈ 85 m, air crossing the campus experiences only about three convective turnovers (h/w\* ≈ 90 s against t ≈ 270 s), so the layer may not be fully uniform; this is an idealization to test by sensitivity to A and γ, and against measured profiles.

Regime selection with `regime="auto"` (the default) is a smooth blend rather than a switch. The weight of the area treatment is

```latex
w = \frac{R^4}{1 + R^4}, \qquad R = \frac{\min(L_x, L_y)}{\max(h_{conv},\, h_s,\, \Delta h_{f,point})}
```

where Δh\_f,point is the point-treatment final rise. R is a source aspect ratio in atmospheric terms: the horizontal size of the source over the vertical scale its heat reaches. For R ≫ 1 the heated air spreads over a width much larger than its depth and cannot organize into an isolated buoyant column, so it forms a thermal internal boundary layer, a shallow area source. For R ≪ 1 the source is narrower than its plume rise and the heat leaves as a concentrated rising column, as in classical Briggs theory. ΔT and Δq are w × area + (1 − w) × point. Unlike the earlier hard threshold (footprint ≥ max(h\_mix, 100 m)), results now change continuously with source size: for the Section 7 campus scaled from 60 m to 600 m, w rises smoothly from 0.77 to 0.99. `regime="area"` or `regime="point"` forces one treatment, and B\_k ≤ 0 gives w = 0.

For the Section 7 dry-cooler case, B\_k = 0.340 K·m/s, t = 271 s, γ = 0.035 K/m, giving h\_mix = 86.0 m from the encroachment limit (the w\* limit is 468 m), Δh\_f,point = 207 m, R = 2.9, and w = 0.986. One effect remains outside the model: inversion warmth mixed down to the surface by the induced turbulence, which would raise near-field ΔT.

The blend exponent (`blend_n`, default 4) is a heuristic. For the Section 7 night it barely matters: exponents 2, 4, 8 give w = 0.89, 0.99, 1.00 and ΔT at 1 km of 0.44, 0.49, 0.49 K. For the daytime case of Section 7.3 it matters more: w = 0.43, 0.36, 0.25 and ΔT at 500 m of 0.10, 0.09, 0.06 K. Report daytime results with this range.

Along-wind extent: the default releases all heat at the downwind edge, which is the upper near-field estimate because every heat element is placed as close to downwind receptors as possible. `alongwind_strips = n` (2–5) instead spreads the release over the footprint length as n Gauss–Legendre strips, each carrying its share of the heat with the same mixed-layer geometry (an ISC-style area integration); this is the lower near-field estimate. For the Section 7 campus the two give 0.49 and 0.45 K at 1 km and 0.16 and 0.15 K at 5 km, a 6–9% spread; results are identical for n = 2 to 5, so two strips suffice. Report near-fence predictions (within a few hundred metres of the footprint) as this range. Receptors beside the footprint (|x| < L\_x/2, |y| > L\_y/2) receive heat only in strip mode, for example 0.08–0.16 K 20 m beside the Section 7 campus, while the edge release gives zero there; use strip mode for receptors flanking a campus. In strip mode the diagnostic outputs (σy, σz, h\_eff, heat\_loss\_factor) refer to the edge-release reference geometry.

### 5.14 Heat loss to the surface and by longwave radiation

On a clear stable night the excess heat is not conserved: the warm layer loses heat to the colder ground and by extra longwave emission. This is treated like dry deposition of a pollutant, as a source-depletion factor applied to ΔT (not to Δq) \[Chamberlain 1953; Horst 1977\]. The heat flux carried by the plume decreases at a rate v\_h times the ground-level excess, and the ratio of ground-level excess to the plume's column-integrated excess is taken from the same vertical kernel used for dispersion (Section 5.9), so loss and dispersion are consistent by construction:

```latex
F_{loss}(d) = \exp\!\left[-v_h \int_0^{d} \frac{V(0;\,h_{eff},\,\sigma_z)}{\sqrt{2\pi}\,\sigma_z\,U_{dil}}\,dx'\right]
```

For an elevated point plume this reduces to the classical Horst form, √(2/π) exp(−h²/2σz²)/σz; for a well-mixed layer it reduces to 1/z\_i; for the area treatment it gives a ground/column ratio of about 1/h\_mix, matching a uniform layer. The transfer velocity is

```latex
v_h = C_H\,U_{10} + v_{rad}, \qquad C_H = 0.004,\quad v_{rad} = 0.001\ \mathrm{m\,s^{-1}}
```

for classes E–F (stable bulk transfer coefficient referenced to the near-surface wind \[Garratt 1992\], plus an approximate longwave term), and zero for A–D, where ambient convection already carries the heat upward. v\_h uses the near-surface wind U10 while the depletion integral uses the plume transport wind U\_dil; on the Section 7 night these differ by a factor of 2.2. Both constants are approximate and should be calibrated against measurements; `Ambient.heat_loss_velocity` overrides v\_h.

The default C\_H = 0.004 is at the upper end for stable nights over desert. Neutral values for roughness lengths of 0.01–0.05 m are about 0.0025–0.004, and stable stratification lowers them further, so 0.001–0.004 is a reasonable range; built-up areas (z0 ≈ 0.5 m) reach about 0.01 under neutral conditions. Lowering C\_H from 0.004 to 0.001 raises F\_loss at 5 km from 0.90 to 0.97 and ΔT there from 0.161 to 0.173 K, so the default leans toward lower far-field estimates.

For the Section 7 case v\_h = 0.009 m/s and F\_loss = 0.98 at 1 km and 0.90 at 5 km. A simple uniform-layer estimate, exp(−v\_h d / (U\_dil h\_mix)), gives 0.90 at 5 km as well, an independent cross-check. An earlier version used the ground-release form √(2/π)/σz for the area treatment, which overstated the near-source loss rate about 1.6-fold and gave F\_loss = 0.85 at 5 km (Section 8, defect 17). The integral is evaluated numerically on a 2,000-point grid, and the remaining heat flux through any plane equals F\_loss exactly (Section 8).

## 6. Assumptions, validity range, and limitations

The model is a screening tool: it is most reliable for relative comparisons (cooling technology, stability class, distance) and for bracketing ground-level ΔT between the area regime (best estimate) and the merged point plume (lower bound). Absolute values have been compared only with four daytime traverses at 100–500 m (Section 8, `validation/benchmarks/`); night-time and far-field values remain untested against measurements.

| Assumption or limitation | Effect on results | Where it matters most |
| --- | --- | --- |
| Steady state, one hour = one equilibrium | Ignores travel time and the build-up and decay of the plume | Rapid changes at sunset and sunrise |
| Horizontally homogeneous, flat terrain | No channeling by washes, slopes, or drainage flows | Night-time cold-air drainage in desert basins |
| Excess heat treated as a passive scalar after rise | Heat loss to the ground and by radiation represented only by an approximate transfer velocity (Section 5.14) | Far field (> 2 km); calibrate v\_h against measurements |
| Pasquill–Gifford/Briggs σz curves; σy from a default or measured σθ | Empirical; valid about 100 m to 10 km in open terrain; Draxler's f\_y was fitted to near-surface releases | Near field (< 100 m) and very stable, low-wind nights |
| Wind clipped at 1 m/s | Calm conditions not represented | Calm nights are often the highest-impact cases; these need LES |
| Final rise = larger of buoyancy and momentum rise | Momentum rise, when it dominates, is applied at full height from the source, overstating rise in the first tens of metres | High exit velocity with small temperature excess; buoyancy dominates in all Section 7 cases |
| Point regime: plume merging by Briggs' multiple-source rule (`fan_heat`), or bracketed by n\_plumes | The rule was derived for rows of stacks and is applied to one row of a 2-D array; q\_f is rarely published | Daytime near-field ΔT; over 50–150 kW, q\_f changes the benchmark predictions by −13% to +6%, far less than the change from the fully merged limit |
| Penetration heuristic | Escaped fraction is removed, not re-entrained | Shallow stable layers with strongly buoyant plumes |
| Building wake only through σz0 | No downwash cavity, no recirculation to the DC's own intakes | Receptors within a few building heights; intake recirculation needs CFD |
| No condensation or evaporation of liquid water | Fog and visible plumes not represented; flagged when the mixing line or a receptor exceeds 100% RH | Cooling towers on cold or humid nights (e.g., 5 °C winter nights); use a wet-plume model such as SACTI there |
| Dense plumes (F\_b ≤ 0) given zero rise | Sinking is not modeled | Rare; towers in very humid, cool conditions |
| Ambient T, RH, U at a single reference point | No pre-existing urban heat island or land-cover contrast | Separate the DC signal from background with upwind/downwind pairs |
| Single source per call | Clusters require superposition | Superpose ΔT from several calls; valid because the equations are linear in Q |
| Area regime: heat mixed uniformly through h\_mix over the footprint | Ignores plumes that overshoot h\_mix and partial merging near the edges; h\_mix depends on the entrainment constant A and on γ | Strong inversions with weak winds; test sensitivity to A (0.1–0.3) and dθ/dz |
| Area regime: ΔT added to the ambient profile | Warmer inversion air mixed down by the induced turbulence is not counted, so near-field ΔT may be underestimated | Within about 1 km on clear, calm nights with strong surface inversions |
| Area/point blend weight w = R⁴/(1 + R⁴) | Continuous, but the functional form and exponent are heuristic; daytime w ranges 0.25–0.43 for exponents 8–2 | Sources whose width is comparable to their plume rise (R ≈ 1); inspect both components |
| Default σθ = 15° for classes E–F | Far-field ΔT scales roughly inversely with σθ: 5°, 15°, 30° give 0.41, 0.17, 0.09 K at 5 km | Beyond about 1 km; replace with measured hourly σθ |

The last row has a useful consequence: because ΔT is linear in Q\_s for a given rise, contributions from several campuses can be summed, provided each campus's plume rise is computed separately.

## 7. Worked example: Phoenix summer night

On the same calm, stable night, the same 100 MW IT campus warms the air 1 km downwind by about 0.49 K if air-cooled and 0.05 K if water-cooled, roughly a tenfold difference. The water-cooled campus instead raises RH by about 2.5 percentage points. All results are 1-hour averages; inputs are illustrative, not measured.

**Comparison design.** Both campuses see the same weather and serve the same IT load. Total heat rejected differs because the air-cooled plant uses more electricity: 137 MW for air cooling and 125 MW for water cooling. Each exhaust state is computed from the heat rejected rather than assumed, with the helpers of Section 3.4.

**Ambient:** T = 32 °C, RH = 20%, U10 = 2 m/s, class F, z\_i = 150 m, σθ = 15° and v\_h = 0.009 m/s (defaults for class F), elevation 340 m, rural.

**Air-cooled (dry coolers):** `dry_cooler_source(137e6, 12.0, night, ...)`: 12 K air-side rise, so exhaust 44.0 °C at ambient humidity and 10,616 m³/s; fans at 12 m, 600 m × 600 m footprint, 10 m building.

**Water-cooled (cooling towers):** `cooling_tower_source(125e6, 1650.0, night, ...)`: 1,650 m³/s inlet airflow, so the energy balance gives an exhaust of 33.8 °C at 95% RH, consistent with condenser water returning near 35 °C; outlets at 15 m, 300 m × 200 m footprint, 10 m building.

### 7.1 Downwind profiles

| Case | Area weight w | Q\_s (MW) | Q\_l (MW) | Mixing depth or rise (m) | ΔT at 1 km (K) | ΔT at 5 km (K) | ΔRH at 1 km (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Air-cooled, default (upper near-field) | 0.986 | 137.0 | 0 | 86.0 (mixing depth) | 0.49 | 0.16 | −0.6 |
| Air-cooled, along-wind strips (lower near-field) | 0.986 | 137.0 | 0 | 86.0 (mixing depth) | 0.45 | 0.15 | −0.5 |
| Air-cooled, without heat loss | 0.986 | 137.0 | 0 | 86.0 (mixing depth) | 0.49 | 0.18 | −0.6 |
| Air-cooled, wind-direction average | 0.986 | 137.0 | 0 | 86.0 (mixing depth) | 0.05 | 0.01 | −0.1 |
| Air-cooled, merged point plume (lower bound) | 0 | 137.0 | 0 | 207 (rise), 83% penetrates | < 0.01 | < 0.01 | ≈ 0 |
| Air-cooled, no rise (diagnostic only) | 0 | 137.0 | 0 | 0 | 3.07 | 0.48 | −3.2 |
| Water-cooled, default | 0.960 | 3.5 | 121.4 | 43.0 (mixing depth) | 0.05 | 0.01 | +2.5 |
| Water-cooled, wind-direction average | 0.960 | 3.5 | 121.4 | 43.0 (mixing depth) | < 0.01 | < 0.01 | +0.2 |
| Water-cooled, point plume | 0 | 3.5 | 121.4 | 90 (rise) | ≈ 0 | ≈ 0 | ≈ 0 |

&#91;image: Ground-level ΔT and ΔRH along the plume axis for the air-cooled and the water-cooled campus on the same night\]

### 7.2 Two-dimensional ground-level fields

On a shared colour scale, the air-cooled campus produces a warm band that leaves its downwind edge 600 m wide and widens to about 1.6 km (width at half peak) by 5 km; the water-cooled campus's warm patch is about a tenth as strong and fades within 2–3 km. Its humidity plume is the more visible signature.

&#91;image: 2-D ground-level fields on the same night: ΔT for the air-cooled campus, ΔT for the water-cooled campus on the same colour scale, and ΔRH for the water-cooled campus\]

Generated with `plume(src, night, X, Y, z=2.0)` on a grid X = −0.5 to 5 km (along the wind) by Y = ±2 km (across it); grey rectangles are the source footprints. The lateral spread comes from the 1-hour wind-direction fluctuation σθ (Section 5.8). With short-term Pasquill–Gifford σy instead (`lateral="pg"`), the bands stay about as wide as the sources for 5 km, like hot air in a duct, which is not what hourly observations show on light-wind nights.

### 7.3 What the results mean

1. **Cooling technology sets the local warming.** The two campuses reject similar total heat (137 vs 125 MW), but 97% of the cooling tower's heat leaves as latent heat, which does not warm the air unless the vapour condenses. Only its 3.5 MW of sensible heat warms the air, against 137 MW for dry coolers.
2. **The magnitude agrees with an independent estimate.** A box model mixing 137 MW through the 150 m stable layer across a 700 m width gives about 0.5 K; the air-cooled default gives 0.49 K at 1 km, and the along-wind strip treatment 0.45 K. Satellite studies report about 2 K in land-surface temperature, which runs higher than air temperature.
3. **Dry coolers slightly lower RH; towers raise it.** Warming dry air lowers its relative humidity (−0.6 pp at 1 km), while the towers add 121 MW worth of water vapour (+2.5 pp at 1 km). For heat stress and for neighbours' evaporative coolers, the tower's humidity matters more than its temperature.
4. **The tower result depends on its exhaust temperature.** With the same heat load, less airflow gives a hotter exhaust and more warming: inlet airflow of 1,200, 1,650, and 2,200 m³/s gives exhausts of 37.8, 33.8, and 30.7 °C and ΔT at 1 km of +0.10, +0.05, and −0.08 K. Warm-water liquid cooling pushes toward the hotter end; in very dry, hot air the exhaust can fall below ambient and the tower slightly cools its surroundings.
5. **Lateral spread and heat loss control the far field.** σθ = 5°, 15°, and 30° give 0.38, 0.16, and 0.08 K at 5 km for the air-cooled campus; heat loss removes 2% of the excess heat by 1 km and 10% by 5 km. Measured σθ and a calibrated v\_h are essential beyond about 1 km.
6. **Earlier, larger values were modeling artifacts.** The no-rise case ignores the convective mixing a 380 W/m² source creates (Section 5.13); single-Gaussian source width and 10-minute σy (Section 5.8) each overstated the centerline. Section 8 lists all corrections.
7. **The wind-direction average is an order of magnitude below the plume axis.** Use the axis for an hour with steady mean wind direction and the radial average for long-term exposure.

Daytime impact is smaller: with T = 43 °C, RH = 10%, class B, U10 = 4 m/s, and z\_i = 2,000 m, the air-cooled exhaust is 55 °C, the area weight drops to 0.36 because the merged plume would rise about 600 m, and ΔT is 0.09 K at 500 m and 0.08 K at 1 km (0.06–0.10 K at 500 m for blend exponents 8–2). The cooling tower's exhaust (36 °C) is then below the 43 °C ambient, giving −0.02 K.

## 8. Verification and review ledger

Writing this document exposed eight coding defects, and reviewing the Section 7 results and 2-D map exposed three physical ones; three rounds of external review then prompted nine more changes (rows 12–20, Sections 8.2–8.4). All are fixed in the current `dc_plume.py`. Defects 1–8 affected edge cases, 2-D maps, and momentum-dominated sources. Defects 9–11 changed the Section 7 results: the 1 km ΔT went from a 0.01–6 K bracket to 0.85 K, then 0.51 K, then 0.46 K, and the 5 km value fell to 0.17 K once lateral spread was realistic.

| # | Issue found | Consequence before fix | Fix |
| --- | --- | --- | --- |
| 1 | Momentum (jet) rise not modeled | Rise underestimated for fast, weakly buoyant exhausts | Added Briggs momentum rise; final rise = larger of buoyancy and momentum (Section 5.5) |
| 2 | Upwind receptors in 2-D maps evaluated the kernel at x = 1 m | Spurious ΔT upwind of the campus | Kernel set to zero for x ≤ −footprint\_x/2 |
| 3 | Footprint mask used x only | Every receptor with x < footprint\_x/2 was NaN, including all upwind and off-axis points | Mask now requires both abs(x) < footprint\_x/2 and abs(y) < footprint\_y/2 |
| 4 | Radial mode accepted negative x | Meaningless values for negative radii | Radial mode uses abs(x) as the distance |
| 5 | `dtheta_dz=0.0` replaced by the default through Python's `or` | User input silently ignored | Explicit `is not None` check |
| 6 | Stable rise formula divided by s = 0 | Crash or infinite rise for a stable class with zero gradient | Neutral formula used, with a warning |
| 7 | Stability class not normalized in the wind-profile lookup | Lower-case class letters raised `KeyError` | Class upper-cased once and validated against A–F |
| 8 | No input validation | Invalid RH (e.g., 20 instead of 0.20) produced wrong results silently | `ValueError` for out-of-range RH, class, flow, exit area, mixing height, n\_plumes |
| 9 | Physical error found in a second review: ambient class F dispersion applied to a source releasing 376 W/m², and a 600 m array treated as one stack | Ground-level ΔT bracketed between 6 K (no rise) and 0.01 K (merged rise), both unrealistic; tower ΔRH similarly overstated or zero | Added the area regime (Section 5.13) as the default for campus-scale sources: ΔT at 1 km became 0.85 K (0.51 K after defect 10); no-rise case kept as a diagnostic only |
| 10 | Found in the 2-D map: the 600 m source width entered as a single Gaussian (σy0 = W/4.3), so the plume at 1 km was narrower than the campus | Centerline ΔT overstated by about 65% near the source (0.85 K instead of 0.51 K at 1 km); lateral extent understated | Exact crosswind integration of a uniform source of width W (erf profile, Sections 5.8 and 5.10) |
| 11 | Found in the 2-D map: class F Pasquill–Gifford σy (10-minute sampling) kept the warm band about 600 m wide for 5 km, like hot air in a duct | Far-field centerline ΔT overstated about 2.7× at 5 km (0.47 K instead of 0.17 K); lateral extent understated | σy from 1-hour σθ with Draxler's factor (default); measured σθ as an input; PG kept as lateral="pg" |
| 12 | Area treatment measured distance from the footprint centre, although the mixed layer forms at the downwind edge | 300 m of spurious dispersion; near-field ΔT about 4–8% low | Release strip at the downwind edge, d = x − L\_x/2 (Section 5.13) |
| 13 | Excess heat treated as fully conserved on stable nights | Far field overstated, about 15% at 5 km in Section 7 | Chamberlain–Horst depletion with transfer velocity v\_h (Section 5.14) |
| 14 | Hard area/point switch at footprint ≥ max(h\_mix, 100 m) | ΔT could jump by orders of magnitude between a 99 m and a 101 m source | Smooth blend w = R⁴/(1 + R⁴) (Section 5.13) |
| 15 | Condensation checked only at receptors, not along the plume's dilution path | Condensing winter tower plumes went unflagged | Mixing-line saturation check with warning (Section 5.2) |
| 16 | Inner function named lateral shadowed the lateral argument; input shapes handled implicitly | Worked by accident of ordering; fragile | Renamed \_lateral\_profile; centerline mode broadcasts x and y explicitly; radial mode ignores y explicitly |
| 17 | Found by cross-checking the reviewer's uniform-layer decay formula: heat loss used the ground-release form √(2/π)/σz in the area treatment, inconsistent with the actual vertical profile | Near-source loss rate about 1.6× too high; F\_loss at 5 km 0.85 instead of 0.90 | Loss rate from the same vertical kernel as the dispersion, V(0)/(√(2π) σz U) (Section 5.14) |
| 18 | A NaN anywhere in x made the heat-loss grid's maximum distance NaN | Heat loss silently switched off for every receptor (ΔT at 5 km 0.179 instead of 0.161 K); NaN receptors returned 0 | nanmax for the grid; NaN x or y now returns NaN |
| 19 | In strip mode, diagnostics were copied from the last strip | σz, h\_eff and heat\_loss\_factor referred to an arbitrary strip (σz 41.4 vs 41.0 m) | Diagnostics from the edge-release reference; flank-receptor behaviour documented (Section 5.13) |
| 20 | Tower bisection could return its search bound silently | Unphysical exhaust state for impossible Q\_W and airflow combinations | ValueError when the solution reaches −20 or 80 °C (Section 3.4) |
| 22 | Building downwash was represented only by σz0 = H\_b/2.15, although the fans release about 2 m above roofs (below the GEP height h\_b + 1.5 L\_B) | Possible underestimate of daytime near-field warming | Optional ISC3 downwash (Schulman–Scire reduced rise, Huber–Snyder wake spreads), test T14. Benchmark change ≤ 0.2 K; the default is unchanged, and downwash is reported as a sensitivity case only |
| 21 | Found by benchmarking against the Sailor et al. (2026) traverses: the fan array was treated as one merged plume | On hot, light-wind days the plume rose 1–2.5 km and predicted < 0.01 K where 0.5–1.0 °C was measured (FB 1.98, FAC2 0) | `fan_heat` option: n fan plumes merging by Briggs' multiple-source rule (Section 5.5); the paper uses 80 kW per fan. Night-time reference values change by < 2% (0.486 → 0.493 K at 1 km). In the geometry-resolved benchmark (`validation/benchmarks/PROTOCOL.md`), the mean predicted warming is 0.54 K against 0.80 °C observed, while the merged plume gives < 0.01 K. Individual cases are not reproduced (FAC2 0.25, FB 0.39). The change was made after an earlier comparison, so the benchmark is not a blind test |

Two statements in the original description were also corrected: the initial vertical spread is added in quadrature rather than through ISC3's virtual distance, and the source width is now integrated exactly (Sections 5.8 and 5.10), and the penetration factor is a heuristic adapted from, not identical to, AERMOD (Section 5.6). The wind profile was also vectorized.

Numerical checks run on the corrected code:

| Check | Expected | Model result |
| --- | --- | --- |
| Heat-flux conservation at x = 2 km, no rise, class F, heat loss off: ∫∫ U ΔT dy dz | Q\_s / (ρ\_a c\_pa) = 121,232.54 K·m³/s | 121,232.54 K·m³/s |
| Radial kernel equals crosswind-integrated centerline ÷ 2πx (2 km) | 0.024546 K | 0.024546 K (area treatment) |
| Well-mixed limit, class B, z\_i = 500 m, x = 20 km | Q\_s / (ρ c\_p U z\_i) × lateral factor at y = 0 = 0.026065 K | 0.026065 K |
| Neutral final rise vs ISC3 closed form | 21.425 F^(3/4)/U for F < 55 | Identical algebraically |
| Momentum-dominated jet (ΔT = 0.5 K, w0 = 20 m/s, class D) | Momentum rise exceeds buoyant rise | 50.5 m momentum rise used |
| Heat-flux conservation in the area treatment, class F, x = 2 km, heat loss off | 121,232.54 K·m³/s | 121,232.54 K·m³/s |
| Area-regime magnitude vs box model (135 MW in a 150 m layer, U = 2 m/s, 700 m wide) | About 0.5 K layer average; higher near the ground in a shallower layer | 0.48 K at 1 km in an 85.5 m layer: consistent |
| Heat budget with loss, class F, x = 2 km: remaining flux ÷ emitted flux | Equal to F\_loss = 0.9619 | 0.9619 |
| Blend continuity: Section 7 campus scaled from 60 to 600 m | No jumps | w rises smoothly 0.77 → 0.99; ΔT at 1 km rises smoothly 0.04 → 0.48 K |
| Image sum (N = 5) vs well-mixed limit at the σz = 1.6 z\_i switch | Negligible discontinuity | Relative difference 4–5 × 10⁻⁶ |
| Mixing-line maximum RH, saturated tower exhaust | No condensation on a dry summer night; condensation on a cold night | 0.95 at 32 °C/20% RH; 1.04 at 5 °C/40% RH (warning raised) |
| Heat-flux conservation with 3 along-wind strips, heat loss off, x = 2 km | Ratio 1 | 1.0000 |
| Along-wind strip convergence, n = 2 to 5 | Converged at small n | Identical to three decimals from 0.35 to 5 km |
| Heat loss vs uniform-layer estimate exp(−v\_h d / (U\_dil h\_mix)), 5 km | 0.895 | 0.899 |
| cooling\_tower\_source energy balance, 125 MW, Section 7 night | Q\_s + Q\_l from source\_terms() = 125 MW | 124.93 MW (0.06% low) |
| NaN handling: x = \[1000, NaN, 5000\] m | Same values as without the NaN; NaN at the NaN receptor | 0.486, NaN, 0.161 K |
| Fan-array rise (test T13): one fan (q\_f ≥ Q) vs `fan_heat=None` | Identical rise | Identical to 1e-9 |
| Fan-array rise: q\_f = 10, 1, 0.1 MW for a 100 MW array | Rise decreases with smaller fans, stays ≥ the rise of n independent plumes | Monotonic; bound holds |

Most checks above were run on the earlier direct-input dry cooler (135.5 MW, 10,500 m³/s); they verify the code, not the Section 7 numbers, and remain valid for the current version.

Agreement with field measurements has been checked against the only published dataset, four single daytime vehicle traverses near Phoenix data centers \[Sailor et al. 2026\], with the Chang and Hanna (2004) statistics (`validation/benchmarks/benchmark_sailor2026.py`, `tutorial/02_validation_sailor2026.ipynb`). The model predicts the observed order of magnitude (0.54 K against 0.80 °C on average) but not the individual cases (FAC2 0.25). Not yet verified: individual-case skill, which needs repeated measurements with on-site wind; the night-time mixed layer, the decay beyond 0.5 km, and agreement with a regulatory model (AERMOD) on an identical case.

### 8.2 Response to external review

An external review raised ten comments. Six led to changes in the code or the stated scope, three were tested and did not hold as stated, and the encroachment concern was kept as a documented assumption.

| Comment | Verdict | Evidence | Action |
| --- | --- | --- | --- |
| Area source anchored at the footprint centre; should use a virtual distance | Partly valid | Moving the release to the downwind edge changes ΔT at 1 km by +8%. Virtual distance and quadrature are both established conventions (ISC3 used virtual distances); for class F a virtual distance would raise far-field ΔT by about 15% | Release moved to the downwind edge (defect 12); quadrature kept and the choice documented (Section 5.13) |
| Crosswind erf integral double-counts the source width | Not valid as stated | The crosswind width is integrated once and exactly. The real gap is the along-wind extent; integrating it would raise, not lower, near-edge values, because the downwind half of the source is closer | Covered by the downwind-edge release, where all heat is already in the mixed layer |
| Encroachment model physically infeasible in 270 s | Overstated | The same h ∝ x^(1/2) form is the standard thermal internal boundary-layer model \[Venkatram 1977\]. Only about three convective turnovers occur, so uniform mixing is an idealization | Documented as an assumption with recommended sensitivity tests (Section 5.13) |
| No surface heat sink; far field unusable | Valid, severity overstated | Estimated and now modeled loss is 3% at 1 km and 15% at 5 km | Heat-loss term added (defect 13, Section 5.14) |
| Inner function lateral shadows the argument | Valid | Worked only because of statement order | Renamed (defect 16) |
| 2-D broadcasting produces misaligned masks | Not valid | NumPy broadcasting is deterministic; 2-D heat-flux conservation checks pass | Inputs now broadcast explicitly for clarity (defect 16) |
| Image sum jumps at σz = 1.6 z\_i | Not valid | Relative difference at the switch is 4–5 × 10⁻⁶ | None |
| Hard area/point threshold | Valid | Results could change by orders of magnitude across the threshold | Smooth blend (defect 14) |
| Saturated tower plumes condense immediately | Valid in cold or humid air, not in dry summer air | Mixing-line maximum RH: 0.95 on the summer night, 1.04 on a 5 °C night | Mixing-line check and warning (defect 15); wet-plume models recommended for those cases |
| Narrow the claimed scope | Valid |  | Section 6 states that the model is a screening tool and that far-field values depend on the calibrated σθ and v\_h |

### 8.3 Second-round review

The reviewer's second response mostly endorsed Section 8.2. Its practical suggestions were tested as follows.

| Suggestion | Verdict | Evidence | Action |
| --- | --- | --- | --- |
| Uniform-layer heat-loss formula exp(−C\_H x / h\_mix) | Adopted as a cross-check, not as the method | It agreed with the corrected model (0.895 vs 0.899 at 5 km) and exposed defect 17. As written it uses one wind for both surface transfer and plume transport, which differ 2.2-fold on the Section 7 night | Defect 17 fixed; general depletion integral kept (Section 5.14) |
| Gauss–Legendre strips for the along-wind extent | Adopted as an option | Strips lower ΔT by 6–9% relative to the edge release, which is therefore the upper near-field estimate; n = 2 is already converged | `alongwind_strips` argument (Section 5.13) |
| Logistic blend on width / max(h\_mix, 100 m) | Not adopted | Re-introduces the arbitrary 100 m length; the current R uses the heights the heat actually reaches | Exponent sensitivity documented instead (Section 5.13) |
| Validate radial-mode inputs | Adopted |  | Warning when a non-zero y is passed in radial mode |
| Rename the log as a review ledger | Adopted |  | Section 8 retitled |

Two statements in the reviewer's response should not be carried into a paper without checking: that AERMOD combines initial dispersion of volume sources in quadrature (to be confirmed in the AERMOD model formulation document before citing), and the use of the background nighttime surface flux (−30 to −50 W/m²) to size the heat loss, which depends instead on the additional exchange caused by the temperature excess.

### 8.4 Third-round review

The third response found no physical issues; its points were checked as follows.

| Comment | Verdict | Evidence | Action |
| --- | --- | --- | --- |
| Strip mode gives flank receptors partial heat; diagnostics come from the last strip | Valid, partly a modeling choice | Flank receptors get 0.08–0.16 K in strip mode and 0 with edge release; strip mode is the more physical of the two there | Defect 19; flank behaviour documented (Section 5.13) |
| NaN input could break the heat-loss grid | Valid and more serious than stated | One NaN silently disabled heat loss for all receptors and returned 0 at the NaN receptor | Defect 18 |
| Bisection bounds and ice saturation | Valid, minor |  | Defect 20; supercooled-water saturation documented (Section 3.4) |
| Explain R as a source aspect ratio | Adopted |  | Section 5.13 |
| C\_H ≈ 0.003–0.005 is typical for stable desert nights | Range adopted for sensitivity; claim corrected | Neutral desert values are about 0.0025–0.004 and stability lowers them, so the default 0.004 is an upper value | Range 0.001–0.004 and its effect documented (Section 5.14) |

## 9. References

These are the standard sources for each equation, compiled from memory rather than retrieved; confirm editions and page numbers before citing them in a paper.

- ASHRAE (2021). *ASHRAE Handbook—Fundamentals*, Chapter 1: Psychrometrics. ASHRAE, Atlanta. (Section 5.1: pressure vs elevation, moist-air relations.)
- Briggs, G.A. (1969). *Plume Rise*. AEC Critical Review Series, TID-25075. U.S. Atomic Energy Commission. (Section 5.5: momentum rise.)
- Briggs, G.A. (1973). *Diffusion Estimation for Small Emissions*. ATDL Contribution File No. 79, NOAA Atmospheric Turbulence and Diffusion Laboratory, Oak Ridge, TN. (Section 5.7: σ formulas.)
- Briggs, G.A. (1975). Plume rise predictions. In D.A. Haugen (Ed.), *Lectures on Air Pollution and Environmental Impact Analyses*, pp. 59–111. American Meteorological Society, Boston. (Sections 5.2, 5.5: buoyancy flux, final and gradual rise.)
- Briggs, G.A. (1974). Plume rise from multiple sources. In *Cooling Tower Environment—1974*, CONF-740302, pp. 161–179. U.S. Energy Research and Development Administration. (Section 5.5: multiple-source enhancement M and spacing parameter S.)
- Briggs, G.A. (1984). Plume rise and buoyancy effects. In D. Randerson (Ed.), *Atmospheric Science and Power Production*, DOE/TIC-27601. U.S. Department of Energy. (Section 5.5: multiple-source plume merging.)
- Buck, A.L. (1981). New equations for computing vapor pressure and enhancement factor. *Journal of Applied Meteorology*, 20(12), 1527–1532. (Section 5.1; the code uses Buck's revised 1996 coefficients.)
- Cimorelli, A.J., et al. (2005). AERMOD: A dispersion model for industrial source applications. Part I: General model formulation and boundary layer characterization. *Journal of Applied Meteorology*, 44(5), 682–693. (Section 5.6: penetration factor.)
- Gifford, F.A. (1961). Use of routine meteorological observations for estimating atmospheric dispersion. *Nuclear Safety*, 2(4), 47–51. (Section 5.7: origin of the Pasquill–Gifford curves.)
- Hanna, S.R., Briggs, G.A., and Hosker, R.P. (1982). *Handbook on Atmospheric Diffusion*. DOE/TIC-11223. U.S. Department of Energy. (Section 5.7: rural and urban σ formulas.)
- Pasquill, F. (1961). The estimation of the dispersion of windborne material. *Meteorological Magazine*, 90, 33–49. (Sections 3.2, 5.7: stability classes.)
- Pasquill, F. (1976). *Atmospheric Dispersion Parameters in Gaussian Plume Modeling, Part II*. EPA-600/4-76-030b. U.S. EPA. (Section 5.8: buoyancy-induced dispersion Δh/3.5.)
- Rogers, R.R., and Yau, M.K. (1989). *A Short Course in Cloud Physics*, 3rd ed. Pergamon Press. (Section 5.1: latent heat.)
- Seinfeld, J.H., and Pandis, S.N. (2016). *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change*, 3rd ed. Wiley. (Sections 5.9–5.10: Gaussian plume and reflections.)
- Stull, R.B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers. (Sections 5.1, 5.4: virtual temperature, stability.)
- Turner, D.B. (1994). *Workbook of Atmospheric Dispersion Estimates: An Introduction to Dispersion Modeling*, 2nd ed. Lewis Publishers. (Sections 5.8–5.9: initial spreads, well-mixed limit.)
- U.S. EPA (1995). *User's Guide for the Industrial Source Complex (ISC3) Dispersion Models, Volume II: Description of Model Algorithms*. EPA-454/B-95-003b. (Sections 5.2–5.5, 5.8, 5.10: wind exponents, default dθ/dz, plume rise, volume sources, sector averaging.)

Added for Section 5.13:

- Batchvarova, E., and Gryning, S.-E. (1991). Applied model for the growth of the daytime mixed layer. *Boundary-Layer Meteorology*, 56, 261–274. (Entrainment factor for convective layer growth.)
- Deardorff, J.W. (1970). Convective velocity and temperature scales for the unstable planetary boundary layer and for Rayleigh convection. *Journal of the Atmospheric Sciences*, 27(8), 1211–1213. (Convective velocity w\*.)

Added for Section 5.8:

- Draxler, R.R. (1976). Determination of atmospheric diffusion parameters. *Atmospheric Environment*, 10(2), 99–105. (Time-dependent factor f\_y.)
- Irwin, J.S. (1983). Estimating plume dispersion: a comparison of several sigma schemes. *Journal of Climate and Applied Meteorology*, 22(1), 92–114. (σθ-based σy.)
- U.S. EPA (2000). *Meteorological Monitoring Guidance for Regulatory Modeling Applications*. EPA-454/R-99-005. (σθ stability classification; computing σθ from wind records.)

Added for model evaluation (Sections 5.5 and 8):

- Chang, J.C., and Hanna, S.R. (2004). Air quality model performance evaluation. *Meteorology and Atmospheric Physics*, 87, 167–196. (FAC2, FB, NMSE, MG and acceptance criteria.)
- Sailor, D.J., Samareh Abolhassani, S., and Martin, E.P. (2026). Data center waste heat as an emerging urban thermal hazard: first field measurements of neighborhood-scale air temperature impacts. *ASME Journal of Engineering for Sustainable Buildings and Cities*, 7(2), 024501. https://doi.org/10.1115/1.4071922 (CC-BY 4.0; benchmark measurements.)

Added for Sections 5.13 and 5.14:

- Chamberlain, A.C. (1953). *Aspects of Travel and Deposition of Aerosol and Vapour Clouds*. Report AERE-HP/R-1261, Atomic Energy Research Establishment, Harwell. (Source-depletion factor.)
- Garratt, J.R. (1992). *The Atmospheric Boundary Layer*. Cambridge University Press. (Bulk heat-transfer coefficients in stable conditions.)
- Horst, T.W. (1977). A surface depletion model for deposition from a Gaussian plume. *Atmospheric Environment*, 11(1), 41–46. (Source depletion for elevated releases.)
- Venkatram, A. (1977). A model of internal boundary-layer development. *Boundary-Layer Meteorology*, 11(4), 419–437. (Thermal internal boundary-layer depth.)
