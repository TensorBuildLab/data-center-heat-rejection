# Benchmark protocol: Sailor et al. (2026) traverses

Written on 2026-10-19, before the geometry-resolved benchmark (`benchmark_sailor2026.py`, version 2) was run. All inputs below are fixed by these rules, and none were changed after the results were seen.

## History

Version 1 (repository commit f577c8e) represented each facility as a square-equivalent source of its site or floor area. It compared the model's mean ΔT at generic distances of 100–500 m with the published values. Its results were seen before this protocol was written: FAC2 0.75, FB 0.15, with case errors from −60 % to +83 %.

Re-reading the source article showed three set-up errors in version 1, not tuning targets:

1. **Geometry.** The heat-rejection areas are individual buildings, not whole sites or square equivalents. The published averaging boxes lie 15–570 m from the nearest building (Fig. 2 of the article). For CyrusOne on 8 August they lie 15–90 m away, not the 100–400 m assumed.
2. **Observable for NTT PH1 (case e).** Table 1 of the article has no upwind box for this case. The published ΔT ≈ 0.9 °C is the contrast between the box nearest the facility (T_d3 = 25.5 °C) and the farthest box in the neighbourhood (T_d2 = 24.6 °C). It is not a downwind-minus-upwind difference.
3. **Upwind reference.** The published ΔT values are differences between two boxes, so the model must be evaluated as the same difference.

## Rules (version 2)

| Item | Rule | Source |
| --- | --- | --- |
| Heat-rejection area | The hatched facility polygons of Fig. 2, digitised with each panel's scale bar (precision about ±15 m), as rectangles or polygons | Sailor et al. (2026), Fig. 2 |
| Heat per building | Facility IT capacity × load fraction × 1.3 (PUE), shared between a facility's polygons in proportion to area | Capacity and PUE: Sailor et al. (2026) §1–2; Digital Realty capacity: listing cited in `sailor2026_cases.csv` |
| Load fraction | 0.75 central; 0.5–1.0 range (as version 1) | Not published; typical colocation utilisation |
| Building height | Single-storey 12 m, two-storey 20 m. The storey count is taken from the article where it is stated (NTT PH1: two storeys). Otherwise a facility is single-storey when its published floor area is ≤ 1.1 × its hatched footprint. | Floor areas: Sailor et al. (2026) §1–2 and listing |
| Fan outlet | Roof + 2 m; exhaust 11 K above ambient at 3 m/s; 80 kW per fan (50–150 kW range) | Sailor et al. (2026) §1 (8–14 K, 2–4 m/s); version 1 values unchanged |
| Weather | Published wind speed, direction and air temperature (upwind box mean). Turner stability class for the date and time; mixing height 1,500 m in October and 2,500–3,000 m in summer (as version 1) | Sailor et al. (2026) §3 |
| Model output | ΔT at 2 m from each polygon, superposed. Each polygon is represented by its along-wind and crosswind extents in the wind frame. Receptors are sampled every 5 m along the traverse path inside each published averaging box (whole box area for the rotated box of case a). Receptors inside a facility polygon are skipped. | – |
| Compared quantity | Mean modelled ΔT in the downwind box minus that in the upwind box (cases a, b, d); T_d3 box minus T_d2 box (case e) | Sailor et al. (2026) Table 1, §3.4 |
| Statistics | FAC2, FB, NMSE, MG (Chang and Hanna 2004) on the central predictions; observation inside the input-uncertainty range | – |

Case c (Aligned) remains excluded because no capacity is published for it.

## Amendments after the first version-2 run (2026-10-19)

Two implementation bugs were found by inspecting box-by-box values. Neither changes a rule above.

1. A receptor beside a polygon in the crosswind direction but upwind of the polygon's centre was given heat. This put 0.01–0.15 K into the upwind boxes. It now receives none.
2. Panel (b) of the article's Fig. 2 shows only part of the CyrusOne campus. The campus heat was therefore shared over the visible buildings only, and the floor-area rule then classed them as two-storey. Case b now uses the heat density and storey count of the full campus, digitised in panel (a).

## Result of version 2 (after the amendments)

| Case | Observed (°C) | Predicted (K), central [range] | Ratio |
| --- | --- | --- | --- |
| a | 0.8 | 0.50 [0.36–0.64] | 0.62 |
| b | 0.5 | 1.32 [0.67–2.02] | 2.63 |
| d | 1.0 | 0.26 [0.16–0.41] | 0.26 |
| e (near − far) | 0.9 | 0.08 [0.06–0.12] | 0.09 |

FAC2 0.25, FB 0.39, NMSE 1.14, MG 2.25. None of the four observations lies inside the input-uncertainty range. The fully merged plume predicts < 0.01 K in every case.

## Post-hoc analyses (2026-10-21 to 2026-10-25)

These analyses were run after the version-2 results above had been seen. None changes a rule of the benchmark or its central result. They report what range of predictions the published inputs allow, and what could explain the residuals.

### Wind-direction and load uncertainty (`wind_uncertainty_sailor2026.py`)

The traverse winds come from airport stations several kilometres from the sites. Daytime hours (10–15 MST) with 1–3 m/s at both stations of four Phoenix-area ASOS pairs (KCHD, KIWA, KFFZ, KPHX; IEM, 2019–2025) give a median absolute direction difference of 40–60°. Only 30–41 % of hours agree within 30°. That corresponds to about 35° between one airport and a site; 50° is run as a sensitivity case. Each case is run for offsets of −90° to +90° (Gaussian weights) and load fractions of 0.5, 0.75 and 1.0.

| Case | Observed | Central | 5–95 % (σ = 35°) | Percentile of observation | P(within 15 %) |
| --- | --- | --- | --- | --- | --- |
| a | 0.8 | 0.50 | −0.12 to 0.82 | 92 | 0.13 |
| b | 0.5 | 1.32 | 0.13 to 1.96 | 20 | 0.06 |
| d | 1.0 | 0.26 | 0.02 to 0.38 | above | 0 |
| e | 0.9 | 0.08 | −0.09 to 0.57 | above | 0 |

Two of the four observations lie inside the 5–95 % range. The results with σ = 50° are almost the same (`results/benchmarks/wind_uncertainty.csv`). A downwash upper bound (`case_studies_sailor2026.py`), with all heat held at roof + 2 m and no rise, gives a 0.68, b 1.99, d 0.68 and e 0.15 K. Missing downwash physics therefore cannot explain cases d and e.

### Per-case diagnosis and neighbour check (`case_studies_sailor2026.py`)

The table gives the single input change that would reproduce each observation, compared with its plausible range (direction within ±70°, i.e. 2σ; load 0.5–1.0; roof 8–24 m):

| Case | Wind direction | Load fraction | Roof height |
| --- | --- | --- | --- |
| a | +30° gives 0.79 K (inside 1σ) | 1.26 (outside) | no effect (0.50–0.51 K) |
| b | −34° (inside 1σ) | 0.38 (outside) | 24 m gives 1.03 K; not reached |
| d | none: maximum 0.34 K over ±90° | 1.85 (outside) | 0.22–0.31 K; not reached |
| e | none: maximum 0.53 K at +30° | > 3 (outside) | 0.08–0.11 K; not reached |

The measured point-to-point spread between the two areas (Table 1 of the article) runs from −0.6 to +2.2 °C (a), −0.1 to +1.2 (b), +0.6 to +1.4 (d) and +0.4 to +1.4 (e).

Neighbour check: some buildings visible in the article's Fig. 2 are not the studied facility. The table gives the heat each would need to reject to close the gap. These are hypotheses, not corrections, and the benchmark sources are unchanged.

| Case | Building | Area (m²) | ΔT difference per 10 MW (K) | Heat needed (MW) |
| --- | --- | --- | --- | --- |
| d | about 90 m north of the facility | 14,700 | 0.030 | about 250 |
| d | about 50 m south of the facility | 7,700 | 0.181 | about 40 |
| e | west of the facility | 44,300 | −0.041 | – (reduces the contrast) |
| e | south of the facility | 9,700 | 0.002 | about 5,000 |

A neighbouring data center of about 40 MW south of Digital Realty could explain case d. No plausible neighbour explains case e.

### Case c (Aligned, Chandler)

A web search on 2026-10-25 (Aligned, Baxtel, DataCenterHawk, Data Center Frontier, datacentermap and City of Chandler records) found floor areas for Aligned's Price Road buildings (PHX-04: 396,343 sq ft) and a 400 MW total for two new Phoenix-area campuses. It found no IT capacity for the facility in the article's Fig. 2c. Case c therefore remains excluded.

### Other datasets

A literature search in English and Chinese, covering journals, theses, news of the Sailor et al. study, cooling-tower and condenser-array studies and wind-farm analogues, found no other published air-temperature measurements near data centers. Satellite studies report land surface temperature, not air temperature. No dataset was added.

### Building downwash sensitivity (`benchmark_sailor2026.py --downwash`, 2026-10-25)

The fans release about 2 m above the roofs, which is below the GEP height h_b + 1.5 L_B. The model was therefore run with the optional ISC3 building-downwash treatment (Schulman–Scire reduced rise and Huber–Snyder wake spreads). The rules above are otherwise unchanged.

| Case | Observed | Without downwash | With downwash | Change |
| --- | --- | --- | --- | --- |
| a | 0.8 | 0.499 | 0.505 | +0.006 |
| b | 0.5 | 1.316 | 1.512 | +0.196 |
| d | 1.0 | 0.259 | 0.285 | +0.026 |
| e | 0.9 | 0.083 | 0.099 | +0.016 |

FAC2 stays at 0.25 and FB changes from 0.39 to 0.29. Downwash does not change the comparison, so the benchmark keeps the default model.
