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
