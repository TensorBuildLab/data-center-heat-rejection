"""
Regression suite for dc_plume.py: code-verification tests T1-T13 (listed in README.md,
"Tests"; the conservation and limit checks are derived in docs/technical_reference.md, Section 8).

Run from the repo root:  python -m pytest -q
Expected values are stored here, not computed at run time. If a deliberate model
change alters them, update the values and record the change in the technical
reference's verification ledger (Section 8).
"""
import warnings
import numpy as np
import pytest
from scipy.integrate import simpson

import dc_plume as dp
from dc_plume import Ambient, HeatSource, plume, source_terms

# ---------------------------------------------------------------- fixtures
NIGHT = dict(T=32.0, RH=0.20, U10=2.0, stability="F", mixing_height=150.0, elevation=340.0)
GEOM_DRY = dict(height=12.0, footprint_x=600.0, footprint_y=600.0, building_height=10.0)
GEOM_TOWER = dict(height=15.0, footprint_x=300.0, footprint_y=200.0, building_height=10.0)
# class -> (U10, mixing height) used for the conservation tests
CLASSES = {"B": (4.0, 1000.0), "D": (4.0, 500.0), "F": (2.0, 150.0)}


def night(**kw):
    return Ambient(**{**NIGHT, **kw})


def dry(amb=None):
    amb = amb or night()
    return dp.dry_cooler_source(137e6, 12.0, amb, **GEOM_DRY)


def tower(amb=None, inlet=1650.0):
    amb = amb or night()
    return dp.cooling_tower_source(125e6, inlet, amb, **GEOM_TOWER)


def cross_section_flux(src, amb, x0, **kw):
    """Integral of U_dil * dT over y in [-8, 8] km and z in [0, z_i] at x = x0
    (K m3/s), plus the result dict at the ground."""
    zi = amb.mixing_height
    y = np.linspace(-8000.0, 8000.0, 4001)
    z = np.linspace(0.0, zi, 601)
    Y, Z = np.meshgrid(y, z, indexing="ij")          # plume() broadcasts z with x and y
    r = plume(src, amb, np.full(Y.shape, x0), Y, z=Z, **kw)
    U = np.broadcast_to(r["U_dilution"], r["dT"].shape)
    flux = simpson(simpson(U * r["dT"], x=z, axis=1), x=y)
    return flux, r


def rel(a, b):
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------- T1
@pytest.mark.parametrize("stab", list(CLASSES))
@pytest.mark.parametrize("regime", ["area", "point"])
def test_T1_heat_flux_conservation(stab, regime):
    U10, zi = CLASSES[stab]
    amb = Ambient(T=32.0, RH=0.20, U10=U10, stability=stab, mixing_height=zi,
                  elevation=340.0, heat_loss_velocity=0.0)
    src = dry(amb)
    flux, r = cross_section_flux(src, amb, 2000.0, regime=regime)
    st = r["source"]
    expected = (1 - r["penetration_fraction"]) * st["Q_s"] / (st["rho_a"] * st["cp_a"])
    assert expected > 0
    assert rel(flux, expected) < 1e-6


# ---------------------------------------------------------------- T2
@pytest.mark.parametrize("regime", ["area", "point"])
def test_T2_heat_budget_with_loss(regime):
    amb = night()                                    # class F: default v_h > 0
    src = dry(amb)
    flux, r = cross_section_flux(src, amb, 2000.0, regime=regime)
    st = r["source"]
    emitted = (1 - r["penetration_fraction"]) * st["Q_s"] / (st["rho_a"] * st["cp_a"])
    F_loss = float(np.ravel(r["heat_loss_factor"])[0])
    assert r["heat_loss_velocity"] > 0 and F_loss < 1
    assert rel(flux / emitted, F_loss) < 1e-6


# ---------------------------------------------------------------- T3
@pytest.mark.parametrize("src_fn", [dry, tower])
def test_T3_radial_centerline_consistency(src_fn):
    amb, x0 = night(), 2000.0
    src = src_fn(amb)
    y = np.linspace(-8000.0, 8000.0, 16001)
    c = plume(src, amb, x0, y)
    crosswind = simpson(c["dT"], x=y)
    radial = plume(src, amb, np.array([x0]), mode="radial")["dT"][0]
    assert rel(crosswind / (2 * np.pi * x0), radial) < 1e-6


# ---------------------------------------------------------------- T4
T4_EXPECTED = 0.0086272   # K, Section 7 dry cooler, class B, U10 = 4 m/s, z_i = 500 m, x = 20 km, y = 0
# (the ledger's 0.026065 K was for the earlier 135.5 MW direct-input source)


def test_T4_well_mixed_limit():
    amb = Ambient(T=32.0, RH=0.20, U10=4.0, stability="B", mixing_height=500.0, elevation=340.0)
    src = dry(amb)
    r = plume(src, amb, np.array([20000.0]), regime="area")
    assert r["sigma_z"][0] > 1.6 * amb.mixing_height
    st = r["source"]
    analytic = (st["Q_s"] / (st["rho_a"] * st["cp_a"])
                * dp._lateral_profile(0.0, r["sigma_y"][0], src.footprint_y)
                / (r["U_dilution"][0] * amb.mixing_height))
    assert rel(r["dT"][0], analytic) < 1e-6
    assert r["dT"][0] == pytest.approx(T4_EXPECTED, abs=1e-6)


# ---------------------------------------------------------------- T5
@pytest.mark.parametrize("regime", ["area", "point"])
def test_T5_image_sum_continuity(regime):
    amb = Ambient(T=32.0, RH=0.20, U10=4.0, stability="B", mixing_height=500.0, elevation=340.0)
    src = dry(amb)
    target = 1.6 * amb.mixing_height
    sz = lambda x: plume(src, amb, np.array([x]), regime=regime)["sigma_z"][0]
    lo, hi = 400.0, 50000.0
    assert sz(lo) < target < sz(hi)
    for _ in range(80):                                  # bisection on sigma_z(x)
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if sz(mid) < target else (lo, mid)
    below = plume(src, amb, np.array([lo]), regime=regime)["dT"][0]
    above = plume(src, amb, np.array([hi * (1 + 1e-9)]), regime=regime)["dT"][0]
    assert rel(above, below) < 1e-4


# ---------------------------------------------------------------- T6
def test_T6_tower_energy_balance():
    amb = night()
    st = source_terms(tower(amb), amb)
    assert rel(st["Q_total"], 125e6) < 0.005


# ---------------------------------------------------------------- T7
def test_T7_blend_continuity():
    """Section 7 campus scaled from 60 to 600 m: square footprint L, heat scaled with
    footprint area (constant flux density), dT at 1 km in 10 m steps.
    Criterion: monotonic, and no jump, i.e. the local log-log slope
    d ln(dT) / d ln(L) of every step stays within [0, 2.5] (heat alone gives 2).
    Note: the slope is continuous except for a kink near L = 112 m, where the
    mixing depth switches from the w* limit to the encroachment limit."""
    amb = night()
    L = np.arange(60.0, 601.0, 10.0)
    w, dT = [], []
    for l in L:
        src = dp.dry_cooler_source(137e6 * (l / 600.0) ** 2, 12.0, amb, height=12.0,
                                   footprint_x=l, footprint_y=l, building_height=10.0)
        r = plume(src, amb, np.array([1000.0]))
        w.append(r["area_weight"]); dT.append(r["dT"][0])
    w, dT = np.array(w), np.array(dT)
    assert np.all(np.diff(w) > 0) and np.all(np.diff(dT) > 0)          # monotonic
    slope = np.diff(np.log(dT)) / np.diff(np.log(L))
    assert np.all((slope >= 0) & (slope <= 2.5))                         # no jumps
    assert (w[0], w[-1]) == pytest.approx((0.766, 0.986), abs=1e-3)
    assert (dT[0], dT[-1]) == pytest.approx((0.0456, 0.4863), abs=1e-3)


# ---------------------------------------------------------------- T8
def test_T8_strip_convergence():
    amb = night()
    src = dry(amb)
    x = np.linspace(350.0, 5000.0, 200)
    ref = plume(src, amb, x, alongwind_strips=2)["dT"]
    for n in (3, 4, 5):
        assert np.max(np.abs(plume(src, amb, x, alongwind_strips=n)["dT"] - ref)) < 1e-3


# ---------------------------------------------------------------- T9
def test_T9_momentum_rise_branch():
    """Fast, weakly buoyant jet (exhaust 0.5 K above ambient, w0 = 20 m/s, class D)."""
    amb = Ambient(T=25.0, RH=0.3, U10=3.0, stability="D", mixing_height=800.0)
    src = HeatSource(height=10.0, T_exit=25.5, flow=60.0, exit_area=3.0,
                     footprint_x=10.0, footprint_y=10.0)
    st = source_terms(src, amb)
    U_s = float(dp.wind_at(amb.U10, src.height, "D"))
    _, dh_buoy = dp.briggs_rise(st["F_b"], U_s, "D", 0.0, np.array([1.0]))
    r = plume(src, amb, np.array([1000.0]), regime="point")
    assert r["final_rise"] > dh_buoy                       # momentum branch selected
    d = np.sqrt(4 * src.exit_area / np.pi)
    assert r["final_rise"] == pytest.approx(3.0 * d * st["w0"] / U_s, rel=1e-12)
    assert r["final_rise"] == pytest.approx(39.09, abs=0.05)   # m; ledger inputs for 50.5 m not recorded


# ---------------------------------------------------------------- T10
def test_T10_robustness():
    amb = night()
    src = dry(amb)
    base = plume(src, amb, np.array([1000.0, 5000.0]))["dT"]
    # NaN in x: NaN there, other receptors unchanged (ledger defect 18)
    r = plume(src, amb, np.array([1000.0, np.nan, 5000.0]))["dT"]
    assert np.isnan(r[1]) and np.allclose(r[[0, 2]], base, rtol=0, atol=1e-12)
    # NaN in y
    r = plume(src, amb, np.array([1000.0, 1000.0]), np.array([0.0, np.nan]))["dT"]
    assert np.isfinite(r[0]) and np.isnan(r[1])
    # upwind of the source: zero; inside the footprint: NaN
    r = plume(src, amb, np.array([-2000.0, 0.0]))["dT"]
    assert r[0] == 0.0 and np.isnan(r[1])
    # radial mode uses |x|
    rr = plume(src, amb, np.array([-2000.0, 2000.0]), mode="radial")["dT"]
    assert rr[0] == pytest.approx(rr[1], rel=1e-12)
    # radial mode with y != 0 warns
    with pytest.warns(UserWarning, match="radial"):
        plume(src, amb, np.array([2000.0]), y=100.0, mode="radial")
    # lowercase stability class accepted, same result
    r = plume(src, night(stability="f"), np.array([1000.0, 5000.0]))["dT"]
    assert np.allclose(r, base, rtol=0, atol=1e-12)
    # out-of-range RH and bad arguments raise
    with pytest.raises(ValueError):
        plume(src, night(RH=20.0), np.array([1000.0]))
    with pytest.raises(ValueError):
        plume(src, night(stability="G"), np.array([1000.0]))
    with pytest.raises(ValueError):
        plume(src, night(mixing_height=0.0), np.array([1000.0]))
    # impossible tower load raises
    with pytest.raises(ValueError):
        dp.cooling_tower_source(5e9, 10.0, amb, **GEOM_TOWER)
    # U10 < 1 m/s warns
    with pytest.warns(UserWarning, match="U10 < 1"):
        plume(src, night(U10=0.5), np.array([1000.0]))


# ---------------------------------------------------------------- T11
def test_T11_mixing_line_check():
    """95%-RH tower exhaust, solved from the energy balance in each ambient.
    Baseline (2026-10-18): 0.950 summer, 1.002 winter. The ledger's 1.04 came from
    unrecorded winter exhaust inputs (e.g. saturated exhaust at 24 C gives 1.046)."""
    summer = night()
    winter = Ambient(T=5.0, RH=0.40, U10=2.0, stability="F", mixing_height=150.0, elevation=340.0)
    ts, tw = tower(summer), tower(winter)
    assert source_terms(ts, summer)["mixing_line_max_RH"] == pytest.approx(0.950, abs=1e-3)
    m_w = source_terms(tw, winter)["mixing_line_max_RH"]
    assert m_w > 1.0 and m_w == pytest.approx(1.002, abs=1e-3)
    x = np.array([1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # no warning in summer
        plume(ts, summer, x)
    with pytest.warns(UserWarning, match="mixing line"):
        plume(tw, winter, x)


# ---------------------------------------------------------------- T12
T12 = {"air": (0.486, 0.161), "water": (0.052, 0.009)}   # K at 1 and 5 km


@pytest.mark.parametrize("case", ["air", "water"])
def test_T12_reference_case(case):
    amb = night()
    src = dry(amb) if case == "air" else tower(amb)
    x = np.linspace(50, 5000, 400)                        # as in the __main__ example
    dT = plume(src, amb, x)["dT"]
    got = (np.interp(1000, x, dT), np.interp(5000, x, dT))
    assert got[0] == pytest.approx(T12[case][0], abs=1e-3)
    assert got[1] == pytest.approx(T12[case][1], abs=1e-3)


# ---------------------------------------------------------------- T13
def test_T13_fan_array_rise():
    """Fan-array rise (Briggs 1974 multiple-source merging): fan_heat=None leaves results
    unchanged; one fan per source reproduces n_plumes=1; more, smaller fans rise less, and
    the merged rise of n fans lies between one fan's rise and the single merged plume."""
    amb = Ambient(T=40.0, RH=0.15, U10=1.5, stability="A", mixing_height=2500.0, elevation=370.0)
    geom = dict(height=22.0, footprint_x=300.0, footprint_y=300.0, building_height=20.0)
    base = dp.dry_cooler_source(100e6, 11.0, amb, **geom)
    one = dp.dry_cooler_source(100e6, 11.0, amb, fan_heat=1e12, **geom)
    x = np.array([400.0])
    assert plume(base, amb, x, regime="point")["final_rise"] == pytest.approx(
        plume(one, amb, x, regime="point")["final_rise"], rel=1e-9)
    rises = [plume(dp.dry_cooler_source(100e6, 11.0, amb, fan_heat=q, **geom), amb, x,
                   regime="point")["final_rise"] for q in (1e7, 1e6, 1e5)]
    assert rises[0] > rises[1] > rises[2] > 0
    single = plume(dp.dry_cooler_source(100e6, 11.0, amb, n_plumes=1000, **geom), amb, x,
                   regime="point")["final_rise"]
    assert rises[2] >= single
    with pytest.raises(ValueError):
        plume(dp.dry_cooler_source(100e6, 11.0, amb, fan_heat=-1.0, **geom), amb, x)
