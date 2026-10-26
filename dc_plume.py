"""
dc_plume.py
Steady-state Gaussian plume model for data-center heat and moisture rejection.
Technical reference: docs/technical_reference.md (equations, assumptions,
verification ledger), kept in sync with this file.

Source treatment (regime="auto" blends the two smoothly by source width):
  - AREA: the rejected heat mixes its own convective layer over the footprint
    (encroachment / w* limits), released as a strip at the DOWNWIND EDGE.
  - POINT: Briggs buoyancy (virtual temperature, so moist plumes are handled)
    and momentum plume rise from the footprint centre, with partial penetration
    of the inversion.
  - Mixing-line saturation check flags condensing (visible) plumes.
Dispersion:
  - lateral: sigma_y from the 1-hour wind-direction fluctuation sigma_theta with
    Draxler's factor (hourly-average plume, includes meander); Pasquill-Gifford
    sigma_y available with lateral="pg"
  - crosswind source width integrated exactly (uniform strip, erf profile)
  - vertical: Pasquill-Gifford/Briggs sigma_z, reflection at ground and mixing height
  - heat loss to the surface and by longwave radiation on stable nights
    (Chamberlain-Horst source depletion, transfer velocity v_h)

Outputs downwind distributions (1-D or 2-D) of temperature rise dT, specific
humidity rise dq, absolute T, and RH at a receptor height.

Key assumptions (state these in any paper):
  - Steady state (1-hour average), horizontally homogeneous atmosphere, flat terrain.
  - After rise/mixing, excess heat and vapor disperse as passive scalars; heat
    loss is represented only through an approximate transfer velocity.
  - No condensation (a flag is raised where receptor RH > 100%).
  - Gaussian theory is unreliable for U < 1 m/s (wind is clipped to 1 m/s).
"""
import math
import warnings
import numpy as np
try:
    from scipy.special import erf as _erf
except ImportError:                      # numpy-only fallback
    _erf = np.vectorize(math.erf, otypes=[float])
from dataclasses import dataclass
from typing import Optional

G, RD, CP_D, CP_V = 9.81, 287.05, 1004.67, 1860.0

# ---------------------------------------------------------------- psychrometrics
def p_sat(T_C):
    """Saturation vapor pressure over water, Pa (Buck)."""
    return 611.21 * np.exp((18.678 - T_C / 234.5) * (T_C / (257.14 + T_C)))

def q_from_RH(T_C, RH, P):
    e = RH * p_sat(T_C)
    return 0.622 * e / (P - 0.378 * e)

def RH_from_q(T_C, q, P):
    e = q * P / (0.622 + 0.378 * q)
    return e / p_sat(T_C)

def Tv_K(T_C, q):
    return (T_C + 273.15) * (1.0 + 0.608 * q)

def rho_moist(T_C, q, P):
    return P / (RD * Tv_K(T_C, q))

def latent_heat(T_C):
    return (2.501 - 0.00237 * T_C) * 1e6

def pressure_from_elevation(z_m):
    return 101325.0 * (1.0 - 2.25577e-5 * z_m) ** 5.25588

# ------------------------------------------------- dispersion coefficients (Briggs)
_SIG_RURAL = {
    "A": (lambda x: 0.22*x*(1+1e-4*x)**-0.5, lambda x: 0.20*x),
    "B": (lambda x: 0.16*x*(1+1e-4*x)**-0.5, lambda x: 0.12*x),
    "C": (lambda x: 0.11*x*(1+1e-4*x)**-0.5, lambda x: 0.08*x*(1+2e-4*x)**-0.5),
    "D": (lambda x: 0.08*x*(1+1e-4*x)**-0.5, lambda x: 0.06*x*(1+1.5e-3*x)**-0.5),
    "E": (lambda x: 0.06*x*(1+1e-4*x)**-0.5, lambda x: 0.03*x*(1+3e-4*x)**-1),
    "F": (lambda x: 0.04*x*(1+1e-4*x)**-0.5, lambda x: 0.016*x*(1+3e-4*x)**-1),
}
_SIG_URBAN = {
    "A": (lambda x: 0.32*x*(1+4e-4*x)**-0.5, lambda x: 0.24*x*(1+1e-3*x)**0.5),
    "B": (lambda x: 0.32*x*(1+4e-4*x)**-0.5, lambda x: 0.24*x*(1+1e-3*x)**0.5),
    "C": (lambda x: 0.22*x*(1+4e-4*x)**-0.5, lambda x: 0.20*x),
    "D": (lambda x: 0.16*x*(1+4e-4*x)**-0.5, lambda x: 0.14*x*(1+3e-4*x)**-0.5),
    "E": (lambda x: 0.11*x*(1+4e-4*x)**-0.5, lambda x: 0.08*x*(1+1.5e-3*x)**-0.5),
    "F": (lambda x: 0.11*x*(1+4e-4*x)**-0.5, lambda x: 0.08*x*(1+1.5e-3*x)**-0.5),
}
# wind-profile power-law exponents (rural, urban) and default stable dθ/dz (K/m)
_P_EXP = {"A": (0.07, 0.15), "B": (0.07, 0.15), "C": (0.10, 0.20),
          "D": (0.15, 0.25), "E": (0.35, 0.30), "F": (0.55, 0.30)}
_DTHETA = {"E": 0.020, "F": 0.035}
# default 1-hour standard deviation of horizontal wind direction (deg).
# A-D: mid-points of the EPA sigma-theta classification ranges; E-F: 15 deg, a
# placeholder for wind meander in light-wind nights (replace with measured values)
_SIGMA_THETA = {"A": 25.0, "B": 20.0, "C": 15.0, "D": 10.0, "E": 15.0, "F": 15.0}

# ------------------------------------------------------------------- inputs
@dataclass
class HeatSource:
    height: float               # m, fan outlet height above ground
    T_exit: float               # °C, exhaust air temperature
    flow: float                 # m3/s, TOTAL exhaust airflow at exit conditions
    exit_area: float            # m2, TOTAL fan outlet area (sets exit velocity)
    footprint_x: float          # m, along-wind size of the heat-rejection area
    footprint_y: float          # m, crosswind size
    RH_exit: Optional[float] = None  # 0-1; None = same specific humidity as ambient (dry cooler)
    n_plumes: int = 1           # 1 = all fans merge (max rise); N = independent plumes (min rise)
    building_height: float = 0.0  # m, adds initial vertical spread (wake mixing)
    fan_heat: Optional[float] = None  # W rejected per fan or tower cell (sensible + latent).
                                      # When set, plume rise follows Briggs' (1974) multiple-
                                      # source merging for the fan array (_point_rise) and
                                      # n_plumes is ignored. None: n_plumes behaviour.
    downwash: bool = False      # building downwash of the point plume (ISC3: Schulman-Scire
                                # reduced rise and linear decay factor, Huber-Snyder wake
                                # sigmas) when the release is within the building wake.
                                # Needs building_height > 0. False: wake only through sigma_z0.

@dataclass
class Ambient:
    T: float                    # °C
    RH: float                   # 0-1
    U10: float                  # m/s, equivalent wind speed at 10 m
    stability: str              # Pasquill class 'A'..'F'
    mixing_height: float        # m (night: stable BL depth; day: convective BL depth)
    elevation: float = 0.0      # m above sea level (Phoenix ~340, Tucson ~730)
    urban: bool = False
    dtheta_dz: Optional[float] = None  # K/m, override default for E/F
    sigma_theta: Optional[float] = None  # deg, 1-hour std of wind direction (measured if possible)
    heat_loss_velocity: Optional[float] = None  # m/s, transfer velocity of excess heat to the
                                                # surface + longwave; None = default (E-F only)

# ------------------------------------------------------------------- source
def source_terms(src: HeatSource, amb: Ambient) -> dict:
    """Heat and moisture fluxes and buoyancy flux implied by the exhaust state.
    No separate power input is needed: Q follows from flow, T, and humidity."""
    P = pressure_from_elevation(amb.elevation)
    q_a = q_from_RH(amb.T, amb.RH, P)
    q_e = q_a if src.RH_exit is None else q_from_RH(src.T_exit, src.RH_exit, P)
    rho_e = rho_moist(src.T_exit, q_e, P)
    rho_a = rho_moist(amb.T, q_a, P)
    w_a, w_e = q_a / (1 - q_a), q_e / (1 - q_e)          # mixing ratios
    m_da = rho_e * src.flow / (1 + w_e)                   # dry-air mass flow, kg/s
    Q_s = m_da * (CP_D + w_e * CP_V) * (src.T_exit - amb.T)   # W, sensible
    E = m_da * (w_e - w_a)                                # kg/s, vapor added
    Q_l = E * latent_heat(amb.T)                          # W, latent
    Tv_e, Tv_a = Tv_K(src.T_exit, q_e), Tv_K(amb.T, q_a)
    F_b = G * src.flow / np.pi * (Tv_e - Tv_a) / Tv_e     # m4/s3, Briggs buoyancy flux
    # highest RH reached anywhere on the mixing line between exhaust and ambient air;
    # > 1 means the plume condenses (visible plume / fog) before it is diluted
    f = np.linspace(0.0, 1.0, 201)
    RH_line = RH_from_q(f * src.T_exit + (1 - f) * amb.T, f * q_e + (1 - f) * q_a, P)
    return dict(P=P, q_a=q_a, q_e=q_e, rho_a=rho_a, rho_e=rho_e, m_da=m_da,
                mixing_line_max_RH=float(RH_line.max()),
                Q_s=Q_s, Q_l=Q_l, Q_total=Q_s + Q_l, E=E, F_b=F_b,
                w0=src.flow / src.exit_area, cp_a=CP_D * (1 - q_a) + CP_V * q_a)

def wind_at(U10, z, stability, urban=False, U_min=1.0):
    """Power-law wind profile U(z) = U10 (z/10)^p for z >= 10 m; U10 below 10 m.
    Clipped at U_min because Gaussian theory fails in calm conditions."""
    p = _P_EXP[stability][1 if urban else 0]
    z = np.maximum(np.asarray(z, float), 10.0)
    return np.maximum(U10 * (z / 10.0) ** p, U_min)

def briggs_rise(F, U, stability, s, x):
    """Gradual and final BUOYANT plume rise (m) for buoyancy flux F of one plume."""
    x = np.asarray(x, float)
    if F <= 0:
        return np.zeros_like(x), 0.0
    if stability in ("E", "F") and s > 0:
        dh_f = min(2.6 * (F / (U * s)) ** (1 / 3), 4.0 * F ** 0.25 * s ** -0.375)
    else:
        x_star = 14 * F ** 0.625 if F < 55 else 34 * F ** 0.4
        dh_f = 1.6 * F ** (1 / 3) * (3.5 * x_star) ** (2 / 3) / U
    dh_x = np.minimum(1.6 * F ** (1 / 3) * np.maximum(x, 0) ** (2 / 3) / U, dh_f)
    return dh_x, dh_f

def momentum_rise(w0, d, Ts_K, Ta_K, U, stability, s):
    """Final MOMENTUM (jet) rise (m) of one plume, Briggs (1969, 1975) as in ISC3."""
    dh_neutral = 3.0 * d * w0 / U
    if stability in ("E", "F") and s > 0:
        F_m = w0 ** 2 * (d / 2) ** 2 * Ta_K / Ts_K
        return min(dh_neutral, 1.5 * (F_m / (U * np.sqrt(s))) ** (1 / 3))
    return dh_neutral

def convective_mixing_depth(B_k, t, gamma, zi, T_K, A=0.2):
    """Depth (m) of the convective layer that a surface buoyancy flux B_k (K m/s)
    builds in time t (s), the smallest of three limits:
      - encroachment against a stable gradient gamma (K/m) with entrainment
        factor (1+2A), A ~ 0.2:  h = sqrt(2 (1+2A) B_k t / gamma)
        (Stull 1988; Batchvarova & Gryning 1991)
      - convective velocity limit h = w* t with w* = (g B_k h / T)^(1/3)
        (Deardorff 1970), i.e. h = (g B_k / T)^(1/2) t^(3/2)
      - the mixing height zi."""
    if B_k <= 0:
        return 0.0
    h_w = np.sqrt(G * B_k / T_K) * t ** 1.5
    h_enc = np.sqrt(2 * (1 + 2 * A) * B_k * t / gamma) if gamma > 0 else np.inf
    return float(min(h_enc, h_w, zi))


# ------------------------------------------------------------------- helpers
def _enthalpy(T_C, q):
    """Moist-air enthalpy per kg of dry air, J/kg."""
    w = q / (1 - q)
    return CP_D * T_C + w * (2.501e6 + CP_V * T_C)


def dry_cooler_source(Q_W, dT_air, amb, **geometry):
    """HeatSource for a dry cooler that rejects Q_W watts with an air-side
    temperature rise dT_air (K). Exhaust temperature follows the ambient, and the
    airflow is whatever carries Q_W at that rise. geometry: height, exit_velocity
    (m/s, default 6), footprint_x, footprint_y, building_height, n_plumes."""
    P = pressure_from_elevation(amb.elevation)
    q_a = q_from_RH(amb.T, amb.RH, P)
    T_e = amb.T + dT_air
    w_a = q_a / (1 - q_a)
    m_da = Q_W / ((CP_D + w_a * CP_V) * dT_air)
    flow = m_da * (1 + w_a) / rho_moist(T_e, q_a, P)
    w0 = geometry.pop("exit_velocity", 6.0)
    return HeatSource(T_exit=T_e, flow=flow, exit_area=flow / w0, RH_exit=None, **geometry)


def cooling_tower_source(Q_W, inlet_flow, amb, RH_exit=0.95, **geometry):
    """HeatSource for a cooling tower that rejects Q_W watts (sensible + latent)
    with inlet airflow inlet_flow (m3/s at ambient conditions). The exhaust state
    follows from the tower energy balance
        Q_W = m_da (h_exit - h_amb) - E c_w T_amb,   E = m_da (w_exit - w_amb),
    (make-up water assumed at ambient temperature) with the exhaust at RH_exit, so
    it responds to the ambient wet-bulb. geometry as for dry_cooler_source
    (default exit_velocity 8 m/s)."""
    P = pressure_from_elevation(amb.elevation)
    q_a = q_from_RH(amb.T, amb.RH, P)
    w_a = q_a / (1 - q_a)
    m_da = rho_moist(amb.T, q_a, P) * inlet_flow / (1 + w_a)
    h_a = _enthalpy(amb.T, q_a)

    def rejected(T):
        q = q_from_RH(T, RH_exit, P)
        return m_da * (_enthalpy(T, q) - h_a) - m_da * (q / (1 - q) - w_a) * 4186.0 * amb.T

    lo, hi = -20.0, 80.0
    for _ in range(80):                       # bisection on exit temperature
        mid = 0.5 * (lo + hi)
        if rejected(mid) > Q_W:
            hi = mid
        else:
            lo = mid
    T_e = 0.5 * (lo + hi)
    if T_e < -19.9 or T_e > 79.9:
        raise ValueError("cooling_tower_source: no exhaust temperature between -20 and 80 C "
                         "satisfies the energy balance; check Q_W and inlet_flow")
    q_e = q_from_RH(T_e, RH_exit, P)
    flow = m_da * (1 + q_e / (1 - q_e)) / rho_moist(T_e, q_e, P)
    w0 = geometry.pop("exit_velocity", 8.0)
    return HeatSource(T_exit=T_e, flow=flow, exit_area=flow / w0, RH_exit=RH_exit, **geometry)

# ------------------------------------------------------------------- model
_C_H_STABLE = 0.004      # bulk heat-transfer coefficient, stable surface layer
_V_RAD = 0.001           # m/s, longwave loss of a warm air-layer anomaly (approx.)


def _fan_array(src, st):
    """Number of fans and their mean spacing (m) when fan_heat is given."""
    n = max(1, int(round(abs(st["Q_total"]) / src.fan_heat)))
    return n, float(np.sqrt(src.footprint_x * src.footprint_y / n))


def _downwash(src, st, amb, stab, s, U_s):
    """Building-downwash state after ISC3 (U.S. EPA 1995, Sections 1.1.4.11 and 1.1.5.3),
    or None when the plume is not in the wake.
    hb = building_height, LB = min(hb, crosswind width). The wake check uses the release
    height plus gradual momentum rise at 2 hb (one fan when fan_heat is set); the plume is
    in the wake when that height he <= hb + 1.5 LB (GEP). The Schulman-Scire treatment
    (reduced rise, linear decay factor A) applies when the release height < hb + 0.5 LB;
    otherwise Huber-Snyder (A = 1, rise unchanged)."""
    hb = src.building_height
    if not src.downwash or hb <= 0:
        return None
    LB = min(hb, src.footprint_y)
    n = _fan_array(src, st)[0] if src.fan_heat is not None else src.n_plumes
    r = 0.5 * np.sqrt(4 * src.exit_area / (np.pi * n))                # one fan / plume radius
    Fm = st["w0"] ** 2 * r ** 2 * (amb.T + 273.15) / (src.T_exit + 273.15)
    bj = 1.0 / 3.0 + U_s / max(st["w0"], 1e-6)
    he = src.height + (3.0 * Fm * 2.0 * hb / (bj ** 2 * U_s ** 2)) ** (1 / 3)
    if he > hb + 1.5 * LB:
        return None
    ss = src.height < hb + 0.5 * LB
    A = 1.0
    if ss and he > hb:
        A = max(0.0, (hb - he) / (2.0 * LB) + 1.0)
    return dict(hb=hb, LB=LB, A=A, schulman_scire=ss, he=he, Fm1=Fm, bj=bj)


def _wake_sigma_z(dw, sigz_curve, x):
    """Huber-Snyder wake sigma_z (squat building), times the linear decay factor A:
    0.7 hb + 0.067 (x - 3 hb) for 3 hb <= x < 10 hb (x < 3 hb: the 3 hb value),
    sigma_z(x + xz) beyond, with xz such that sigma_z(10 hb + xz) = 1.2 A hb."""
    hb, A = dw["hb"], dw["A"]
    x = np.asarray(x, float)
    near = A * (0.7 * hb + 0.067 * (np.clip(x, 3 * hb, 10 * hb) - 3 * hb))
    target = 1.2 * A * hb
    lo, hi = 0.0, 1e5                                   # sigma_z curve increases with x
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if sigz_curve(mid) < target else (lo, mid)
    xz = max(0.0, 0.5 * (lo + hi) - 10 * hb)
    far = sigz_curve(np.maximum(x, 10 * hb) + xz)
    return np.where(x < 10 * hb, near, far)


def _ss_rise(dw, F, Fm, bj, U, stab, s, x, x_final, sigz_curve):
    """Schulman-Scire reduced rise of one downwashed plume (ISC3 Eqs. 1-28 to 1-31):
    Z^3 + (3 R0/b) Z^2 + (3 R0^2/b^2) Z = RHS, R0 = sqrt(2) A sigma_z(3 LB), b = 0.6
    (line-source length Ly = 0: for wide buildings the wake sigma_y < sigma_z).
    Buoyant and momentum rise are solved separately and the larger is used."""
    b = 0.6
    R0 = np.sqrt(2.0) * dw["A"] * max(0.7 * dw["hb"], float(sigz_curve(3 * dw["LB"])))
    x = np.asarray(x, float)
    if stab in ("E", "F") and s > 0:
        xs = np.minimum(np.maximum(x, 0), np.pi * U / np.sqrt(s))
        rhs_b = 6.0 * F / (b ** 2 * U * s) * (1 - np.cos(np.sqrt(s) * xs / U))
        rhs_m = 3.0 * Fm * np.sin(np.sqrt(s) * xs / U) / (bj ** 2 * U * np.sqrt(s))
    else:
        xs = np.minimum(np.maximum(x, 0), x_final)
        rhs_b = 3.0 * F * xs ** 2 / (2 * b ** 2 * U ** 3)
        rhs_m = 3.0 * Fm * xs / (bj ** 2 * U ** 2)
    a1, a2 = 3 * R0 / b, 3 * R0 ** 2 / b ** 2

    def solve(rhs):
        rhs = np.maximum(rhs, 0.0)
        Z = np.cbrt(rhs)                                 # Newton from the no-dilution root
        for _ in range(40):
            f = Z ** 3 + a1 * Z ** 2 + a2 * Z - rhs
            Z = np.maximum(Z - f / (3 * Z ** 2 + 2 * a1 * Z + a2), 0.0)
        return Z
    return np.maximum(solve(rhs_b), solve(rhs_m))


def _ss_rise_pair(dw, src, st, amb, stab, s, U_s, d, F):
    """(gradual, final) Schulman-Scire rise of one plume of buoyancy flux F."""
    sig = _SIG_URBAN if amb.urban else _SIG_RURAL
    if F > 0 and not (stab in ("E", "F") and s > 0):
        x_star = 14 * F ** 0.625 if F < 55 else 34 * F ** 0.4
        x_final = 3.5 * x_star
    else:
        x_final = 1e9
    sz = sig[stab][1]
    F = max(F, 0.0)
    dh_x = _ss_rise(dw, F, dw["Fm1"], dw["bj"], U_s, stab, s, d, x_final, sz)
    dh_f = float(_ss_rise(dw, F, dw["Fm1"], dw["bj"], U_s, stab, s, np.array([1e7]), x_final, sz)[0])
    return dh_x, dh_f


def _point_rise(src, st, amb, stab, s, U_s, d):
    """Point-regime rise along distance d: buoyant (Briggs) or momentum, whichever
    gives the larger final rise. Returns (dh(d), dh_final).
    With fan_heat set, the array is n fans on a square grid; the buoyant rise of one fan
    is enhanced by merging along each crosswind row of N = sqrt(n) fans spaced dx apart
    (Briggs 1974, plume rise from multiple sources):
        dh = dh_1 E,  E = ((N + S)/(1 + S))^(1/3),  S = 6 [(N - 1) dx / (N^(1/3) dh_1)]^(3/2)."""
    dw = _downwash(src, st, amb, stab, s, U_s)
    ss = dw is not None and dw["schulman_scire"]
    if src.fan_heat is not None:
        n, dx = _fan_array(src, st)
        dh_x, dh_f = briggs_rise(st["F_b"] / n, U_s, stab, s, d)
        if ss:
            dh_x, dh_f = _ss_rise_pair(dw, src, st, amb, stab, s, U_s, d, st["F_b"] / n)
        if dh_f > 0 and n > 1:
            N = max(1.0, np.sqrt(n))
            S = 6.0 * ((N - 1.0) * dx / (N ** (1 / 3) * dh_f)) ** 1.5
            E = ((N + S) / (1.0 + S)) ** (1 / 3)
            dh_x, dh_f = dh_x * E, dh_f * E
        d_plume = np.sqrt(4 * src.exit_area / (np.pi * n))
        dh_m = momentum_rise(st["w0"], d_plume, src.T_exit + 273.15, amb.T + 273.15, U_s, stab, s)
        if dh_m > dh_f and not ss:                  # SS rise already includes momentum
            return np.full_like(np.asarray(d, float), dh_m), dh_m
        return dh_x, dh_f
    dh_x, dh_f = briggs_rise(st["F_b"] / src.n_plumes, U_s, stab, s, d)
    if ss:
        return _ss_rise_pair(dw, src, st, amb, stab, s, U_s, d, st["F_b"] / src.n_plumes)
    d_plume = np.sqrt(4 * src.exit_area / (np.pi * src.n_plumes))     # one plume
    dh_m = momentum_rise(st["w0"], d_plume, src.T_exit + 273.15, amb.T + 273.15,
                         U_s, stab, s)
    if dh_m > dh_f:          # momentum-dominated: jet rise is reached within tens of metres
        return np.full_like(np.asarray(d, float), dh_m), dh_m
    return dh_x, dh_f


def _lateral_profile(yy, sy, W):
    """Uniform crosswind strip of width W convolved with a Gaussian of std sy (1/m).
    Integrates to 1 over y."""
    a = np.sqrt(2.0) * sy
    return (_erf((yy + W / 2) / a) - _erf((yy - W / 2) / a)) / (2 * W)


def _component(kind, src, amb, st, stab, s, U_s, h_conv, x, y, z, mode, lateral,
               plume_rise, n_images, x_release=None):
    """Ground-level field for one source treatment.
    kind = 'area': heat mixed through h_mix, released as a crosswind strip at
                   x_release (default: the DOWNWIND EDGE of the footprint);
                   distances are measured from that strip.
    kind = 'point': plume released at the footprint centre with plume rise."""
    zi = amb.mixing_height
    sig = _SIG_URBAN if amb.urban else _SIG_RURAL
    W = src.footprint_y
    r = np.abs(x) if mode == "radial" else x

    # ---- release geometry, as a function of distance d from the release line
    dw = None
    if kind == "area":
        d = r - (src.footprint_x / 2 if x_release is None else x_release)
        h_mix = max(h_conv, src.height)
        rise = lambda dd: (np.zeros_like(np.asarray(dd, float)), 0.0)
        sz0 = max(src.building_height, h_mix) / 2.15
        release = lambda dh: np.minimum(h_mix / 2.0 + 0 * dh, 0.98 * zi)
        P_pen = 0.0
    else:
        d = r
        h_mix = 0.0
        if plume_rise:
            rise = lambda dd: _point_rise(src, st, amb, stab, s, U_s, dd)
        else:
            rise = lambda dd: (np.zeros_like(np.asarray(dd, float)), 0.0)
        sz0 = src.building_height / 2.15
        dw = _downwash(src, st, amb, stab, s, U_s)
        if dw is not None:
            sz0 = 0.0                                   # wake spread handled by Huber-Snyder
        release = lambda dh: np.minimum(src.height + dh, 0.98 * zi)
        _, dh_f0 = rise(np.array([1.0]))
        if src.height >= zi:
            P_pen = 1.0
        elif dh_f0 > 0:     # heuristic adapted from AERMOD's penetration factor
            P_pen = float(np.clip(1.5 - (zi - src.height) / dh_f0, 0.0, 1.0))
        else:
            P_pen = 0.0

    def spreads(dd):
        """sigma_y, sigma_z, release height and dilution wind at distance dd."""
        ds = np.maximum(dd, 1.0)
        dh, dh_f = rise(ds)
        h_rel = release(dh)
        U = wind_at(amb.U10, h_rel, stab, amb.urban)
        if lateral == "sigma_theta":
            th = np.radians(amb.sigma_theta if amb.sigma_theta is not None
                            else _SIGMA_THETA[stab])
            sy_t = th * ds / (1.0 + 0.9 * np.sqrt(ds / U / 1000.0))   # Draxler (1976)
        else:
            sy_t = sig[stab][0](ds)
        sz_t = sig[stab][1](ds)
        if dw is not None:                              # ISC3 building-wake sigmas
            sz_t = np.maximum(sz_t, _wake_sigma_z(dw, sig[stab][1], ds))
            hb = dw["hb"]
            sy_t = np.maximum(sy_t, 0.35 * hb + 0.067 * (np.clip(ds, 3 * hb, None) - 3 * hb))
        sy = np.sqrt(sy_t ** 2 + (dh / 3.5) ** 2)
        sz = np.sqrt(sz_t ** 2 + sz0 ** 2 + (dh / 3.5) ** 2)
        return sy, sz, h_rel, U, dh, dh_f

    sy, sz, h_rel, U, dh_x, dh_f = spreads(d)

    def vertical(zz, h, sz_):
        """Image sum for reflections at the ground and the mixing height."""
        V = np.zeros_like(sz_)
        for n in range(-n_images, n_images + 1):
            V += np.exp(-(zz - h + 2 * n * zi) ** 2 / (2 * sz_ ** 2))
            V += np.exp(-(zz + h + 2 * n * zi) ** 2 / (2 * sz_ ** 2))
        return np.where(sz_ > 1.6 * zi, np.sqrt(2 * np.pi) * sz_ / zi, V)

    vert = vertical(z, h_rel, sz) / (np.sqrt(2 * np.pi) * U * sz)   # crosswind-integrated, s/m2

    if mode == "centerline":
        kernel = _lateral_profile(y, sy, W) * vert
    else:  # radial: uniform wind-direction average, y ignored
        kernel = np.minimum(vert / (2 * np.pi * np.maximum(r, 1.0)),
                            _lateral_profile(0.0, sy, W) * vert)

    # ---- heat loss to the surface and by longwave radiation (source depletion):
    #      d(heat flux)/dx = -v_h * (ground-level excess / column excess), with the
    #      ground/column ratio taken from the SAME vertical kernel as the dispersion:
    #      F_loss = exp[-v_h * int V(0; h, sz) / (sqrt(2 pi) sz U) dx]
    v_h = amb.heat_loss_velocity
    if v_h is None:
        v_h = _C_H_STABLE * amb.U10 + _V_RAD if stab in ("E", "F") else 0.0
    loss = np.ones(np.shape(d))
    d_max = float(np.nanmax(d)) if np.any(np.isfinite(d)) else 0.0
    if v_h > 0 and d_max > 1.0:
        g = np.linspace(1.0, d_max, 2000)
        _, gsz, gh, gU, _, _ = spreads(g)
        integ = vertical(0.0, gh, gsz) / (np.sqrt(2 * np.pi) * gsz * gU)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(g))])
        loss = np.exp(-v_h * np.interp(np.maximum(d, 1.0), g, cum))

    dT = (1 - P_pen) * st["Q_s"] / (st["rho_a"] * st["cp_a"]) * kernel * loss
    dq = (1 - P_pen) * st["E"] / st["rho_a"] * kernel
    released = d > 0 if kind == "area" else np.ones_like(d, dtype=bool)
    if mode == "centerline":
        released = released & (x > -src.footprint_x / 2)   # nothing upwind of the source
    dT, dq = np.where(released, dT, 0.0), np.where(released, dq, 0.0)
    return dict(dT=dT, dq=dq, sigma_y=sy, sigma_z=sz, h_eff=h_rel, U_dilution=U,
                final_rise=dh_f, penetration_fraction=P_pen, mixing_depth=h_mix,
                heat_loss_factor=loss, heat_loss_velocity=v_h)


def plume(src: HeatSource, amb: Ambient, x, y=0.0, z=2.0,
          mode="centerline", plume_rise=True, regime="auto", lateral="sigma_theta",
          blend_n=4, alongwind_strips=1, n_images=5) -> dict:
    """
    Steady-state (1-hour average) ground-level distribution of dT and dq.

    x : downwind distance from the CENTRE of the heat-rejection area, m
    y : crosswind offset, m. x and y are broadcast together, so
        x[:, None], y[None, :] gives a 2-D map of shape (len(x), len(y))
    z : receptor height, m (2 m = weather-station height)
    mode : 'centerline' -> field for one mean wind direction (x, y)
           'radial'     -> average over uniformly distributed wind directions,
                           as a function of distance |x| (y is ignored)
    plume_rise : False -> point release at fan height with no rise.
                 Diagnostic only: physically inconsistent for strong heat sources.
    regime : 'auto'  -> smooth blend of the area and point treatments, weighted by
                        the source width relative to the depth the heat would reach
                        (w = R^4 / (1 + R^4), R = width / max(h_conv, point rise))
             'area'  -> heat mixes its own convective layer over the footprint
             'point' -> Briggs plume rise from a point source
    lateral : 'sigma_theta' -> sigma_y from the 1-hour wind-direction fluctuation
                               (includes meander; default)
              'pg'          -> Pasquill-Gifford/Briggs sigma_y (10-min snapshot)
    blend_n : exponent of the area/point blend weight (default 4)
    alongwind_strips : 1 (default) releases the area treatment at the downwind
              edge, the upper near-field estimate. n = 2..5 spreads the release
              over the footprint length with n-point Gauss-Legendre strips
              (ISC-style area integration), the lower near-field estimate.
    """
    stab = amb.stability.upper()
    if stab not in _SIG_RURAL:
        raise ValueError("stability must be one of 'A'..'F'")
    if not (0 <= amb.RH <= 1) or (src.RH_exit is not None and not (0 <= src.RH_exit <= 1)):
        raise ValueError("RH inputs must be fractions between 0 and 1")
    if src.flow <= 0 or src.exit_area <= 0 or amb.mixing_height <= 0 or src.n_plumes < 1:
        raise ValueError("flow, exit_area, mixing_height must be > 0 and n_plumes >= 1")
    if src.fan_heat is not None and not src.fan_heat > 0:
        raise ValueError("fan_heat must be > 0 (W per fan) or None")
    if mode not in ("centerline", "radial"):
        raise ValueError("mode must be 'centerline' or 'radial'")
    if lateral not in ("sigma_theta", "pg"):
        raise ValueError("lateral must be 'sigma_theta' or 'pg'")
    if regime not in ("auto", "area", "point"):
        raise ValueError("regime must be 'auto', 'area' or 'point'")
    if int(alongwind_strips) != alongwind_strips or not 1 <= alongwind_strips <= 5:
        raise ValueError("alongwind_strips must be an integer from 1 to 5")
    if mode == "radial" and np.any(np.asarray(y) != 0):
        warnings.warn("mode='radial' averages over wind directions; the y input is ignored.")

    st = source_terms(src, amb)
    if st["mixing_line_max_RH"] > 1.0:
        warnings.warn("Exhaust/ambient mixing line exceeds 100% RH: a condensing (visible) "
                      "plume is likely; dry-plume rise and passive-vapour assumptions do "
                      "not hold near the source.")
    if mode == "centerline":
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    else:
        x, y = np.asarray(x, float), 0.0
    if amb.U10 < 1.0:
        warnings.warn("U10 < 1 m/s: Gaussian plume theory is unreliable; wind clipped to 1 m/s.")

    U_s = float(wind_at(amb.U10, src.height, stab, amb.urban))
    Ta_K = amb.T + 273.15
    dtheta = amb.dtheta_dz if amb.dtheta_dz is not None else _DTHETA.get(stab, 0.0)
    s = G / Ta_K * dtheta                                   # stability parameter, 1/s2
    if stab in ("E", "F") and s <= 0:
        warnings.warn("Stable class with dtheta_dz <= 0: neutral plume-rise formula used.")

    # depth of the convective layer the source builds over its footprint
    B_k = (st["Q_s"] / (st["rho_a"] * st["cp_a"])
           + 0.608 * Ta_K * st["E"] / st["rho_a"]) / (src.footprint_x * src.footprint_y)
    t_res = src.footprint_x / U_s
    h_conv = convective_mixing_depth(B_k, t_res, s * Ta_K / G, amb.mixing_height, Ta_K)

    # weight of the area treatment
    if not plume_rise or B_k <= 0 or regime == "point":
        w = 0.0
    elif regime == "area":
        w = 1.0
    else:
        _, dh_pt = _point_rise(src, st, amb, stab, s, U_s, np.array([1.0]))
        R = min(src.footprint_x, src.footprint_y) / max(h_conv, src.height, dh_pt)
        w = float(R ** blend_n / (1.0 + R ** blend_n))

    args = (src, amb, st, stab, s, U_s, h_conv, x, y, z, mode, lateral, plume_rise, n_images)
    comp = {}
    if w > 0:
        if alongwind_strips == 1:
            comp["area"] = _component("area", *args)
        else:                         # Gauss-Legendre strips over the footprint length
            nodes, wts = np.polynomial.legendre.leggauss(int(alongwind_strips))
            parts = [_component("area", *args, x_release=nd * src.footprint_x / 2)
                     for nd in nodes]
            # field = heat-weighted sum of strips; diagnostics (sigma, loss, ...) are
            # those of the edge-release reference geometry
            comp["area"] = dict(_component("area", *args))
            for key in ("dT", "dq"):
                comp["area"][key] = sum(0.5 * wt * p[key] for wt, p in zip(wts, parts))
    if w < 1:
        comp["point"] = _component("point", *args)
    zero = np.zeros(np.shape(x))
    dT = w * comp.get("area", {}).get("dT", zero) + (1 - w) * comp.get("point", {}).get("dT", zero)
    dq = w * comp.get("area", {}).get("dq", zero) + (1 - w) * comp.get("point", {}).get("dq", zero)

    r = np.abs(x) if mode == "radial" else x
    inside = (np.abs(r) < src.footprint_x / 2)
    if mode == "centerline":
        inside = inside & (np.abs(y) < src.footprint_y / 2)
    bad = ~np.isfinite(x) | (~np.isfinite(y) if mode == "centerline" else False)
    dT = np.where(inside | bad, np.nan, dT)
    dq = np.where(inside | bad, np.nan, dq)
    T = amb.T + dT
    q = st["q_a"] + dq
    RH = RH_from_q(T, q, st["P"])
    if np.nanmax(RH) > 1.0:
        warnings.warn("Receptor RH > 100% somewhere: condensation/fog likely; model invalid there.")

    main = comp["area"] if w >= 0.5 else comp["point"]
    return dict(x=x, dT=dT, T=T, dq=dq, q=q, RH=RH,
                regime="area" if w >= 0.99 else ("point" if w <= 0.01 else "blend"),
                area_weight=w, mixing_depth=max(h_conv, src.height) if w > 0 else 0.0,
                buoyancy_flux_per_area=B_k,
                final_rise=comp["point"]["final_rise"] if "point" in comp else 0.0,
                penetration_fraction=comp["point"]["penetration_fraction"] if "point" in comp else 0.0,
                h_eff=main["h_eff"], sigma_y=main["sigma_y"], sigma_z=main["sigma_z"],
                U_rise=U_s, U_dilution=main["U_dilution"], stability_param=s,
                heat_loss_factor=main["heat_loss_factor"],
                heat_loss_velocity=main["heat_loss_velocity"],
                components={k: dict(dT=v["dT"], dq=v["dq"]) for k, v in comp.items()},
                source=st)


# ------------------------------------------------------------------- example
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Phoenix summer night, rural edge, strongly stable
    night = Ambient(T=32.0, RH=0.20, U10=2.0, stability="F", mixing_height=150.0, elevation=340.0)

    # Same 100 MW IT campus, two cooling technologies (total heat rejected differs
    # because the air-cooled plant uses more electricity)
    dry = dry_cooler_source(137e6, 12.0, night, height=12.0, footprint_x=600.0,
                            footprint_y=600.0, building_height=10.0)
    tower = cooling_tower_source(125e6, 1650.0, night, height=15.0, footprint_x=300.0,
                                 footprint_y=200.0, building_height=10.0)

    x = np.linspace(50, 5000, 400)
    cases = {
        "Air-cooled (dry coolers)": plume(dry, night, x),
        "Air-cooled, along-wind strips": plume(dry, night, x, alongwind_strips=2),
        "Water-cooled (cooling towers)": plume(tower, night, x),
        "Air-cooled, merged point plume (lower bound)": plume(dry, night, x, regime="point"),
        "Air-cooled, no rise (diagnostic only)": plume(dry, night, x, plume_rise=False),
    }
    for k, r in cases.items():
        st = r["source"]
        print(
              f"{k:46s} w={r['area_weight']:.3f} Q_s={st['Q_s']/1e6:6.1f} MW "
              f"Q_l={st['Q_l']/1e6:6.1f} MW h_mix={r['mixing_depth']:5.1f} m "
              f"dT@1km={np.interp(1000, x, r['dT']):.3f} K dT@5km={np.interp(5000, x, r['dT']):.3f} K "
              f"dRH@1km={100*(np.interp(1000, x, r['RH'])-night.RH):.1f} pp")
    print(f"exhaust: dry cooler {dry.T_exit:.1f} C, cooling tower {tower.T_exit:.1f} C at "
          f"{100*tower.RH_exit:.0f}% RH")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a, t = cases["Air-cooled (dry coolers)"], cases["Water-cooled (cooling towers)"]
    s = cases["Air-cooled, along-wind strips"]
    ax[0].fill_between(x / 1000, s["dT"], a["dT"], color="C3", alpha=0.25,
                       label="Air-cooled, near-field range")
    ax[0].plot(x / 1000, a["dT"], "C3", lw=2, label="Air-cooled (dry coolers)")
    ax[0].plot(x / 1000, t["dT"], "C0", lw=2, label="Water-cooled (cooling towers)")
    ax[0].set(xlabel="Downwind distance from campus center (km)", ylabel="ΔT at 2 m (K)",
              title="Temperature rise, Phoenix summer night")
    ax[0].set_ylim(bottom=0); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(x / 1000, 100 * (a["RH"] - night.RH), "C3", lw=2, label="Air-cooled (dry coolers)")
    ax[1].plot(x / 1000, 100 * (t["RH"] - night.RH), "C0", lw=2, label="Water-cooled (cooling towers)")
    ax[1].set(xlabel="Downwind distance from campus center (km)",
              ylabel="ΔRH at 2 m (percentage points)", title="Humidity change")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "dc_plume_example.png"), dpi=150)
