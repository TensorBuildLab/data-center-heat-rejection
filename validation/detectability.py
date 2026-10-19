"""
detectability.py - predicted signal vs achievable precision for each validation data source.

Scatter per independent sample (night or scene) is back-calculated from the achieved
bootstrap CIs of station_analysis.py and satellite_analysis.py (values entered below); the
campaign scatter is an assumption (0.3-0.5 K per night after site-bias correction).
Minimum detectable effect (MDE) at 95% two-sided, 80% power: 2.80 x SE.
Outputs: results/detectability/{detectability.csv, fig_detectability.png}
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/detectability"
os.makedirs(OUT, exist_ok=True)
Z = 1.96 + 0.84

# ---- scatter per sample, from achieved CIs
# KGYR evening DiD: half-widths 0.75 (KLUF, 34 / 39 nights) and 0.73 (KBXK, 34 / 42)
sig_kgyr = np.mean([0.75 / 1.96 / np.sqrt(1 / 34 + 1 / 39), 0.725 / 1.96 / np.sqrt(1 / 34 + 1 / 42)])
# ECOSTRESS post-commissioning ring contrast: half-widths 0.37 (36 scenes), 0.35 (37)
sig_eco = np.mean([0.37 / 1.96 * np.sqrt(36), 0.35 / 1.96 * np.sqrt(37)])
SIG_CAMP = (0.3, 0.5)          # assumed per-night scatter of downwind-minus-reference
N_BIAS = 200                   # non-downwind stable nights for each sensor's site-bias correction (12-month campaign)

rows = [
    dict(source="KGYR evening record vs KLUF/KBXK (V6)", statistic="Before/after DiD, campus sector",
         predicted_low=0.06, predicted_high=0.16, sigma=sig_kgyr, n_available="34 before / 39-42 after sector evenings",
         halfwidth_achieved=0.74, mde=Z * sig_kgyr * np.sqrt(1 / 34 + 1 / 40),
         n_needed_015=2 * (Z * sig_kgyr / 0.15) ** 2, rules_out="post-commissioning warming > 0.45-0.62 K"),
    dict(source="ECOSTRESS night LST (V5)", statistic="Downwind-upwind contrast after commissioning, 0.5-2 km",
         predicted_low=0.06, predicted_high=0.11, sigma=sig_eco, n_available="36-37 scenes (about 12 usable a year)",
         halfwidth_achieved=0.36, mde=Z * sig_eco / np.sqrt(36),
         n_needed_015=(Z * sig_eco / 0.15) ** 2, rules_out="LST band > 0.15-0.49 K (by ring and wind station)"),
]
for s in SIG_CAMP:
    n = 30
    se = s * np.sqrt(1 / n + 1 / N_BIAS)
    rows.append(dict(source=f"On-site campaign, one axis sensor (scatter {s} K, assumed)",
                     statistic="Downwind minus bias-corrected reference, 1 km",
                     predicted_low=0.16, predicted_high=0.49, sigma=s,
                     n_available="about 30 downwind nights a year (assumed)",
                     halfwidth_achieved=1.96 * se, mde=Z * se,
                     n_needed_015=(Z * s / 0.15) ** 2 / (1 - (Z * s / 0.15) ** 2 / N_BIAS)
                     if (Z * s / 0.15) ** 2 < N_BIAS else np.inf,
                     rules_out="[TBD: campaign]"))
tab = pd.DataFrame(rows)
tab.to_csv(f"{OUT}/detectability.csv", index=False, float_format="%.3f")

# ---- figure: MDE vs number of independent samples
fig, ax = plt.subplots(figsize=(6.6, 4.0))
n = np.logspace(0.7, 3.6, 200)
ax.axhspan(0.06, 0.16, color="C3", alpha=0.15, lw=0, label="Predicted, PHXA1 at 1-1.3 km (4.5-40 MW IT)")
ax.axhspan(0.16, 0.49, color="C1", alpha=0.10, lw=0, label="Predicted, 40-100 MW campus at 1 km, class F")
ax.plot(n, Z * sig_kgyr * np.sqrt(2 / n), color="0.25", lw=1.8, label=f"Airport station DiD (scatter {sig_kgyr:.1f} K)")
ax.plot(n, Z * sig_eco / np.sqrt(n), color="C0", lw=1.8, label=f"ECOSTRESS contrast (scatter {sig_eco:.1f} K)")
for s, ls in zip(SIG_CAMP, ("-", "--")):
    ax.plot(n, Z * s * np.sqrt(1 / n + 1 / N_BIAS), color="C2", lw=1.8, ls=ls,
            label=f"Campaign sensor (scatter {s} K, assumed)")
ax.plot(37, Z * sig_kgyr * np.sqrt(2 / 37), "o", color="0.25", ms=7)
ax.plot(36, Z * sig_eco / np.sqrt(36), "o", color="C0", ms=7)
ax.plot([30, 30], [Z * 0.3 * np.sqrt(1 / 30 + 1 / N_BIAS), Z * 0.5 * np.sqrt(1 / 30 + 1 / N_BIAS)], "o", color="C2", ms=7)
ax.text(0.02, 0.03, "Dots: samples available (campaign: one year, assumed)", transform=ax.transAxes, fontsize=7)
ax.set(xscale="log", yscale="log", xlim=(5, 4000), ylim=(0.03, 3),
       xlabel="Independent samples per period (downwind nights or scenes)",
       ylabel="Minimum detectable effect, 95% / 80% power (K)",
       title="What each data source can detect")
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=6.5, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_detectability.png", dpi=200)
pd.set_option("display.width", 250)
print(tab.round(3).to_string(index=False))
