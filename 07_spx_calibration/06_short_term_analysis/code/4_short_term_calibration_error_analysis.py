# -*- coding: utf-8 -*-
"""
06 - Short-term analysis
4 - Short-term calibration error analysis - Rough Heston vs classical Heston

This script does not recalibrate either model.

Il lit les résultats détaillés définitifs de 05 - Calibration court terme,
vérifie que Rough Heston et Heston classique ont été évalués sur exactement
les mêmes contrats, puis analyse où se concentrent leurs erreurs.

Analyses :
1. Erreur absolue moyenne (MAE) et RMSE par horizon calendaire.
2. Erreur absolue par zone de moneyness forward K/F0(T).
3. Biais moyen de volatilité implicite par horizon calendaire.
4. Gain relatif de RMSE de Rough Heston par rapport à Heston classique.

Convention :
    Résidu = IV_modèle - IV_marché

Donc :
- biais > 0 : le modèle surestime en moyenne la volatilité implicite ;
- biais < 0 : le modèle sous-estime en moyenne la volatilité implicite.

Pour le gain relatif :
    Gain_RMSE_RH(%) = 100 * (RMSE_Heston - RMSE_RH) / RMSE_Heston

Donc :
- gain > 0 : Rough Heston est meilleur ;
- gain < 0 : Heston classique est meilleur.

Sorties CSV :
- CSV/4 - Erreurs court terme par maturité.csv
- CSV/4 - Erreurs court terme par zone de moneyness forward.csv

Sorties FIGURES :
- FIGURES/4 - Erreur absolue court terme par maturité - Rough Heston vs Heston classique.png
- FIGURES/4 - Erreur absolue court terme par moneyness forward - Rough Heston vs Heston classique.png
- FIGURES/4 - Biais court terme par maturité - Rough Heston vs Heston classique.png
- FIGURES/4 - Gain RMSE court terme de Rough Heston par maturité.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========
# 1. PATHS
# ===========

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

CALIB_CSV_DIR = (
    ROOT_DIR / "07_spx_calibration" / "05_short_term_calibration" / "results"
)
ANALYSIS_DIR = ROOT_DIR / "07_spx_calibration" / "06_short_term_analysis"
CSV_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RH_FILE = CALIB_CSV_DIR / "01_rough_heston_short_term_calibration_results.csv"
HESTON_FILE = CALIB_CSV_DIR / "03_heston_short_term_calibration_results.csv"

OUT_MATURITY = CSV_DIR / "05_short_term_calibration_errors_by_maturity.csv"
OUT_MONEYNESS = CSV_DIR / "06_short_term_calibration_errors_by_forward_moneyness.csv"
FIG_MATURITY = FIGURES_DIR / "05_short_term_absolute_error_by_maturity.png"
FIG_MONEYNESS = FIGURES_DIR / "06_short_term_absolute_error_by_forward_moneyness.png"
FIG_BIAS = FIGURES_DIR / "07_short_term_bias_by_maturity.png"
FIG_GAIN = FIGURES_DIR / "08_short_term_rough_heston_rmse_gain_by_maturity.png"

# ===============================
# 2. FORWARD-MONEYNESS REGIONS
# ===============================

# The bounds cover the final 0.70 <= K/F0 <= 1.30 filter.
MONEYNESS_BINS = [
    0.70,
    0.85,
    0.95,
    1.05,
    1.15,
    1.30,
]

MONEYNESS_LABELS = [
    "0.70–0.85",
    "0.85–0.95",
    "0.95–1.05",
    "1.05–1.15",
    "1.15–1.30",
]


# =============
# 3. LOADING
# =============

if not RH_FILE.exists():
    raise FileNotFoundError(
        f"Fichier Rough Heston introuvable : {RH_FILE}"
    )

if not HESTON_FILE.exists():
    raise FileNotFoundError(
        f"Fichier Heston classique introuvable : {HESTON_FILE}"
    )

rh = pd.read_csv(RH_FILE)
he = pd.read_csv(HESTON_FILE)


# ===================
# 4. COLUMN CHECKS
# ===================

required_rh = [
    "Contract",
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "Option_type",
    "IV_market",
    "IV_RH",
]

required_h = [
    "Contract",
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "Option_type",
    "IV_market",
    "IV_Heston",
]

missing_rh = [c for c in required_rh if c not in rh.columns]
missing_h = [c for c in required_h if c not in he.columns]

if missing_rh:
    raise ValueError(
        "Colonnes absentes du fichier Rough Heston : "
        + ", ".join(missing_rh)
    )

if missing_h:
    raise ValueError(
        "Colonnes absentes du fichier Heston classique : "
        + ", ".join(missing_h)
    )

if rh["Contract"].duplicated().any():
    raise ValueError(
        "Le fichier Rough Heston contient des contrats dupliqués."
    )

if he["Contract"].duplicated().any():
    raise ValueError(
        "Le fichier Heston classique contient des contrats dupliqués."
    )


# =================================
# 5. MERGE ON THE SAME CONTRACTS
# =================================

rh_keep = rh[
    [
        "Contract",
        "T_days",
        "T",
        "Strike",
        "K_over_F0",
        "Option_type",
        "IV_market",
        "IV_RH",
    ]
].copy()

he_keep = he[
    [
        "Contract",
        "T_days",
        "T",
        "Strike",
        "K_over_F0",
        "Option_type",
        "IV_market",
        "IV_Heston",
    ]
].copy()

merged = rh_keep.merge(
    he_keep,
    on="Contract",
    how="outer",
    suffixes=("_RH_file", "_H_file"),
    indicator=True,
)

if not (merged["_merge"] == "both").all():
    unmatched = merged.loc[
        merged["_merge"] != "both",
        ["Contract", "_merge"],
    ]

    raise ValueError(
        "Les deux calibrations ne portent pas exactement "
        "sur les mêmes contrats.\n"
        + unmatched.to_string(index=False)
    )

merged = merged.drop(columns="_merge")


# =============================
# 6. COMMON-DATA CONSISTENCY
# =============================

for col in [
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "IV_market",
]:
    a = merged[f"{col}_RH_file"].to_numpy(dtype=float)
    b = merged[f"{col}_H_file"].to_numpy(dtype=float)

    if not np.allclose(
        a,
        b,
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError(
            f"Incohérence entre les deux fichiers pour {col}."
        )

a_type = merged["Option_type_RH_file"].astype(str).to_numpy()
b_type = merged["Option_type_H_file"].astype(str).to_numpy()

if not np.array_equal(a_type, b_type):
    raise ValueError(
        "Incohérence entre les deux fichiers pour Option_type."
    )


# ==================================
# 7. COMMON DATASET AND RESIDUALS
# ==================================

df = pd.DataFrame({
    "Contract": merged["Contract"],
    "T_days": merged["T_days_RH_file"],
    "T": merged["T_RH_file"],
    "Strike": merged["Strike_RH_file"],
    "K_over_F0": merged["K_over_F0_RH_file"],
    "Option_type": merged["Option_type_RH_file"],
    "IV_market": merged["IV_market_RH_file"],
    "IV_RH": merged["IV_RH"],
    "IV_Heston": merged["IV_Heston"],
})

valid = (
    np.isfinite(df["T_days"])
    & np.isfinite(df["T"])
    & np.isfinite(df["Strike"])
    & np.isfinite(df["K_over_F0"])
    & np.isfinite(df["IV_market"])
    & np.isfinite(df["IV_RH"])
    & np.isfinite(df["IV_Heston"])
    & (df["T"] > 0)
    & (df["Strike"] > 0)
    & (df["K_over_F0"] > 0)
    & (df["IV_market"] > 0)
    & (df["IV_RH"] > 0)
    & (df["IV_Heston"] > 0)
)

df = df.loc[valid].copy().reset_index(drop=True)

df["Calendar_days"] = df["T_days"].round().astype(int)

EXPECTED_DAYS = [1, 3, 7, 15, 31]
actual_days = sorted(df["Calendar_days"].unique().tolist())

if actual_days != EXPECTED_DAYS:
    raise ValueError(
        f"Horizons court terme inattendus : {actual_days}. "
        f"Attendu : {EXPECTED_DAYS}"
    )

# Common convention throughout the analysis:
# residual = model - market.
df["Residual_RH"] = df["IV_RH"] - df["IV_market"]
df["Residual_Heston"] = df["IV_Heston"] - df["IV_market"]

df["Abs_error_RH"] = np.abs(df["Residual_RH"])
df["Abs_error_Heston"] = np.abs(df["Residual_Heston"])

df["Squared_error_RH"] = df["Residual_RH"] ** 2
df["Squared_error_Heston"] = df["Residual_Heston"] ** 2


print()
print("=" * 80)
print("SHORT-TERM CALIBRATION ERROR ANALYSIS")
print("ROUGH HESTON vs CLASSICAL HESTON")
print("=" * 80)
print("Observations communes :", len(df))
print("Horizons calendaires :", df["Calendar_days"].nunique())
print("Maturités effectives :", df["T"].nunique())


# ====================
# 8. GLOBAL METRICS
# ====================

def compute_metrics(group):
    residual_rh = group["Residual_RH"].to_numpy(dtype=float)
    residual_h = group["Residual_Heston"].to_numpy(dtype=float)

    rmse_rh = float(np.sqrt(np.mean(residual_rh ** 2)))
    rmse_h = float(np.sqrt(np.mean(residual_h ** 2)))

    mae_rh = float(np.mean(np.abs(residual_rh)))
    mae_h = float(np.mean(np.abs(residual_h)))

    bias_rh = float(np.mean(residual_rh))
    bias_h = float(np.mean(residual_h))

    max_rh = float(np.max(np.abs(residual_rh)))
    max_h = float(np.max(np.abs(residual_h)))

    if rmse_h > 0:
        gain_rmse = 100.0 * (rmse_h - rmse_rh) / rmse_h
    else:
        gain_rmse = np.nan

    if mae_h > 0:
        gain_mae = 100.0 * (mae_h - mae_rh) / mae_h
    else:
        gain_mae = np.nan

    return {
        "N": int(len(group)),
        "RMSE_RH": rmse_rh,
        "RMSE_Heston": rmse_h,
        "MAE_RH": mae_rh,
        "MAE_Heston": mae_h,
        "Bias_RH": bias_rh,
        "Bias_Heston": bias_h,
        "Max_error_RH": max_rh,
        "Max_error_Heston": max_h,
        "Gain_RMSE_RH_pct": gain_rmse,
        "Gain_MAE_RH_pct": gain_mae,
    }


global_metrics = compute_metrics(df)

print()
print("GLOBAL METRICS")
print("-" * 80)

for key, value in global_metrics.items():
    if key == "N":
        print(f"{key:<22s}: {value}")
    else:
        print(f"{key:<22s}: {value:.8f}")


# ================================
# 9. ERRORS BY CALENDAR HORIZON
# ================================

maturity_rows = []

for days, group in df.groupby("Calendar_days", sort=True):
    metrics = compute_metrics(group)

    maturity_rows.append({
        "Calendar_days": int(days),
        "N_effective_maturities": int(group["T"].nunique()),
        **metrics,
    })

errors_maturity = pd.DataFrame(maturity_rows)

errors_maturity["Winner_RMSE"] = np.where(
    errors_maturity["RMSE_RH"] < errors_maturity["RMSE_Heston"],
    "Rough Heston",
    np.where(
        errors_maturity["RMSE_RH"] > errors_maturity["RMSE_Heston"],
        "Heston classique",
        "Égalité",
    ),
)

print()
print("=" * 80)
print("ERRORS BY CALENDAR HORIZON")
print("=" * 80)

print(
    errors_maturity[
        [
            "Calendar_days",
            "N",
            "RMSE_RH",
            "RMSE_Heston",
            "MAE_RH",
            "MAE_Heston",
            "Bias_RH",
            "Bias_Heston",
            "Gain_RMSE_RH_pct",
            "Winner_RMSE",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# =========================================
# 10. ERRORS BY FORWARD-MONEYNESS REGION
# =========================================

df["Moneyness_zone"] = pd.cut(
    df["K_over_F0"],
    bins=MONEYNESS_BINS,
    labels=MONEYNESS_LABELS,
    include_lowest=True,
    right=True,
    ordered=True,
)

if df["Moneyness_zone"].isna().any():
    outside = df.loc[
        df["Moneyness_zone"].isna(),
        "K_over_F0",
    ]

    raise ValueError(
        "Certaines observations sont hors des zones de moneyness "
        f"définies. Min={outside.min():.6f}, "
        f"Max={outside.max():.6f}"
    )

moneyness_rows = []

for zone in MONEYNESS_LABELS:
    group = df[
        df["Moneyness_zone"].astype(str) == zone
    ].copy()

    if group.empty:
        continue

    metrics = compute_metrics(group)

    moneyness_rows.append({
        "Moneyness_zone": zone,
        "K_over_F0_min_observed": float(group["K_over_F0"].min()),
        "K_over_F0_max_observed": float(group["K_over_F0"].max()),
        **metrics,
    })

errors_moneyness = pd.DataFrame(moneyness_rows)

errors_moneyness["Winner_RMSE"] = np.where(
    errors_moneyness["RMSE_RH"] < errors_moneyness["RMSE_Heston"],
    "Rough Heston",
    np.where(
        errors_moneyness["RMSE_RH"] > errors_moneyness["RMSE_Heston"],
        "Heston classique",
        "Égalité",
    ),
)

print()
print("=" * 80)
print("ERRORS BY FORWARD-MONEYNESS REGION")
print("=" * 80)

print(
    errors_moneyness[
        [
            "Moneyness_zone",
            "N",
            "RMSE_RH",
            "RMSE_Heston",
            "MAE_RH",
            "MAE_Heston",
            "Bias_RH",
            "Bias_Heston",
            "Gain_RMSE_RH_pct",
            "Winner_RMSE",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================
# 11. FIGURE 1 - ABSOLUTE ERROR BY MATURITY
# ============================================

x = errors_maturity["Calendar_days"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(
    x,
    errors_maturity["MAE_RH"],
    marker="o",
    linewidth=1.9,
    label="Rough Heston",
)

ax.plot(
    x,
    errors_maturity["MAE_Heston"],
    marker="o",
    linestyle="--",
    linewidth=1.8,
    label="Classical Heston",
)

ax.set_xlabel("Calendar horizon (days)")
ax.set_ylabel("Mean absolute implied-volatility error (MAE)")
ax.set_title(
    "Mean absolute error by maturity: "
    "Rough Heston vs Heston classique"
)

ax.grid(True, alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_MATURITY,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================
# 12. FIGURE 2 - ABSOLUTE ERROR BY FORWARD MONEYNESS
# =====================================================

positions = np.arange(len(errors_moneyness))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.bar(
    positions - width / 2,
    errors_moneyness["MAE_RH"],
    width=width,
    label="Rough Heston",
)

ax.bar(
    positions + width / 2,
    errors_moneyness["MAE_Heston"],
    width=width,
    label="Classical Heston",
)

ax.set_xticks(positions)
ax.set_xticklabels(
    errors_moneyness["Moneyness_zone"],
)

ax.set_xlabel(r"Forward-moneyness region $K/F_0(T)$")
ax.set_ylabel("Mean absolute implied-volatility error (MAE)")
ax.set_title(
    "Mean absolute error by forward moneyness"
)

ax.grid(
    True,
    axis="y",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_MONEYNESS,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ==================================
# 13. FIGURE 3 - BIAS BY MATURITY
# ==================================

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(
    x,
    errors_maturity["Bias_RH"],
    marker="o",
    linewidth=1.9,
    label="Rough Heston",
)

ax.plot(
    x,
    errors_maturity["Bias_Heston"],
    marker="o",
    linestyle="--",
    linewidth=1.8,
    label="Classical Heston",
)

ax.axhline(
    0.0,
    linewidth=1.0,
    linestyle=":",
    alpha=0.8,
)

ax.set_xlabel("Calendar horizon (days)")
ax.set_ylabel(
    r"Biais moyen $\sigma_{\mathrm{modèle}}-\sigma_{\mathrm{marché}}$"
)

ax.set_title(
    "Mean implied-volatility bias by maturity"
)

ax.grid(True, alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG_BIAS,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ========================================
# 14. FIGURE 4 - ROUGH HESTON RMSE GAIN
# ========================================

gain = errors_maturity["Gain_RMSE_RH_pct"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.bar(
    x,
    gain,
    width=np.maximum(1.5, 0.025 * np.maximum(x, 1.0)),
)

ax.axhline(
    0.0,
    linewidth=1.0,
    linestyle="--",
    alpha=0.8,
)

ax.set_xlabel("Calendar horizon (days)")
ax.set_ylabel("Relative RMSE gain of Rough Heston (%)")

ax.set_title(
    "Calibration gain of Rough Heston versus classical Heston"
)

ax.grid(
    True,
    axis="y",
    alpha=0.25,
)

# Annotate cases where classical Heston has the lower RMSE.
for days, value in zip(x, gain):
    if np.isfinite(value) and value < 0:
        ax.annotate(
            f"{int(days)} j",
            xy=(days, value),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
        )

fig.tight_layout()

fig.savefig(
    FIG_GAIN,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ==================
# 15. CSV EXPORTS
# ==================

errors_maturity.to_csv(
    OUT_MATURITY,
    index=False,
)

errors_moneyness.to_csv(
    OUT_MONEYNESS,
    index=False,
)


# ====================
# 16. FINAL SUMMARY
# ====================

n_rh_wins = int(
    (
        errors_maturity["RMSE_RH"]
        < errors_maturity["RMSE_Heston"]
    ).sum()
)

n_h_wins = int(
    (
        errors_maturity["RMSE_Heston"]
        < errors_maturity["RMSE_RH"]
    ).sum()
)

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Rough Heston meilleur en RMSE sur "
    f"{n_rh_wins}/{len(errors_maturity)} horizons."
)

print(
    f"Heston classique meilleur en RMSE sur "
    f"{n_h_wins}/{len(errors_maturity)} horizons."
)

print(
    f"Gain RMSE global de Rough Heston : "
    f"{global_metrics['Gain_RMSE_RH_pct']:.2f} %"
)

print(
    f"Gain MAE global de Rough Heston : "
    f"{global_metrics['Gain_MAE_RH_pct']:.2f} %"
)

print()
print("SAVED FILES")
print("-" * 80)

for path in [
    OUT_MATURITY,
    OUT_MONEYNESS,
    FIG_MATURITY,
    FIG_MONEYNESS,
    FIG_BIAS,
    FIG_GAIN,
]:
    print(" -", path)

print()
print("Done.")
