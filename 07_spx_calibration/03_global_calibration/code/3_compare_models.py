"""
Global comparison of Rough Heston and classical Heston calibrations.

This script compares the global calibration results of the Rough Heston and
classical Heston models on the same SPX option sample. It evaluates global and
maturity-level errors and visualizes the market and model implied volatility
smiles across calendar horizons.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========
# 1. PATHS
# ===========

# Chapter 7 root path
ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

# Path to the global calibration results
CALIB_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "03_global_calibration"
)

CSV_DIR = CALIB_DIR / "results"
FIGURES_DIR = CALIB_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load the Rough Heston calibration results
RH_FILE = (
    CSV_DIR
    / "01_rough_heston_global_calibration_results.csv"
)

# Load the classical Heston calibration results
HESTON_FILE = (
    CSV_DIR
    / "03_heston_global_calibration_results.csv"
)

# Global RH + H comparison summary
OUT_GLOBAL = (
    CSV_DIR
    / "06_global_model_comparison_summary.csv"
)

# RH + H comparison by maturity
OUT_HORIZON = (
    CSV_DIR
    / "07_model_comparison_by_horizon.csv"
)

# RH + H smile comparison figure
OUT_FIG = (
    FIGURES_DIR
    / "07_rough_heston_vs_heston_smiles.png"
)


# ===============================
# 2. LOADING BOTH CALIBRATIONS
# ===============================

# df_rh contains the Rough Heston calibration results
df_rh = pd.read_csv(RH_FILE)

# df_h contains the Heston calibration results
df_h = pd.read_csv(HESTON_FILE)

required_rh = [
    "Contract",
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "Option_type",
    "IV_market",
    "IV_RH"
]

required_h = [
    "Contract",
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "Option_type",
    "IV_market",
    "IV_Heston"
]

missing_rh = [
    col for col in required_rh
    if col not in df_rh.columns
]

missing_h = [
    col for col in required_h
    if col not in df_h.columns
]

if missing_rh:
    raise ValueError(
        f"Missing columns in the Rough Heston file: {missing_rh}"
    )

if missing_h:
    raise ValueError(
        f"Missing columns in the Heston file: {missing_h}"
    )


# ========================================
# 3. CHECK: SAME SAMPLE FOR BOTH MODELS
# ========================================

if df_rh["Contract"].duplicated().any():
    raise ValueError(
        "The Rough Heston file contains duplicated contracts."
    )

if df_h["Contract"].duplicated().any():
    raise ValueError(
        "The Heston file contains duplicated contracts."
    )

rh = df_rh[
    [
        "Contract",
        "T_days",
        "T",
        "Strike",
        "K_over_F0",
        "Option_type",
        "IV_market",
        "IV_RH"
    ]
].copy()

he = df_h[
    [
        "Contract",
        "T_days",
        "T",
        "Strike",
        "K_over_F0",
        "Option_type",
        "IV_market",
        "IV_Heston"
    ]
].copy()

# Create df with all information for each contract
df = rh.merge(
    he,
    on="Contract",
    how="outer",
    suffixes=("_RH_file", "_H_file"),
    indicator=True
)

# Check that both calibrations use the same contracts
if not (df["_merge"] == "both").all():
    only_rh = int((df["_merge"] == "left_only").sum())
    only_h = int((df["_merge"] == "right_only").sum())

    raise ValueError(
        "The two calibrations do not use exactly "
        "the same contracts. "
        f"RH only: {only_rh} | "
        f"Heston only: {only_h}"
    )

# Check the common variables.
numeric_checks = [
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "IV_market"
]

'''Check that the information shared by both files
# is identical: strike, maturity, moneyness, market IV, etc.'''
for col in numeric_checks:
    # Values from the Rough Heston calibration file
    left = df[f"{col}_RH_file"].to_numpy(dtype=float)
    # Values from the Heston calibration file
    right = df[f"{col}_H_file"].to_numpy(dtype=float)

    # Compare values
    if not np.allclose(
        left,
        right,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True # consider two NaN values equivalent
    ):
        raise ValueError(
            f"Inconsistency between both files for column {col}."
        )

# Check that the option type (call or put)
# is also identical in both files.
if not (
    df["Option_type_RH_file"].astype(str).to_numpy()
    == df["Option_type_H_file"].astype(str).to_numpy()
).all():
    raise ValueError(
        "Inconsistency between both files for Option_type."
    )

# Keep a single version of the common variables.
df["T_days"] = df["T_days_RH_file"].astype(int)
df["T"] = df["T_RH_file"].astype(float)
df["Strike"] = df["Strike_RH_file"].astype(float)
df["K_over_F0"] = df["K_over_F0_RH_file"].astype(float)
df["Option_type"] = df["Option_type_RH_file"]
df["IV_market"] = df["IV_market_RH_file"].astype(float)

df = df[
    [
        "Contract",
        "T_days",
        "T",
        "Strike",
        "K_over_F0",
        "Option_type",
        "IV_market",
        "IV_RH",
        "IV_Heston"
    ]
].copy()

# Sort observations by T_days, T, strike and contract
df = (
    df.sort_values(
        ["T_days", "T", "Strike", "Contract"]
    )
    .reset_index(drop=True) # reset the index after sorting
)

print()
print("=" * 72)
print("GLOBAL COMPARISON: ROUGH HESTON vs CLASSICAL HESTON")
print("=" * 72)
print()
print("Number of observations:", len(df))
print("Calendar horizons     :", df["T_days"].nunique())
print("Effective maturities  :", df["T"].nunique())


# =============
# 4. METRICS
# =============

def metrics(market, model):
    # Keep only observations for which
    # market and model implied volatilities are finite.
    valid = (
        np.isfinite(market)
        & np.isfinite(model)
    )
    # Compute the error, i.e. the difference
    errors = (
        model[valid]
        - market[valid]
    )

    # If there are no valid observations, return NaN
    if len(errors) == 0:
        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAX_ERROR": np.nan
        }
    # Otherwise return the errors
    return {
        "N": int(valid.sum()),
        "RMSE": float(
            np.sqrt(np.mean(errors**2))
        ),
        "MAE": float(
            np.mean(np.abs(errors))
        ),
        "MAX_ERROR": float(
            np.max(np.abs(errors))
        )
    }


IV_market = df["IV_market"].to_numpy(dtype=float)
IV_RH = df["IV_RH"].to_numpy(dtype=float)
IV_H = df["IV_Heston"].to_numpy(dtype=float)

# Rough Heston metrics
MET_RH = metrics(
    IV_market,
    IV_RH
)

# Heston metrics
MET_H = metrics(
    IV_market,
    IV_H
)

# Check for inconsistencies
if (
    MET_RH["N"] != len(df)
    or MET_H["N"] != len(df)
):
    raise ValueError(
        "Some model implied volatilities are invalid. "
        f"RH : {MET_RH['N']}/{len(df)} | "
        f"Heston : {MET_H['N']}/{len(df)}"
    )


# =======================
# 5. GLOBAL COMPARISON
# =======================

gain_rmse = (
    MET_H["RMSE"]
    - MET_RH["RMSE"]
)

gain_rmse_pct = (
    100.0
    * gain_rmse
    / MET_H["RMSE"]
)

gain_mae = (
    MET_H["MAE"]
    - MET_RH["MAE"]
)

gain_mae_pct = (
    100.0
    * gain_mae
    / MET_H["MAE"]
)

best_global = (
    "Rough Heston"
    if MET_RH["RMSE"] < MET_H["RMSE"]
    else "Classical Heston"
)

print()
print("=" * 72)
print("GLOBAL RESULTS")
print("=" * 72)
print()

print("ROUGH HESTON")
print(f"  N          = {MET_RH['N']}")
print(f"  RMSE       = {MET_RH['RMSE']:.8f}")
print(f"  MAE        = {MET_RH['MAE']:.8f}")
print(f"  Max error = {MET_RH['MAX_ERROR']:.8f}")

print()
print("CLASSICAL HESTON")
print(f"  N          = {MET_H['N']}")
print(f"  RMSE       = {MET_H['RMSE']:.8f}")
print(f"  MAE        = {MET_H['MAE']:.8f}")
print(f"  Max error = {MET_H['MAX_ERROR']:.8f}")

print()
print("COMPARISON")
print(f"  Absolute RH RMSE gain = {gain_rmse:.8f}")
print(f"  Relative RH RMSE gain = {gain_rmse_pct:.2f} %")
print(f"  Relative RH MAE gain  = {gain_mae_pct:.2f} %")
print(f"  Best global model  = {best_global}")


# ===========================
# 6. GLOBAL SUMMARY EXPORT
# ===========================

global_summary = pd.DataFrame([
    {
        "Modele": "Rough Heston",
        "N": MET_RH["N"],
        "RMSE": MET_RH["RMSE"],
        "MAE": MET_RH["MAE"],
        "MAX_ERROR": MET_RH["MAX_ERROR"],
        "Gain_RMSE_RH_vs_Heston": gain_rmse,
        "Gain_RMSE_RH_vs_Heston_pct": gain_rmse_pct,
        "Gain_MAE_RH_vs_Heston": gain_mae,
        "Gain_MAE_RH_vs_Heston_pct": gain_mae_pct,
        "Meilleur_modele_global": best_global
    },
    {
        "Modele": "Classical Heston",
        "N": MET_H["N"],
        "RMSE": MET_H["RMSE"],
        "MAE": MET_H["MAE"],
        "MAX_ERROR": MET_H["MAX_ERROR"],
        "Gain_RMSE_RH_vs_Heston": gain_rmse,
        "Gain_RMSE_RH_vs_Heston_pct": gain_rmse_pct,
        "Gain_MAE_RH_vs_Heston": gain_mae,
        "Gain_MAE_RH_vs_Heston_pct": gain_mae_pct,
        "Meilleur_modele_global": best_global
    }
])

global_summary.to_csv(
    OUT_GLOBAL,
    index=False,
    encoding="utf-8-sig"
)


# =======================================
# 7. COMPARISON PAR HORIZON CALENDAIRE
# =======================================

''' Compare the performance of Rough Heston and Classical Heston
for each calendar horizon. For each maturity, RMSE, MAE
and maximum errors are computed, then the model with the
lowest RMSE is identified. The results are then grouped
in a summary table, exported to CSV and displayed in the console.'''
rows = []

for days, group in df.groupby(
    "T_days",
    sort=True
):
    market = group["IV_market"].to_numpy(dtype=float)
    rh_model = group["IV_RH"].to_numpy(dtype=float)
    h_model = group["IV_Heston"].to_numpy(dtype=float)

    rh_met = metrics(
        market,
        rh_model
    )

    h_met = metrics(
        market,
        h_model
    )

    rmse_diff = (
        h_met["RMSE"]
        - rh_met["RMSE"]
    )

    rmse_gain_pct = (
        100.0
        * rmse_diff
        / h_met["RMSE"]
    )

    if rh_met["RMSE"] < h_met["RMSE"]:
        winner = "Rough Heston"
    elif rh_met["RMSE"] > h_met["RMSE"]:
        winner = "Classical Heston"
    else:
        winner = "Equal"

    rows.append({
        "T_days": int(days),
        "N": int(len(group)),
        "N_maturites_effectives": int(group["T"].nunique()),
        "RMSE_Rough_Heston": rh_met["RMSE"],
        "RMSE_Heston": h_met["RMSE"],
        "MAE_Rough_Heston": rh_met["MAE"],
        "MAE_Heston": h_met["MAE"],
        "MAX_ERROR_Rough_Heston": rh_met["MAX_ERROR"],
        "MAX_ERROR_Heston": h_met["MAX_ERROR"],
        "Gain_RMSE_RH_vs_Heston": rmse_diff,
        "Gain_RMSE_RH_vs_Heston_pct": rmse_gain_pct,
        "Meilleur_modele_RMSE": winner
    })

summary_horizon = pd.DataFrame(rows)

summary_horizon.to_csv(
    OUT_HORIZON,
    index=False,
    encoding="utf-8-sig"
)

n_rh_better = int(
    (
        summary_horizon["Meilleur_modele_RMSE"]
        == "Rough Heston"
    ).sum()
)

n_h_better = int(
    (
        summary_horizon["Meilleur_modele_RMSE"]
        == "Classical Heston"
    ).sum()
)

print()
print("=" * 72)
print("RMSE BY CALENDAR HORIZON")
print("=" * 72)
print()

print(
    summary_horizon[
        [
            "T_days",
            "N",
            "RMSE_Rough_Heston",
            "RMSE_Heston",
            "Gain_RMSE_RH_vs_Heston_pct",
            "Meilleur_modele_RMSE"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print(
    f"Rough Heston better on {n_rh_better}/"
    f"{len(summary_horizon)} horizons."
)

print(
    f"Classical Heston better on {n_h_better}/"
    f"{len(summary_horizon)} horizons."
)


# ===================================================
# 8. FIGURE: MARKET / ROUGH HESTON / HESTON SMILES
# ===================================================

days_list = sorted(
    df["T_days"].unique()
)

n_panels = len(days_list)
n_cols = 4
n_rows = int(
    np.ceil(n_panels / n_cols)
)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(18, 4.0 * n_rows)
)

axes = np.atleast_1d(
    axes
).ravel()

for ax, days in zip(
    axes,
    days_list
):
    group = df[
        df["T_days"] == days
    ].copy()

    ax.scatter(
        group["K_over_F0"],
        group["IV_market"],
        marker="o",
        s=28,
        label="Market",
        zorder=3
    )

    # Some calendar horizons contain several effective
    # maturities (SPX / SPXW coexistence). Each maturity is therefore
    # plotted separately to avoid artificially connecting
    # two different curves.
    for j, (_, gt) in enumerate(
        group.groupby(
            "T",
            sort=True
        )
    ):
        gt = gt.sort_values(
            "K_over_F0"
        )

        ax.plot(
            gt["K_over_F0"],
            gt["IV_RH"],
            linewidth=2.0,
            label=(
                "Rough Heston"
                if j == 0
                else None
            )
        )

        ax.plot(
            gt["K_over_F0"],
            gt["IV_Heston"],
            linestyle="--",
            linewidth=1.7,
            label=(
                "Classical Heston"
                if j == 0
                else None
            )
        )

    row = summary_horizon[
        summary_horizon["T_days"] == days
    ].iloc[0]

    ax.set_title(
        f"{days} days | "
        f"RH = {row['RMSE_Rough_Heston']:.4f} | "
        f"H = {row['RMSE_Heston']:.4f}"
    )

    ax.set_xlabel(
        r"Forward moneyness $K/F_0(T)$"
    )

    ax.set_ylabel(
        "Implied volatility"
    )

    ax.grid(
        alpha=0.25
    )

for ax in axes[n_panels:]:
    ax.axis("off")

handles, labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=3
)

fig.suptitle(
    "Global comparison: Rough Heston vs Classical Heston",
    y=1.01
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

fig.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========
# 9. END
# =========

print()
print("=" * 72)
print("FILES SAVED")
print("=" * 72)
print()
print(" -", OUT_GLOBAL)
print(" -", OUT_HORIZON)
print(" -", OUT_FIG)
print()
print("FIN DE LA COMPARISON GLOBALE.")
