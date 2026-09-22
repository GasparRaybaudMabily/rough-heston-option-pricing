"""
Market vs Rough Heston vs classical Heston implied volatility smiles.

This script does not recalibrate either model. It directly reads the market
implied volatilities and model implied volatilities already produced in the
global calibration section.

The final figure contains four panels corresponding to four representative
horizons from the global sample.
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
    ROOT_DIR
    / "07_spx_calibration"
    / "03_global_calibration"
    / "results"
)

ANALYSIS_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "04_global_analysis"
)

FIGURES_DIR = ANALYSIS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RH_FILE = (
    CALIB_CSV_DIR
    / "01_rough_heston_global_calibration_results.csv"
)

HESTON_FILE = (
    CALIB_CSV_DIR
    / "03_heston_global_calibration_results.csv"
)

OUTPUT_FIG = (
    FIGURES_DIR
    / "03_market_vs_models_implied_volatility_smiles.png"
)


# ===============================
# 2. REPRESENTATIVE MATURITIES
# ===============================

# Four horizons distributed across the entire term structure.
#
# 7 days   : very short term
# 31 days  : approximately one month
# 185 days : medium term
# 647 days : longest maturity
#
# Each of these horizons has a single effective maturity in the sample,
# which avoids mixing SPX and SPXW in the same panel.

SELECTED_DAYS = [
    7,
    31,
    185,
    647,
]


# =================================
# 3. LOADING CALIBRATION RESULTS
# =================================

if not RH_FILE.exists():
    raise FileNotFoundError(
        f"Rough Heston file not found: {RH_FILE}"
    )

if not HESTON_FILE.exists():
    raise FileNotFoundError(
        f"Classical Heston file not found: {HESTON_FILE}"
    )

rh = pd.read_csv(RH_FILE)
he = pd.read_csv(HESTON_FILE)


# ============
# 4. CHECKS
# ============

required_rh = [
    "Contract",
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "IV_market",
    "IV_RH",
]

required_h = [
    "Contract",
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "IV_market",
    "IV_Heston",
]

missing_rh = [
    c for c in required_rh
    if c not in rh.columns
]

missing_h = [
    c for c in required_h
    if c not in he.columns
]

if missing_rh:
    raise ValueError(
        "Missing columns in the Rough Heston file: "
        + ", ".join(missing_rh)
    )

if missing_h:
    raise ValueError(
        "Missing columns in the classical Heston file: "
        + ", ".join(missing_h)
    )

if rh["Contract"].duplicated().any():
    raise ValueError(
        "The Rough Heston file contains duplicated contracts."
    )

if he["Contract"].duplicated().any():
    raise ValueError(
        "The classical Heston file contains duplicated contracts."
    )


# ==============================
# 5. MERGING THE SAME OPTIONS
# ==============================

df = rh[
    [
        "Contract",
        "T_days",
        "T",
        "Strike",
        "K_over_F0",
        "IV_market",
        "IV_RH",
    ]
].merge(
    he[
        [
            "Contract",
            "T_days",
            "T",
            "Strike",
            "K_over_F0",
            "IV_market",
            "IV_Heston",
        ]
    ],
    on="Contract",
    how="outer",
    suffixes=("_RH_file", "_H_file"),
    indicator=True,
)

if not (df["_merge"] == "both").all():
    missing = df[df["_merge"] != "both"][
        ["Contract", "_merge"]
    ]

    raise ValueError(
        "The two calibrations do not use exactly "
        "the same contracts.\n"
        + missing.to_string(index=False)
    )

df = df.drop(columns="_merge")


# =============================
# 6. COMMON DATA CONSISTENCY
# =============================

for col in [
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "IV_market",
]:
    a = df[f"{col}_RH_file"].to_numpy(dtype=float)
    b = df[f"{col}_H_file"].to_numpy(dtype=float)

    if not np.allclose(
        a,
        b,
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError(
            f"Inconsistency between both files for {col}."
        )

df = pd.DataFrame({
    "Contract": df["Contract"],
    "T_days": df["T_days_RH_file"],
    "T": df["T_RH_file"],
    "Strike": df["Strike_RH_file"],
    "K_over_F0": df["K_over_F0_RH_file"],
    "IV_market": df["IV_market_RH_file"],
    "IV_RH": df["IV_RH"],
    "IV_Heston": df["IV_Heston"],
})

valid = (
    np.isfinite(df["T_days"])
    & np.isfinite(df["T"])
    & np.isfinite(df["K_over_F0"])
    & np.isfinite(df["IV_market"])
    & np.isfinite(df["IV_RH"])
    & np.isfinite(df["IV_Heston"])
    & (df["T"] > 0)
    & (df["K_over_F0"] > 0)
    & (df["IV_market"] > 0)
)

df = (
    df.loc[valid]
    .copy()
    .reset_index(drop=True)
)

df["T_days"] = (
    df["T_days"]
    .round()
    .astype(int)
)


# ================================
# 7. CHECKING SELECTED HORIZONS
# ================================

available_days = set(
    df["T_days"].unique()
)

missing_days = [
    days
    for days in SELECTED_DAYS
    if days not in available_days
]

if missing_days:
    raise ValueError(
        "Horizons missing from the calibration results: "
        + ", ".join(str(x) for x in missing_days)
    )

for days in SELECTED_DAYS:
    subset = df[df["T_days"] == days]

    n_effective = subset["T"].nunique()

    if n_effective != 1:
        raise ValueError(
            f"The {days}-day horizon contains "
            f"{n_effective} effective maturities. "
            "Choose a horizon with a single effective maturity "
            "or adapt the plot."
        )


# ===================
# 8. LOCAL METRICS
# ===================

def metrics(market, model):
    market = np.asarray(market, dtype=float)
    model = np.asarray(model, dtype=float)

    mask = (
        np.isfinite(market)
        & np.isfinite(model)
    )

    error = model[mask] - market[mask]

    if len(error) == 0:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
        }

    return {
        "RMSE": float(
            np.sqrt(np.mean(error**2))
        ),
        "MAE": float(
            np.mean(np.abs(error))
        ),
    }


# ==================
# 9. 2 x 2 FIGURE
# ==================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
)

axes = axes.ravel()

for ax, days in zip(
    axes,
    SELECTED_DAYS,
):
    g = (
        df[df["T_days"] == days]
        .sort_values("K_over_F0")
        .copy()
    )

    met_rh = metrics(
        g["IV_market"],
        g["IV_RH"],
    )

    met_h = metrics(
        g["IV_market"],
        g["IV_Heston"],
    )

    # Market: discrete observations.
    ax.scatter(
        g["K_over_F0"],
        g["IV_market"],
        s=34,
        marker="o",
        label="Market",
        zorder=4,
    )

    # Rough Heston.
    ax.plot(
        g["K_over_F0"],
        g["IV_RH"],
        linewidth=2.0,
        label="Rough Heston",
        zorder=3,
    )

    # Classical Heston.
    ax.plot(
        g["K_over_F0"],
        g["IV_Heston"],
        linewidth=1.8,
        linestyle="--",
        label="Classical Heston",
        zorder=2,
    )

    # ATM forward.
    ax.axvline(
        1.0,
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
    )

    ax.set_title(
        f"{days} days\n"
        f"RMSE RH = {met_rh['RMSE']:.4f} | "
        f"RMSE Heston = {met_h['RMSE']:.4f}"
    )

    ax.set_xlabel(
        r"Forward moneyness $K/F_0(T)$"
    )

    ax.set_ylabel(
        r"Implied volatility $\sigma_{\mathrm{impl}}$"
    )

    ax.grid(
        True,
        alpha=0.25,
    )


# ==============================
# 10. COMMON LEGEND AND TITLE
# ==============================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=3,
    frameon=True,
)

fig.suptitle(
    "Implied volatility smiles: "
    "Market vs Rough Heston vs classical Heston",
    fontsize=15,
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.90]
)


# =============
# 11. SAVING
# =============

fig.savefig(
    OUTPUT_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ==============
# 12. SUMMARY
# ==============

print("\n" + "=" * 76)
print("SMILES - MARKET vs ROUGH HESTON vs CLASSICAL HESTON")
print("=" * 76)

print("Displayed horizons:", SELECTED_DAYS)
print("Number of available observations:", len(df))

for days in SELECTED_DAYS:
    g = df[df["T_days"] == days]

    met_rh = metrics(
        g["IV_market"],
        g["IV_RH"],
    )

    met_h = metrics(
        g["IV_market"],
        g["IV_Heston"],
    )

    print(
        f"{days:3d} days | "
        f"N={len(g):2d} | "
        f"RMSE RH={met_rh['RMSE']:.6f} | "
        f"RMSE Heston={met_h['RMSE']:.6f}"
    )

print("\nFigure saved to:")
print(OUTPUT_FIG)
print("\nDone.")
