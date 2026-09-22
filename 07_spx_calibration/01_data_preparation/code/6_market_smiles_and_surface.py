"""
Market implied volatility smiles and surface.

This script uses the final calibration sample to visualize the market implied
volatility surface and the volatility smiles observed across representative
maturities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ===========
# 1. PATHS
# ===========

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

RAW_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "00_raw_data"
)

PREP_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "01_data_preparation"
)

CSV_DIR = PREP_DIR / "results"
FIGURES_DIR = PREP_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# FINAL sample used for calibration (398 options)
# Input
file_path = (
    CSV_DIR
    / "05_calibration_sample.csv"
)

# Output
surface_path = (
    FIGURES_DIR
    / "07_market_implied_volatility_surface.png"
)

smiles_path = (
    FIGURES_DIR
    / "08_market_implied_volatility_smiles.png"
)


# ==========================================
# 2. MATURITIES TO DISPLAY FOR THE SMILES
# ==========================================

# Five representative horizons are selected from the global sample.
# Each of these horizons has a single effective maturity in the sample,
# which avoids mixing SPX and SPXW on the same curve.
#
# Coverage:
# 7 days   : very short term
# 31 days  : short term
# 185 days : medium term
# 367 days : approximately one year
# 647 days : longest maturity

SELECTED_DAYS = [
    7,
    31,
    185,
    367,
    647
]


# ====================================
# 3. LOADING THE CALIBRATION SAMPLE
# ====================================

df = pd.read_csv(
    file_path
)


# ============
# 4. CHECKS
# ============

required_columns = [
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "IV_market"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns in the calibration file: "
        + ", ".join(missing_columns)
    )


# Only observations usable for the figures are retained.

mask_valid = (
    np.isfinite(df["T_days"])
    & np.isfinite(df["T"])
    & np.isfinite(df["Strike"])
    & np.isfinite(df["K_over_F0"])
    & np.isfinite(df["IV_market"])
    & (df["T"] > 0)
    & (df["IV_market"] > 0)
)

df = (
    df.loc[mask_valid]
    .copy()
    .reset_index(drop=True)
)


print()
print("============================================================")
print("MARKET IMPLIED VOLATILITY STRUCTURE")
print("FINAL CALIBRATION SAMPLE")
print("============================================================")
print()

print(
    "Number of observations:",
    len(df)
)

print(
    "Number of calendar horizons:",
    df["T_days"].nunique()
)

print(
    "Number of effective maturities:",
    df["T"].nunique()
)

print(
    "Forward moneyness min / max:",
    f"{df['K_over_F0'].min():.6f}",
    "/",
    f"{df['K_over_F0'].max():.6f}"
)

print()


# ==========================================
# 5. EMPIRICAL IMPLIED VOLATILITY SURFACE
# ==========================================

fig = plt.figure(
    figsize=(10, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


# If Root is available, SPX and SPXW are distinguished.
# Otherwise, all observations are simply plotted.

if "Root" in df.columns:

    markers = {
        "SPX": "o",
        "SPXW": "^"
    }

    for root in [
        "SPX",
        "SPXW"
    ]:

        subset = df[
            df["Root"] == root
        ]

        if len(subset) == 0:
            continue

        ax.scatter(
            subset["K_over_F0"],
            subset["T"],
            subset["IV_market"],
            s=28,
            marker=markers[root],
            alpha=0.75,
            label=root
        )

    ax.legend()

else:

    ax.scatter(
        df["K_over_F0"],
        df["T"],
        df["IV_market"],
        s=28,
        alpha=0.75
    )


ax.set_xlabel(
    r"Forward moneyness $K/F_0(T)$"
)

ax.set_ylabel(
    r"Effective maturity $T$ (years)"
)

ax.set_zlabel(
    r"Implied volatility $\sigma_{\mathrm{impl}}$"
)

ax.set_title(
    "Market implied volatility surface"
)

plt.tight_layout()

plt.savefig(
    surface_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ======================================================
# 6. MARKET SMILES FOR FIVE REPRESENTATIVE MATURITIES
# ======================================================

available_days = set(
    df["T_days"]
    .round()
    .astype(int)
    .unique()
)

missing_days = [
    days
    for days in SELECTED_DAYS
    if days not in available_days
]

if missing_days:
    raise ValueError(
        "The following horizons are not present in "
        "the sample: "
        + ", ".join(
            str(x)
            for x in missing_days
        )
    )


plt.figure(
    figsize=(10, 6.5)
)


for days in SELECTED_DAYS:

    data_T = (
        df[
            df["T_days"].round().astype(int)
            == days
        ]
        .sort_values(
            "K_over_F0"
        )
        .copy()
    )

    # Useful check: the selected horizons must correspond
    # to a single effective maturity.
    n_effective_T = data_T["T"].nunique()

    if n_effective_T != 1:

        print(
            f"Warning: horizon {days} days -> "
            f"{n_effective_T} effective maturities."
        )

    plt.plot(
        data_T["K_over_F0"],
        data_T["IV_market"],
        marker="o",
        markersize=4,
        linewidth=1.4,
        label=f"{days} days"
    )


# Forward ATM line: K/F0(T) = 1

plt.axvline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    label="ATM forward"
)


plt.xlabel(
    r"Forward moneyness $K/F_0(T)$"
)

plt.ylabel(
    r"Implied volatility $\sigma_{\mathrm{impl}}$"
)

plt.title(
    "Market implied volatility smiles "
    "for several maturities"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    title="Horizon"
)

plt.tight_layout()

plt.savefig(
    smiles_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =============
# 7. SUMMARY
# =============

print()
print("============================================================")
print("SAVED FIGURES")
print("============================================================")
print()

print(
    "Surface:"
)

print(
    surface_path
)

print()

print(
    "Selected smiles:"
)

print(
    smiles_path
)

print()

print(
    "Horizons used for the smiles:",
    SELECTED_DAYS
)

print()
