"""
06 - Short-term analysis
2 - Short-term smiles - Market vs Rough Heston vs classical Heston

This script does not recalibrate any model.
It directly reads the market implied volatilities and model implied volatilities
already produced in 05 - Short-term calibration.

The final figure contains four panels corresponding to four representative
horizons of the global sample.

Output:
- figures/03_short_term_market_vs_models_implied_volatility_smiles.png
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
    / "05_short_term_calibration"
    / "results"
)

ANALYSIS_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "06_short_term_analysis"
)
FIGURES_DIR = ANALYSIS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RH_FILE = (
    CALIB_CSV_DIR
    / "01_rough_heston_short_term_calibration_results.csv"
)

HESTON_FILE = (
    CALIB_CSV_DIR
    / "03_heston_short_term_calibration_results.csv"
)

OUTPUT_FIG = (
    FIGURES_DIR
    / "03_short_term_market_vs_models_implied_volatility_smiles.png"
)


# ===============================
# 2. REPRESENTATIVE MATURITIES
# ===============================

# Four horizons distributed across the entire term structure.
#
# 7 days   : very short term
# 31 days  : about one month
# 185 days : medium term
# 647 days : longest maturity
#
# Each of these horizons has a single effective maturity in the sample,
# which avoids mixing SPX and SPXW in the same panel.

SELECTED_DAYS = [1, 3, 7, 15, 31]


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
        "The two calibrations do not cover exactly the same "
        "contracts.\n"
        + missing.to_string(index=False)
    )

df = df.drop(columns="_merge")


# ================================
# 6. CONSISTENCY OF COMMON DATA
# ================================

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
            f"Inconsistency between the two files for {col}."
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
    if subset.empty:
        raise ValueError(
            f"No observation for the {days}-day horizon."
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


# =====================================================
# 9. SHORT-TERM FIGURE: 3 PANELS ON TOP, 2 ON BOTTOM
# =====================================================

fig = plt.figure(figsize=(15, 8.5))

gs = fig.add_gridspec(
    2,
    6,
    hspace=0.38,
    wspace=0.45,
)

axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[0, 4:6]),
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 3:5]),
]

for ax, days in zip(axes, SELECTED_DAYS):
    g = df[df["T_days"] == days].copy()

    met_rh = metrics(g["IV_market"], g["IV_RH"])
    met_h = metrics(g["IV_market"], g["IV_Heston"])

    ax.scatter(
        g["K_over_F0"],
        g["IV_market"],
        s=34,
        marker="o",
        label="Market",
        zorder=4,
    )

    # If several effective maturities coexist for the same horizon,
    # they are plotted separately.
    for j, (_, gt) in enumerate(g.groupby("T", sort=True)):
        gt = gt.sort_values("K_over_F0")

        ax.plot(
            gt["K_over_F0"],
            gt["IV_RH"],
            linewidth=2.0,
            label="Rough Heston" if j == 0 else None,
            zorder=3,
        )

        ax.plot(
            gt["K_over_F0"],
            gt["IV_Heston"],
            linewidth=1.8,
            linestyle="--",
            label="Classical Heston" if j == 0 else None,
            zorder=2,
        )

    ax.axvline(
        1.0,
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
    )

    ax.set_title(
        f"{days} days | "
        f"RH = {met_rh['RMSE']:.4f} | "
        f"H = {met_h['RMSE']:.4f}"
    )

    ax.set_xlabel(r"Forward moneyness $K/F_0(T)$")
    ax.set_ylabel(r"Implied volatility $\sigma_{\mathrm{impl}}$")
    ax.grid(True, alpha=0.25)


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
    "Short-term implied volatility smiles: "
    "Market vs Rough Heston vs classical Heston",
    fontsize=15,
    y=0.995,
)

fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98)


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
        f"{days:3d} jours | "
        f"N={len(g):2d} | "
        f"RMSE RH={met_rh['RMSE']:.6f} | "
        f"RMSE Heston={met_h['RMSE']:.6f}"
    )

print("\nFigure saved:")
print(OUTPUT_FIG)
print("\nDone.")
