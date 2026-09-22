"""
ATM skew term structure: market vs Rough Heston vs classical Heston.

This script does not recalibrate either model. It reads the detailed global
calibration results for Rough Heston and classical Heston.

For each effective maturity, the ATM skew is estimated consistently for the
market, Rough Heston, and classical Heston using the local regression

    sigma_impl(k,T) = a(T) + Skew(T) * k,
    k = log(K/F0(T)),

over the N_LOCAL observations closest to ATM-forward.

When a calendar horizon contains several effective maturities (SPX / SPXW),
the skews are aggregated using a weighted average based on the number of local
points used.

This script only performs the global term-structure analysis. It deliberately
does not include any log-log regression or T^(H-1/2) power-law test, which is
reserved for the short-term analysis.
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

CSV_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RH_FILE = (
    CALIB_CSV_DIR
    / "01_rough_heston_global_calibration_results.csv"
)

HESTON_FILE = (
    CALIB_CSV_DIR
    / "03_heston_global_calibration_results.csv"
)

OUT_EFFECTIVE = (
    CSV_DIR
    / "03_atm_skew_by_effective_maturity.csv"
)

OUT_CALENDAR = (
    CSV_DIR
    / "04_atm_skew_by_calendar_horizon.csv"
)

OUT_FIG = (
    FIGURES_DIR
    / "04_atm_skew_term_structure.png"
)


# =================================
# 2. LOCAL ESTIMATION PARAMETERS
# =================================

# Same convention as for the market ATM skew constructed during data preparation:
# at most 7 observations closest to ATM-forward.
N_LOCAL = 7

# Minimum number of points required to estimate a local slope.
MIN_LOCAL = 3


# ============================
# 3. LOADING GLOBAL RESULTS
# ============================

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
    "Root",
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
    "Root",
    "IV_market",
    "IV_Heston",
]

missing_rh = [
    col
    for col in required_rh
    if col not in rh.columns
]

missing_h = [
    col
    for col in required_h
    if col not in he.columns
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

rh_keep = rh[
    [
        "Contract",
        "T_days",
        "T",
        "Strike",
        "K_over_F0",
        "Option_type",
        "Root",
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
        "Root",
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
        "The two calibrations do not use exactly "
        "the same contracts.\n"
        + unmatched.to_string(index=False)
    )

merged = merged.drop(columns="_merge")


# ==========================================
# 6. CONSISTENCY CHECK BETWEEN BOTH FILES
# ==========================================

numeric_common = [
    "T_days",
    "T",
    "Strike",
    "K_over_F0",
    "IV_market",
]

for col in numeric_common:

    a = merged[
        f"{col}_RH_file"
    ].to_numpy(dtype=float)

    b = merged[
        f"{col}_H_file"
    ].to_numpy(dtype=float)

    if not np.allclose(
        a,
        b,
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError(
            f"Inconsistency between the files for column {col}."
        )


for col in [
    "Option_type",
    "Root",
]:

    a = (
        merged[f"{col}_RH_file"]
        .astype(str)
        .to_numpy()
    )

    b = (
        merged[f"{col}_H_file"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(a, b):
        raise ValueError(
            f"Inconsistency between the files for column {col}."
        )


# ============================
# 7. COMMON ANALYSIS SAMPLE
# ============================

df = pd.DataFrame({
    "Contract": merged["Contract"],
    "T_days": merged["T_days_RH_file"],
    "T": merged["T_RH_file"],
    "Strike": merged["Strike_RH_file"],
    "K_over_F0": merged["K_over_F0_RH_file"],
    "Option_type": merged["Option_type_RH_file"],
    "Root": merged["Root_RH_file"],
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

df = (
    df.loc[valid]
    .copy()
    .reset_index(drop=True)
)

df["Calendar_days"] = (
    df["T_days"]
    .round()
    .astype(int)
)

df["log_forward_moneyness"] = np.log(
    df["K_over_F0"]
)


print()
print("=" * 78)
print("ATM SKEW TERM STRUCTURE")
print("MARKET vs ROUGH HESTON vs CLASSICAL HESTON")
print("=" * 78)

print("Common observations:", len(df))
print(
    "Calendar horizons:",
    df["Calendar_days"].nunique(),
)
print(
    "Effective maturities:",
    df["T"].nunique(),
)
print(
    "Maximum number of local points:",
    N_LOCAL,
)


# ===============================
# 8. LOCAL ATM SKEW ESTIMATION
# ===============================

def local_atm_slope(
    group,
    y_col,
    n_local=N_LOCAL,
):
    """
    Estimates:
        sigma_impl = intercept + slope * log(K/F0)

    over the n_local observations closest to k = 0.

    The same set of strikes is used for all three series,
    because the local point selection depends only on moneyness.
    """

    g = group[
        np.isfinite(
            group["log_forward_moneyness"]
        )
        & np.isfinite(
            group[y_col]
        )
    ].copy()

    if len(g) < MIN_LOCAL:

        return {
            "Skew_ATM": np.nan,
            "IV_ATM_estimee": np.nan,
            "R2_local": np.nan,
            "N_local": int(len(g)),
            "K_over_F0_min_local": np.nan,
            "K_over_F0_max_local": np.nan,
            "Brackets_ATM": False,
        }

    g["abs_k"] = np.abs(
        g["log_forward_moneyness"]
    )

    g = (
        g.sort_values("abs_k")
        .head(
            min(
                n_local,
                len(g),
            )
        )
        .sort_values(
            "log_forward_moneyness"
        )
        .copy()
    )

    x = g[
        "log_forward_moneyness"
    ].to_numpy(dtype=float)

    y = g[
        y_col
    ].to_numpy(dtype=float)

    X = np.column_stack([
        np.ones(len(x)),
        x,
    ])

    coef, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    intercept, slope = coef

    fitted = X @ coef

    ss_res = np.sum(
        (y - fitted) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    return {
        "Skew_ATM": float(slope),
        "IV_ATM_estimee": float(intercept),
        "R2_local": (
            float(r2)
            if np.isfinite(r2)
            else np.nan
        ),
        "N_local": int(len(g)),
        "K_over_F0_min_local": float(
            np.exp(np.min(x))
        ),
        "K_over_F0_max_local": float(
            np.exp(np.max(x))
        ),
        "Brackets_ATM": bool(
            np.min(x) <= 0.0 <= np.max(x)
        ),
    }


# ================================
# 9. SKEW BY EFFECTIVE MATURITY
# ================================

rows = []

for T, group in df.groupby(
    "T",
    sort=True,
):

    # The ATM point selection depends only on K/F0.
    # The three regressions are therefore estimated on the same strikes.
    market = local_atm_slope(
        group,
        "IV_market",
    )

    rough = local_atm_slope(
        group,
        "IV_RH",
    )

    heston = local_atm_slope(
        group,
        "IV_Heston",
    )

    calendar_days = int(
        group["Calendar_days"].iloc[0]
    )

    roots = ",".join(
        sorted(
            group["Root"]
            .astype(str)
            .unique()
        )
    )

    rows.append({

        "T": float(T),

        "Effective_days": (
            float(T) * 365.25
        ),

        "Calendar_days": calendar_days,

        "Root": roots,

        "N_total": int(
            len(group)
        ),

        "N_local": market["N_local"],

        "K_over_F0_min_local":
            market["K_over_F0_min_local"],

        "K_over_F0_max_local":
            market["K_over_F0_max_local"],

        "Brackets_ATM":
            market["Brackets_ATM"],

        "Skew_Market":
            market["Skew_ATM"],

        "Skew_Rough_Heston":
            rough["Skew_ATM"],

        "Skew_Heston":
            heston["Skew_ATM"],

        "IV_ATM_Market":
            market["IV_ATM_estimee"],

        "IV_ATM_Rough_Heston":
            rough["IV_ATM_estimee"],

        "IV_ATM_Heston":
            heston["IV_ATM_estimee"],

        "R2_local_Market":
            market["R2_local"],

        "R2_local_Rough_Heston":
            rough["R2_local"],

        "R2_local_Heston":
            heston["R2_local"],
    })


skew_effective = pd.DataFrame(
    rows
)


print()
print("=" * 78)
print("ATM SKEW BY EFFECTIVE MATURITY")
print("=" * 78)

print(
    skew_effective[
        [
            "Effective_days",
            "Calendar_days",
            "Root",
            "N_local",
            "Brackets_ATM",
            "Skew_Market",
            "Skew_Rough_Heston",
            "Skew_Heston",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ======================================
# 10. AGGREGATION BY CALENDAR HORIZON
# ======================================

# For 3, 66, 94, and 129 days, the sample contains two effective
# maturities. The slopes are therefore aggregated using a weighted average based on
# the number of local points used, exactly as for the market skew
# constructed during data preparation.

calendar_rows = []

for days, group in skew_effective.groupby(
    "Calendar_days",
    sort=True,
):

    weights = group[
        "N_local"
    ].to_numpy(dtype=float)

    def weighted_mean(column):

        values = group[
            column
        ].to_numpy(dtype=float)

        mask = (
            np.isfinite(values)
            & np.isfinite(weights)
            & (weights > 0)
        )

        if not np.any(mask):
            return np.nan

        return float(
            np.average(
                values[mask],
                weights=weights[mask],
            )
        )

    skew_market = weighted_mean(
        "Skew_Market"
    )

    skew_rough = weighted_mean(
        "Skew_Rough_Heston"
    )

    skew_heston = weighted_mean(
        "Skew_Heston"
    )

    valid_weights = (
        np.isfinite(weights)
        & (weights > 0)
    )

    calendar_rows.append({

        "Calendar_days":
            int(days),

        "N_effective_maturities":
            int(len(group)),

        "N_local_total":
            int(
                np.sum(
                    weights[valid_weights]
                )
            ),

        "Skew_Market":
            skew_market,

        "Skew_Rough_Heston":
            skew_rough,

        "Skew_Heston":
            skew_heston,

        "Abs_Skew_Market":
            abs(skew_market)
            if np.isfinite(skew_market)
            else np.nan,

        "Abs_Skew_Rough_Heston":
            abs(skew_rough)
            if np.isfinite(skew_rough)
            else np.nan,

        "Abs_Skew_Heston":
            abs(skew_heston)
            if np.isfinite(skew_heston)
            else np.nan,
    })


skew_calendar = pd.DataFrame(
    calendar_rows
)


print()
print("=" * 78)
print("ATM SKEW BY CALENDAR HORIZON")
print("=" * 78)

print(
    skew_calendar.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ======================================
# 11. FIGURE: ATM SKEW TERM STRUCTURE
# ======================================

# Two panels:
#
# left  : global structure from 3 to 647 days;
# right : zoom from 1 to 31 days.
#
# The 1-day maturity is deliberately isolated from the lines in the
# short-term panel because its market skew can be highly atypical and visually
# dominate the structure of the other maturities.

fig, (
    ax_global,
    ax_short,
) = plt.subplots(
    1,
    2,
    figsize=(15, 6),
    gridspec_kw={
        "width_ratios": [
            1.65,
            1.0,
        ]
    },
)


# ---------------------------------------
# 11.1 Global structure: 3 to 647 days
# ---------------------------------------

regular = skew_calendar[
    (skew_calendar["Calendar_days"] >= 3)
].copy()

ax_global.plot(
    regular["Calendar_days"],
    regular["Skew_Market"],
    marker="o",
    linewidth=1.9,
    label="Market",
)

ax_global.plot(
    regular["Calendar_days"],
    regular["Skew_Rough_Heston"],
    marker="o",
    linewidth=1.9,
    label="Rough Heston",
)

ax_global.plot(
    regular["Calendar_days"],
    regular["Skew_Heston"],
    marker="o",
    linestyle="--",
    linewidth=1.8,
    label="Classical Heston",
)

ax_global.axhline(
    0.0,
    linewidth=0.9,
    linestyle=":",
    alpha=0.7,
)

ax_global.set_xlabel(
    "Calendar horizon (days)"
)

ax_global.set_ylabel(
    r"Skew ATM "
    r"$\left.\partial \sigma_{\mathrm{impl}}/"
    r"\partial \log(K/F_0)\right|_{ATM}$"
)

ax_global.set_title(
    "Global ATM skew term structure (3–647 days)"
)

ax_global.grid(
    True,
    alpha=0.25,
)

ax_global.legend()


# --------------------------
# 11.2 Zoom: 1 to 31 days
# --------------------------

short = skew_calendar[
    skew_calendar["Calendar_days"] <= 31
].copy()

short_regular = short[
    short["Calendar_days"] >= 3
].copy()

one_day = short[
    short["Calendar_days"] == 1
].copy()


# The lines start at 3 days.
ax_short.plot(
    short_regular["Calendar_days"],
    short_regular["Skew_Market"],
    marker="o",
    linewidth=1.9,
    label="Market",
)

ax_short.plot(
    short_regular["Calendar_days"],
    short_regular["Skew_Rough_Heston"],
    marker="o",
    linewidth=1.9,
    label="Rough Heston",
)

ax_short.plot(
    short_regular["Calendar_days"],
    short_regular["Skew_Heston"],
    marker="o",
    linestyle="--",
    linewidth=1.8,
    label="Classical Heston",
)


# The 1-day point is displayed but not connected to the other maturities.
if not one_day.empty:

    x1 = float(
        one_day["Calendar_days"].iloc[0]
    )

    for column, marker in [
        ("Skew_Market", "o"),
        ("Skew_Rough_Heston", "s"),
        ("Skew_Heston", "^"),
    ]:

        y1 = float(
            one_day[column].iloc[0]
        )

        if np.isfinite(y1):

            ax_short.scatter(
                [x1],
                [y1],
                s=65,
                marker=marker,
                zorder=5,
            )

    market_1d = float(
        one_day["Skew_Market"].iloc[0]
    )

    if np.isfinite(market_1d):

        ax_short.annotate(
            "1 day",
            xy=(
                x1,
                market_1d,
            ),
            xytext=(
                8,
                8,
            ),
            textcoords="offset points",
        )


ax_short.axhline(
    0.0,
    linewidth=0.9,
    linestyle=":",
    alpha=0.7,
)

ax_short.set_xlabel(
    "Calendar horizon (days)"
)

ax_short.set_ylabel(
    "Skew ATM"
)

ax_short.set_title(
    "Zoom on short maturities"
)

ax_short.set_xticks(
    [
        1,
        3,
        7,
        15,
        31,
    ]
)

ax_short.grid(
    True,
    alpha=0.25,
)


# --------------------
# 11.3 Common title
# --------------------

fig.suptitle(
    "ATM skew term structure: "
    "Market vs Rough Heston vs classical Heston",
    fontsize=15,
    y=0.995,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)

fig.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ==============
# 12. EXPORTS
# ==============

skew_effective.to_csv(
    OUT_EFFECTIVE,
    index=False,
)

skew_calendar.to_csv(
    OUT_CALENDAR,
    index=False,
)


print()
print("=" * 78)
print("FILES SAVED")
print("=" * 78)

print(
    " -",
    OUT_EFFECTIVE,
)

print(
    " -",
    OUT_CALENDAR,
)

print(
    " -",
    OUT_FIG,
)

print()
print(
    "No log-log regression is performed in this script: "
    "the power-law analysis is reserved for the short term."
)

print()
print("Done.")
