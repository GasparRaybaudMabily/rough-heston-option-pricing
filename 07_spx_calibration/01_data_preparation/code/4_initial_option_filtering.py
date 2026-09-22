"""
Initial filtering of the SPX option dataset.

This script applies liquidity, moneyness and no-arbitrage filters to SPX
and SPXW options, constructs an OTM option sample using forward moneyness,
and exports the filtered dataset and diagnostic figures.
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



file_path = (
    RAW_DIR
    / "spx_quotedata.csv"
)

output_path = (
    CSV_DIR
    / "04_filtered_otm_options.csv"
)

figure_path_1 = (
    FIGURES_DIR
    / "04_options_after_filtering.png"
)

figure_path_2 = (
    FIGURES_DIR
    / "05_strike_maturity_after_filtering.png"
)

# =======================
# 2. MARKET PARAMETERS
# =======================

# Values previously estimated using the GSW curve

S0 = 3936.710926
q = 0.0114029141

delta_hours_SPX_SPXW = 6.5

reference_date = pd.Timestamp(
    "2022-09-13"
)


# ===============================
# 3. GSW / SVENSSON PARAMETERS
# ===============================

# Board of Governors of the Federal Reserve System
# Gürkaynak, Sack and Wright (GSW)
# Date: September 13, 2022
#
# The rates are continuously compounded
# zero-coupon rates, expressed as percentages.

BETA0 = 4.132531172114100
BETA1 = -0.483330087369049
BETA2 = 249.569929179392005
BETA3 = -250.639461221968986

TAU1 = 2.726371171770240
TAU2 = 2.748541956813830


# =====================
# 4. GSW YIELD CURVE
# =====================

def zero_coupon_rate_percent(T):

    T = np.asarray(
        T,
        dtype=float
    )

    if np.any(T <= 0):

        raise ValueError(
            "Maturities T must be strictly positive."
        )

    x1 = T / TAU1
    x2 = T / TAU2

    term1 = (
        1.0 - np.exp(-x1)
    ) / x1

    term2 = (
        term1
        - np.exp(-x1)
    )

    term3 = (
        (1.0 - np.exp(-x2)) / x2
        - np.exp(-x2)
    )

    z = (
        BETA0
        + BETA1 * term1
        + BETA2 * term2
        + BETA3 * term3
    )

    return z


def zero_coupon_rate(T):

    return (
        zero_coupon_rate_percent(T)
        / 100.0
    )


def discount_factor(T):

    T = np.asarray(
        T,
        dtype=float
    )

    return np.exp(
        -zero_coupon_rate(T) * T
    )


# =====================
# 5. DATASET LOADING
# =====================

df = pd.read_csv(
    file_path
)


# ==============================================
# 6. IDENTIFICATION OF SPX AND SPXW CONTRACTS
# ==============================================

df["RootC"] = (
    df["Calls"]
    .astype(str)
    .str.extract(
        r"^(SPXW|SPX)",
        expand=False
    )
)

df["RootP"] = (
    df["Puts"]
    .astype(str)
    .str.extract(
        r"^(SPXW|SPX)",
        expand=False
    )
)

df["Root"] = (
    df["RootC"]
    .fillna(
        df["RootP"]
    )
)


# ==========================
# 7. DATES AND MATURITIES
# ==========================

df["Expiration Date"] = pd.to_datetime(
    df["Expiration Date"],
    format="%d-%m-%y"
)

df["T_days"] = (
    df["Expiration Date"]
    - reference_date
).dt.days

df["T_calendar"] = (
    df["T_days"]
    / 365.25
)


# Standard SPX contracts are settled in the morning.
# Their maturity is reduced by 6.5 hours compared with
# SPXW contracts settled at the end of the trading session.

df["T"] = np.where(
    df["Root"] == "SPX",
    df["T_calendar"]
    - delta_hours_SPX_SPXW
    / (24.0 * 365.25),
    df["T_calendar"]
)


# ================
# 8. MID-PRICES
# ================

df["Call_mid"] = (
    df["BidC"]
    + df["AskC"]
) / 2.0

df["Put_mid"] = (
    df["BidP"]
    + df["AskP"]
) / 2.0


# =====================
# 9. BID-ASK SPREADS
# =====================

df["Call_spread"] = (
    df["AskC"]
    - df["BidC"]
)

df["Put_spread"] = (
    df["AskP"]
    - df["BidP"]
)


df["Call_spread_rel"] = np.where(
    df["Call_mid"] > 0,
    df["Call_spread"]
    / df["Call_mid"],
    np.nan
)

df["Put_spread_rel"] = np.where(
    df["Put_mid"] > 0,
    df["Put_spread"]
    / df["Put_mid"],
    np.nan
)


# =================================================
# 10. YIELD CURVE, FORWARD AND FORWARD MONEYNESS
# =================================================

# K/S0 is retained only as a control/comparison column.
df["K_over_S0"] = (
    df["Strike"]
    / S0
)

df["r_GSW"] = np.nan
df["Discount_factor"] = np.nan
df["Forward"] = np.nan
df["K_over_F0"] = np.nan

positive_T = (
    df["T"] > 0
)

df.loc[
    positive_T,
    "r_GSW"
] = zero_coupon_rate(
    df.loc[
        positive_T,
        "T"
    ].values
)

df.loc[
    positive_T,
    "Discount_factor"
] = discount_factor(
    df.loc[
        positive_T,
        "T"
    ].values
)

# F0(T) = S0 exp(-qT) / P(0,T)
#       = S0 exp((r_GSW(T) - q) T)
df.loc[
    positive_T,
    "Forward"
] = (
    S0
    * np.exp(
        -q * df.loc[
            positive_T,
            "T"
        ].values
    )
    / df.loc[
        positive_T,
        "Discount_factor"
    ].values
)

df.loc[
    positive_T,
    "K_over_F0"
] = (
    df.loc[
        positive_T,
        "Strike"
    ].values
    / df.loc[
        positive_T,
        "Forward"
    ].values
)

print()
print("============================================================")
print("MONEYNESS FORWARD")
print("============================================================")
print("Definition used for filtering: K / F0(T)")
print(
    "Forward min/max:",
    f"{df.loc[positive_T, 'Forward'].min():.6f}",
    "/",
    f"{df.loc[positive_T, 'Forward'].max():.6f}"
)


# =======================
# 12. ARBITRAGE BOUNDS
# =======================

# With a yield curve:
#
# P(0,T) = exp(-z(T) T)
#
# The discounted strike becomes:
#
# K * P(0,T)

discounted_K = (
    df["Strike"]
    * df["Discount_factor"]
)

discounted_S = (
    S0
    * np.exp(
        -q * df["T"]
    )
)


# -------
# CALL
# -------

call_lower = np.maximum(
    discounted_S
    - discounted_K,
    0.0
)

call_upper = (
    discounted_S
)

df["Call_arbitrage_ok"] = (
    (df["Call_mid"] >= call_lower)
    & (df["Call_mid"] <= call_upper)
)


# ------
# PUT
# ------

put_lower = np.maximum(
    discounted_K
    - discounted_S,
    0.0
)

put_upper = (
    discounted_K
)

df["Put_arbitrage_ok"] = (
    (df["Put_mid"] >= put_lower)
    & (df["Put_mid"] <= put_upper)
)


# =======================
# 13. DISPLAY FUNCTION
# =======================

def print_step(name, mask):

    remaining = int(
        np.sum(mask)
    )

    print(
        f"{name:<45} : "
        f"{remaining:5d} options"
    )


# =====================
# 14. CALL FILTERING
# =====================

print()
print(
    "============================================================"
)
print(
    "CALL FILTERING"
)
print(
    "============================================================"
)


mask_call = np.ones(
    len(df),
    dtype=bool
)

print_step(
    "Initial dataset",
    mask_call
)


# -------------------------
# 14.1 Positive maturity
# -------------------------

mask_call &= (
    df["T"] > 0
)

print_step(
    "T > 0",
    mask_call
)


# -----------------------------------
# 14.2 Strictly positive Bid / Ask
# -----------------------------------

mask_call &= (
    (df["BidC"] > 0)
    & (df["AskC"] > 0)
)

print_step(
    "Bid > 0 and Ask > 0",
    mask_call
)


# -----------------------------
# 14.3 Bid / Ask consistency
# -----------------------------

mask_call &= (
    df["AskC"]
    >= df["BidC"]
)

print_step(
    "Ask >= Bid",
    mask_call
)


# -------------------------------
# 14.4 Relative spread <= 10 %
# -------------------------------

mask_call &= (
    df["Call_spread_rel"]
    <= 0.10
)

print_step(
    "Relative spread <= 10 %",
    mask_call
)


# ------------------------------
# 14.5 Positive Open Interest
# ------------------------------

mask_call &= (
    df["Open InterestC"]
    > 0
)

print_step(
    "Open Interest > 0",
    mask_call
)


# -----------------------
# 14.6 Moneyness range
# -----------------------

mask_call &= (
    (df["K_over_F0"] >= 0.70)
    & (df["K_over_F0"] <= 1.30)
)

print_step(
    "0.70 <= K/F0(T) <= 1.30",
    mask_call
)


# --------------------------------
# 14.7 No-arbitrage constraints
# --------------------------------

mask_call &= (
    df["Call_arbitrage_ok"]
)

print_step(
    "Arbitrage constraints",
    mask_call
)


df_call_filtered = df[
    mask_call
].copy()


# ====================
# 15. PUT FILTERING
# ====================

print()
print(
    "============================================================"
)
print(
    "PUT FILTERING"
)
print(
    "============================================================"
)


mask_put = np.ones(
    len(df),
    dtype=bool
)

print_step(
    "Initial dataset",
    mask_put
)


# -------------------------
# 15.1 Positive maturity
# -------------------------

mask_put &= (
    df["T"] > 0
)

print_step(
    "T > 0",
    mask_put
)


# -----------------------------------
# 15.2 Strictly positive Bid / Ask
# -----------------------------------

mask_put &= (
    (df["BidP"] > 0)
    & (df["AskP"] > 0)
)

print_step(
    "Bid > 0 and Ask > 0",
    mask_put
)


# -----------------------------
# 15.3 Bid / Ask consistency
# -----------------------------

mask_put &= (
    df["AskP"]
    >= df["BidP"]
)

print_step(
    "Ask >= Bid",
    mask_put
)


# -------------------------------
# 15.4 Relative spread <= 10 %
# -------------------------------

mask_put &= (
    df["Put_spread_rel"]
    <= 0.10
)

print_step(
    "Relative spread <= 10 %",
    mask_put
)


# ------------------------------
# 15.5 Positive Open Interest
# ------------------------------

mask_put &= (
    df["Open InterestP"]
    > 0
)

print_step(
    "Open Interest > 0",
    mask_put
)


# -----------------------
# 15.6 Moneyness range
# -----------------------

mask_put &= (
    (df["K_over_F0"] >= 0.70)
    & (df["K_over_F0"] <= 1.30)
)

print_step(
    "0.70 <= K/F0(T) <= 1.30",
    mask_put
)


# --------------------------------
# 15.7 No-arbitrage constraints
# --------------------------------

mask_put &= (
    df["Put_arbitrage_ok"]
)

print_step(
    "Arbitrage constraints",
    mask_put
)


df_put_filtered = df[
    mask_put
].copy()


# =====================================
# 16. CONSTRUCTION OF THE OTM SAMPLE
# =====================================


# ----------------------
# OTM PUTS: K < F0(T)
# ----------------------

puts_otm = df_put_filtered[
    df_put_filtered["Strike"]
    < df_put_filtered["Forward"]
].copy()

puts_otm["Option_type"] = "Put"

puts_otm["Market_price"] = (
    puts_otm["Put_mid"]
)

# Volatility provided directly by the CSV file.
# It will only be used as a reference / control.

puts_otm["IV_BDD"] = (
    puts_otm["IVP"]
)

puts_otm["Contract"] = (
    puts_otm["Puts"]
)


# ------------------------
# OTM CALLS: K >= F0(T)
# ------------------------

calls_otm = df_call_filtered[
    df_call_filtered["Strike"]
    >= df_call_filtered["Forward"]
].copy()

calls_otm["Option_type"] = "Call"

calls_otm["Market_price"] = (
    calls_otm["Call_mid"]
)

calls_otm["IV_BDD"] = (
    calls_otm["IVC"]
)

calls_otm["Contract"] = (
    calls_otm["Calls"]
)


# =======================
# 17. CALL / PUT MERGE
# =======================

columns_keep = [
    "Expiration Date",
    "T_days",
    "T",
    "Strike",
    "K_over_S0",
    "Forward",
    "K_over_F0",
    "Option_type",
    "Root",
    "Contract",
    "Market_price",
    "IV_BDD",
    "r_GSW",
    "Discount_factor"
]


df_otm = pd.concat(
    [
        puts_otm[
            columns_keep
        ],
        calls_otm[
            columns_keep
        ]
    ],
    ignore_index=True
)


df_otm = (
    df_otm
    .sort_values(
        by=[
            "T",
            "Strike"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ====================
# 18. FINAL SUMMARY
# ====================

print()
print(
    "============================================================"
)
print(
    "FINAL SUMMARY"
)
print(
    "============================================================"
)
print()


print(
    "Filtered calls:",
    len(
        df_call_filtered
    )
)

print(
    "Filtered puts:",
    len(
        df_put_filtered
    )
)

print()

print(
    "Retained OTM calls:",
    len(
        calls_otm
    )
)

print(
    "Retained OTM puts:",
    len(
        puts_otm
    )
)

print()

print(
    "Total number of OTM options:",
    len(
        df_otm
    )
)

print(
    "Number of distinct expiration dates:",
    df_otm[
        "Expiration Date"
    ].nunique()
)

print(
    "Number of distinct date-root pairs:",
    df_otm[
        [
            "Expiration Date",
            "Root"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


print()
print(
    "S0 used:",
    f"{S0:.6f}"
)

print(
    "q used:",
    f"{100*q:.6f} %"
)

print(
    "Rates: GSW / Svensson zero-coupon curve"
)


print()
print("OTM definition used:")
print("  Put  OTM : K < F0(T)")
print("  Call OTM : K >= F0(T)")
print(
    "K/F0(T) min/max in the OTM dataset:",
    f"{df_otm['K_over_F0'].min():.6f}",
    "/",
    f"{df_otm['K_over_F0'].max():.6f}"
)
print()

# ====================================
# 19. NUMBER OF OPTIONS BY MATURITY
# ====================================

count_by_T = (
    df_otm
    .groupby(
        [
            "Expiration Date",
            "Root",
            "T"
        ]
    )
    .size()
    .reset_index(
        name="Nombre_options"
    )
)


print()
print(
    "============================================================"
)
print(
    "OTM OPTIONS BY MATURITY"
)
print(
    "============================================================"
)
print()

print(
    count_by_T.to_string(
        index=False
    )
)


# ============
# 20. PLOT:
# NUMBER OF OPTIONS AFTER FILTERING
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    count_by_T["T"],
    count_by_T["Nombre_options"],
    marker="o"
)

plt.xlabel(
    "Maturity T (years)"
)

plt.ylabel(
    "Number of OTM options"
)

plt.title(
    "Number of options after filtering"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    figure_path_1,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============
# 21. PLOT:
# STRIKE / MATURITY
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.scatter(
    df_otm["T"],
    df_otm["Strike"],
    s=10
)

plt.axhline(
    S0,
    linestyle="--",
    label=r"$S_0$"
)

# OTM boundary actually used: K = F0(T)
forward_curve = (
    df_otm[["T", "Forward"]]
    .drop_duplicates()
    .sort_values("T")
)

plt.plot(
    forward_curve["T"],
    forward_curve["Forward"],
    linestyle=":",
    label=r"$F_0(T)$"
)

plt.xlabel(
    "Maturity T (years)"
)

plt.ylabel(
    "Strike K"
)

plt.title(
    "Distribution of OTM options after filtering"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    figure_path_2,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================
# 22. OPTIONAL PLOT:
# IV PROVIDED BY THE DATASET
# ============================================================

# WARNING:
# these are only the implied volatilities
# already contained in the CSV file.
#
# They do NOT yet constitute the final
# calibration target.

df_iv_plot = df_otm[
    np.isfinite(
        df_otm["IV_BDD"]
    )
    & (
        df_otm["IV_BDD"] > 0
    )
].copy()


plt.figure(
    figsize=(9, 5)
)

plt.scatter(
    df_iv_plot["Strike"],
    df_iv_plot["IV_BDD"],
    s=10
)

plt.axvline(
    S0,
    linestyle="--",
    label=r"$K=S_0$"
)

plt.xlabel(
    "Strike K"
)

plt.ylabel(
    "Provided implied volatility"
)

plt.title(
    "Implied volatilities provided in the dataset"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# =============
# 23. EXPORT
# =============

df_otm.to_csv(
    output_path,
    index=False
)


print()
print(
    "============================================================"
)
print(
    "EXPORT"
)
print(
    "============================================================"
)
print()

print(
    "Filtered dataset saved to:"
)

print(
    output_path
)

print()

print(
    "Figures saved to:"
)

print(
    figure_path_1
)

print(
    figure_path_2
)
