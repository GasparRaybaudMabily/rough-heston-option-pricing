"""
Spot level and dividend yield estimation from SPX option data.

This script estimates S0 and the continuous dividend yield q using
call-put parity together with the GSW / Svensson zero-coupon yield curve.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================
# 1. PARAMETERS AND PATHS
# ==========================

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
    FIGURES_DIR
    / "03_spot_and_dividend_yield_estimation.png"
)

# Difference between the morning SPX settlement
# and the end-of-day SPXW settlement
delta_hours_SPX_SPXW = 6.5


# =====================================================
# 2. GSW / SVENSSON PARAMETERS ON SEPTEMBER 13, 2022
# =====================================================

# Source:
# Board of Governors of the Federal Reserve System
# Gürkaynak, Sack and Wright (GSW)
# U.S. Treasury yield curve
#
# Date: September 13, 2022
#
# The rates obtained with this parameterization are
# continuously compounded zero-coupon rates, expressed in %.

BETA0 = 4.132531172114100
BETA1 = -0.483330087369049
BETA2 = 249.569929179392005
BETA3 = -250.639461221968986
TAU1 = 2.726371171770240
TAU2 = 2.748541956813830


# =================================
# 3. GSW ZERO-COUPON YIELD CURVE
# =================================

def zero_coupon_rate_percent(T):
    """
    GSW / Svensson zero-coupon rate at maturity T.

    Parameter
    ---------
    T : float or array_like
        Maturity expressed in years.

    Returns
    -------
    z(T) as an annual percentage rate,
    continuously compounded.
    """

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
    """
    Zero-coupon rate as a decimal value.
    """

    return (
        zero_coupon_rate_percent(T)
        / 100.0
    )


def discount_factor(T):
    """
    Discount factor associated with the GSW curve:

        P(0,T) = exp(-z(T) * T)

    with z(T) continuously compounded.
    """

    T = np.asarray(
        T,
        dtype=float
    )

    return np.exp(
        -zero_coupon_rate(T) * T
    )


# =====================
# 4. DATASET LOADING
# =====================

df = pd.read_csv(
    file_path
)

df["Expiration Date"] = pd.to_datetime(
    df["Expiration Date"],
    format="%d-%m-%y",
    errors="coerce"
)


# ==============================================
# 5. IDENTIFICATION OF SPX AND SPXW CONTRACTS
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

df["Root"] = df["RootC"].fillna(
    df["RootP"]
)


# ===========================
# 6. MATURITY CONSTRUCTION
# ===========================

# Valuation date
t0 = pd.Timestamp(
    "2022-09-13"
)

df["T_days"] = (
    df["Expiration Date"]
    - t0
).dt.days

df["T_calendar"] = (
    df["T_days"]
    / 365.25
)


# Standard SPX contracts are settled in the morning.
# Their maturity is therefore reduced by 6.5 hours.
# SPXW contracts are settled at the end of the trading session.

df["T"] = np.where(
    df["Root"] == "SPX",
    df["T_calendar"]
    - delta_hours_SPX_SPXW
    / (24 * 365.25),
    df["T_calendar"]
)


# ===========================
# 7. MID-PRICE CALCULATION
# ===========================

df["Call_mid"] = (
    df["BidC"]
    + df["AskC"]
) / 2

df["Put_mid"] = (
    df["BidP"]
    + df["AskP"]
) / 2


# ==========================
# 8. GSW DISCOUNT FACTORS
# ==========================

# Rates and discount factors are computed only
# for strictly positive maturities.

positive_T = (
    df["T"] > 0
)

df["r_GSW"] = np.nan
df["Discount_factor"] = np.nan

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


# ==================================================
# 9. RECONSTRUCTION OF S(T) USING CALL-PUT PARITY
# ==================================================

# With a yield curve:
#
# C - P = S0 exp(-qT) - K P(0,T)
#
# where:
#
# P(0,T) = exp(-z(T) T)
#
# therefore:
#
# S_hat(T) = C - P + K P(0,T)

df["S_hat"] = (
    df["Call_mid"]
    - df["Put_mid"]
    + df["Strike"]
    * df["Discount_factor"]
)


# ===========================================
# 10. OBSERVATIONS USED FOR THE REGRESSION
# ===========================================

mask = (
    (df["T"] > 0)
    & (df["BidC"] > 0)
    & (df["AskC"] > 0)
    & (df["BidP"] > 0)
    & (df["AskP"] > 0)
    & (df["AskC"] >= df["BidC"])
    & (df["AskP"] >= df["BidP"])
    & (df["RootC"] == df["RootP"])
    & np.isfinite(df["S_hat"])
    & (df["S_hat"] > 0)
)

df_reg = df.loc[
    mask,
    [
        "Expiration Date",
        "Root",
        "T",
        "Strike",
        "r_GSW",
        "Discount_factor",
        "S_hat"
    ]
].copy()


# =====================================
# 11. ONE S(T) ESTIMATE PER MATURITY
# =====================================

# Several strikes exist for the same maturity.
#
# Theoretically:
#
# C - P + K P(0,T)
#
# should be independent of the strike.
#
# Therefore, as in the initial code, the median by maturity
# is retained in order to limit the influence
# of outlier quotes.

reg_by_T = (
    df_reg
    .groupby(
        [
            "Expiration Date",
            "Root",
            "T"
        ],
        as_index=False
    )
    .agg(
        S_hat=(
            "S_hat",
            "median"
        ),
        r_GSW=(
            "r_GSW",
            "median"
        ),
        Discount_factor=(
            "Discount_factor",
            "median"
        ),
        Nombre_observations=(
            "S_hat",
            "size"
        )
    )
)


# ========================
# 12. LINEAR REGRESSION
# ========================

# Theoretically:
#
# S_hat(T) = S0 exp(-qT)
#
# therefore:
#
# ln(S_hat(T)) = ln(S0) - q*T

reg_by_T["log_S_hat"] = np.log(
    reg_by_T["S_hat"]
)

T_values = (
    reg_by_T["T"].values
)

Y_values = (
    reg_by_T["log_S_hat"].values
)


# Regression:
#
# Y = beta*T + alpha

beta, alpha = np.polyfit(
    T_values,
    Y_values,
    1
)


# =============================
# 13. ESTIMATION OF S0 AND q
# =============================

S0_est = np.exp(
    alpha
)

q_est = -beta


print()
print(
    "============================================================"
)

print(
    "ESTIMATION OF S0 AND q USING CALL-PUT PARITY"
)

print(
    "GSW / SVENSSON YIELD CURVE"
)

print(
    "============================================================"
)

print()

print(
    "Number of maturity-root pairs used:",
    len(reg_by_T)
)

print()

print(
    f"Intercept alpha = {alpha:.10f}"
)

print(
    f"Slope beta      = {beta:.10f}"
)

print()

print(
    f"Estimated S0 = {S0_est:.6f}"
)

print(
    f"Estimated q  = {q_est:.8f}"
)

print(
    f"Estimated q  = {100*q_est:.6f} %"
)


# ====================================
# 14. INFORMATION ON THE RATES USED
# ====================================

print()
print(
    "============================================================"
)

print(
    "GSW RATES USED"
)

print(
    "============================================================"
)

print()

print(
    "Minimum maturity: "
    f"{reg_by_T['T'].min():.8f} year"
)

print(
    "Maximum maturity: "
    f"{reg_by_T['T'].max():.8f} year"
)

print()

print(
    "Minimum GSW rate: "
    f"{100 * reg_by_T['r_GSW'].min():.6f} %"
)

print(
    "Maximum GSW rate: "
    f"{100 * reg_by_T['r_GSW'].max():.6f} %"
)


# =============================
# 15. REGRESSION QUALITY: R²
# =============================

Y_pred = (
    alpha
    + beta * T_values
)

SS_res = np.sum(
    (
        Y_values
        - Y_pred
    )**2
)

SS_tot = np.sum(
    (
        Y_values
        - np.mean(Y_values)
    )**2
)

R2 = (
    1
    - SS_res / SS_tot
)


print()

print(
    f"R² = {R2:.8f}"
)


# ======================
# 16. REGRESSION PLOT
# ======================

plt.figure(
    figsize=(9, 6)
)

for root, color in [
    (
        "SPX",
        "tab:blue"
    ),
    (
        "SPXW",
        "tab:orange"
    )
]:

    root_data = reg_by_T[
        reg_by_T["Root"]
        == root
    ]

    plt.scatter(
        root_data["T"],
        root_data["log_S_hat"],
        s=35,
        color=color,
        label=root
    )


T_line = np.linspace(
    T_values.min(),
    T_values.max(),
    300
)

Y_line = (
    alpha
    + beta * T_line
)

plt.plot(
    T_line,
    Y_line,
    color="black",
    linewidth=1.6,
    label="Linear regression"
)

plt.xlabel(
    "Maturity T (years)"
)

plt.ylabel(
    r"$\ln(\widehat{S}(T))$"
)

plt.title(
    r"Estimation of $S_0$ and $q$ "
    r"using the GSW yield curve"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print(
    "Figure saved to:",
    output_path
)

plt.show()


# ==============
# 17. SUMMARY
# ==============

print()
print(
    "============================================================"
)

print(
    "VALUES TO USE IN THE FOLLOWING STEPS"
)

print(
    "============================================================"
)

print()

print(
    f"alpha = {alpha:.10f}"
)

print(
    f"beta  = {beta:.10f}"
)

print(
    f"S0    = {S0_est:.6f}"
)

print(
    f"q     = {q_est:.8f}"
)

print()

print(
    "Rates: GSW / Svensson zero-coupon curve "
    "on September 13, 2022"
)
