"""
Construction of the SPX calibration sample.

This script selects observations across 16 calendar horizons, computes market
implied volatilities by Black-Scholes inversion using the GSW / Svensson yield
curve, checks price reconstruction, and exports the final calibration sample.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import norm
from scipy.optimize import brentq


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
    CSV_DIR
    / "04_filtered_otm_options.csv"
)

output_path = (
    CSV_DIR
    / "05_calibration_sample.csv"
)

summary_path = (
    CSV_DIR
    / "06_calibration_sample_summary.csv"
)

figure_path = (
    FIGURES_DIR
    / "06_options_selected_for_calibration.png"
)


# =======================

# 2. MARKET PARAMETERS

# =======================

# Values previously estimated using the GSW curve

S0 = 3936.710926

q = 0.0114029141

# Maximum number of observations retained

# for each calendar horizon

N_options_per_horizon = 25


# ===============================

# 3. GSW / SVENSSON PARAMETERS

# ===============================

# Board of Governors of the Federal Reserve System

# Gürkaynak, Sack and Wright (GSW)

#

# Date: September 13, 2022

#

# Continuously compounded zero-coupon rates,

# expressed as percentages.

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

    return (

        BETA0

        + BETA1 * term1

        + BETA2 * term2

        + BETA3 * term3

    )


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


# ====================================================

# 5. LOADING THE DATASET FROM THE INITIAL FILTERING

# ====================================================

df = pd.read_csv(

    file_path

)

df["Expiration Date"] = pd.to_datetime(

    df["Expiration Date"]

)


# ================================

# 6. SELECTED CALENDAR HORIZONS

# ================================

selected_days = [

    1,

    3,

    7,

    15,

    31,

    66,

    94,

    129,

    185,

    220,

    248,

    276,

    311,

    367,

    458,

    647

]


print()

print("============================================================")

print("SELECTED CALENDAR HORIZONS")

print("============================================================")

for days in selected_days:

    print(

        f"{days:4d} days"

        f"   -> T_calendar = {days / 365.25:.8f}"

    )


# ================================

# 7. SPX / SPXW STRUCTURE CHECK

# ================================

required_columns = [

    "Expiration Date",

    "T_days",

    "T",

    "Strike",

    "K_over_S0",

    "Forward",

    "K_over_F0",

    "r_GSW",

    "Discount_factor",

    "Option_type",

    "Root",

    "Contract",

    "Market_price"

]

missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]

if missing_columns:

    raise ValueError(

        "Missing columns in the dataset from the "

        f"initial filtering: {missing_columns}"

    )


df = df[

    df["Root"].isin(

        ["SPX", "SPXW"]

    )

].copy()


valid_rows = (

    np.isfinite(df["T_days"])

    & np.isfinite(df["T"])

    & np.isfinite(df["Strike"])

    & np.isfinite(df["Forward"])

    & np.isfinite(df["K_over_F0"])

    & np.isfinite(df["r_GSW"])

    & np.isfinite(df["Discount_factor"])

    & np.isfinite(df["Market_price"])

    & (df["T"] > 0)

    & (df["Strike"] > 0)

    & (df["Market_price"] > 0)

)

df = (

    df.loc[valid_rows]

    .copy()

    .reset_index(drop=True)

)


# =======================================

# 8. REGULAR SELECTION OF OBSERVATIONS

# =======================================

def select_regular_observations(

    df_horizon,

    n_options

):

    """

    Selects at most n_options observations

    regularly across the forward moneyness space K/F0(T).

    SPX and SPXW remain distinct contracts.

    """

    data = (

        df_horizon

        .sort_values(

            by=[

                "K_over_F0",

                "Root",

                "T"

            ],

            kind="stable"

        )

        .drop_duplicates(

            subset=[

                "Root",

                "Contract"

            ]

        )

        .reset_index(drop=True)

        .copy()

    )

    n_available = len(data)

    if n_available <= n_options:

        return data


    positions = np.linspace(

        0,

        n_available - 1,

        n_options

    )

    positions = np.rint(

        positions

    ).astype(int)

    positions = np.unique(

        positions

    )


    if len(positions) < n_options:

        remaining = np.setdiff1d(

            np.arange(n_available),

            positions

        )

        n_missing = (

            n_options

            - len(positions)

        )

        positions = np.sort(

            np.concatenate(

                [

                    positions,

                    remaining[:n_missing]

                ]

            )

        )


    return (

        data.iloc[positions]

        .copy()

        .reset_index(drop=True)

    )


# ==================================

# 9. SELECTION OF THE 16 HORIZONS

# ==================================

selected_frames = []

selection_summary = []


print()

print("============================================================")

print("OPTION SELECTION")

print("============================================================")


for days in selected_days:

    df_horizon = df[

        df["T_days"] == days

    ].copy()


    if len(df_horizon) == 0:

        print()

        print(

            f"Horizon {days} day(s): "

            "no observation available."

        )

        continue


    n_spx_available = int(

        np.sum(

            df_horizon["Root"] == "SPX"

        )

    )

    n_spxw_available = int(

        np.sum(

            df_horizon["Root"] == "SPXW"

        )

    )


    selected = select_regular_observations(

        df_horizon=df_horizon,

        n_options=N_options_per_horizon

    )

    selected_frames.append(

        selected

    )


    n_spx_selected = int(

        np.sum(

            selected["Root"] == "SPX"

        )

    )

    n_spxw_selected = int(

        np.sum(

            selected["Root"] == "SPXW"

        )

    )


    selection_summary.append(

        {

            "T_days": days,

            "Available_total": len(df_horizon),

            "Available_SPX": n_spx_available,

            "Available_SPXW": n_spxw_available,

            "Selected_total": len(selected),

            "Selected_SPX": n_spx_selected,

            "Selected_SPXW": n_spxw_selected,

            "N_effective_T": selected["T"].nunique()

        }

    )


    print()

    print(

        f"Horizon: {days} day(s)"

    )

    print(

        f"Available: {len(df_horizon)} "

        f"(SPX={n_spx_available}, "

        f"SPXW={n_spxw_available})"

    )

    print(

        f"Selected : {len(selected)} "

        f"(SPX={n_spx_selected}, "

        f"SPXW={n_spxw_selected})"

    )


# ==============================================

# 10. CONSTRUCTION OF THE CALIBRATION DATASET

# ==============================================

if len(selected_frames) == 0:

    raise ValueError(

        "No observation was selected."

    )


df_calibration = pd.concat(

    selected_frames,

    ignore_index=True

)


df_calibration = (

    df_calibration

    .sort_values(

        by=[

            "T_days",

            "T",

            "Strike",

            "Root"

        ],

        kind="stable"

    )

    .reset_index(drop=True)

)


selection_summary = pd.DataFrame(

    selection_summary

)


print()

print("============================================================")

print("SAMPLE BEFORE BLACK-SCHOLES INVERSION")

print("============================================================")

print(

    "Number of options:",

    len(df_calibration)

)

print(

    "Number of calendar horizons:",

    df_calibration["T_days"].nunique()

)

print(

    "Number of effective maturities T:",

    df_calibration["T"].nunique()

)

print()

print(

    selection_summary.to_string(

        index=False

    )

)


# ==================================

# 11. GSW CURVE AND FORWARD CHECK

# ==================================

# The r_GSW, Discount_factor, Forward and K_over_F0 columns
# come directly from the initial forward filtering.
#
# Their consistency is simply checked:

discount_check = np.exp(
    -df_calibration["r_GSW"].values
    * df_calibration["T"].values
)

forward_check = (
    S0
    * np.exp(
        -q * df_calibration["T"].values
    )
    / df_calibration["Discount_factor"].values
)

max_discount_error = np.max(
    np.abs(
        df_calibration["Discount_factor"].values
        - discount_check
    )
)

max_forward_error = np.max(
    np.abs(
        df_calibration["Forward"].values
        - forward_check
    )
)

max_moneyness_error = np.max(
    np.abs(
        df_calibration["K_over_F0"].values
        - (
            df_calibration["Strike"].values
            / df_calibration["Forward"].values
        )
    )
)

print()
print("============================================================")
print("GSW CURVE / FORWARD CHECK")
print("============================================================")
print("Max Discount_factor error:", max_discount_error)
print("Max Forward error        :", max_forward_error)
print("Max K/F0(T) error        :", max_moneyness_error)

if max_discount_error > 1e-10:
    raise ValueError(
        "Inconsistency between r_GSW and Discount_factor."
    )

if max_forward_error > 1e-8:
    raise ValueError(
        "Inconsistency in the Forward column."
    )

if max_moneyness_error > 1e-12:
    raise ValueError(
        "Inconsistency in the K_over_F0 column."
    )


# =====================================

# 12. BLACK-SCHOLES WITH YIELD CURVE

# =====================================

def bs_call(

    S0,

    K,

    T,

    q,

    sigma

):

    if T <= 0:

        return np.nan


    DF = float(

        discount_factor(T)

    )

    r_T = float(

        zero_coupon_rate(T)

    )


    if sigma <= 0:

        return max(

            S0 * np.exp(-q * T)

            - K * DF,

            0.0

        )


    d1 = (

        np.log(S0 / K)

        + (

            r_T

            - q

            + 0.5 * sigma**2

        ) * T

    ) / (

        sigma * np.sqrt(T)

    )


    d2 = (

        d1

        - sigma * np.sqrt(T)

    )


    return (

        S0

        * np.exp(-q * T)

        * norm.cdf(d1)

        - K

        * DF

        * norm.cdf(d2)

    )


def bs_put(

    S0,

    K,

    T,

    q,

    sigma

):

    if T <= 0:

        return np.nan


    DF = float(

        discount_factor(T)

    )

    r_T = float(

        zero_coupon_rate(T)

    )


    if sigma <= 0:

        return max(

            K * DF

            - S0 * np.exp(-q * T),

            0.0

        )


    d1 = (

        np.log(S0 / K)

        + (

            r_T

            - q

            + 0.5 * sigma**2

        ) * T

    ) / (

        sigma * np.sqrt(T)

    )


    d2 = (

        d1

        - sigma * np.sqrt(T)

    )


    return (

        K

        * DF

        * norm.cdf(-d2)

        - S0

        * np.exp(-q * T)

        * norm.cdf(-d1)

    )


# ===================================

# 13. IMPLIED VOLATILITY INVERSION

# ===================================

def implied_volatility(

    market_price,

    option_type,

    S0,

    K,

    T,

    q,

    sigma_min=1e-8,

    sigma_max=5.0

):

    if (

        not np.isfinite(market_price)

        or market_price <= 0

        or not np.isfinite(T)

        or T <= 0

    ):

        return np.nan


    DF = float(

        discount_factor(T)

    )


    discounted_spot = (

        S0

        * np.exp(-q * T)

    )


    if option_type == "Call":

        lower_bound = max(

            discounted_spot

            - K * DF,

            0.0

        )

        upper_bound = (

            discounted_spot

        )

        pricing_function = bs_call


    elif option_type == "Put":

        lower_bound = max(

            K * DF

            - discounted_spot,

            0.0

        )

        upper_bound = (

            K * DF

        )

        pricing_function = bs_put


    else:

        return np.nan


    # Preliminary check of static bounds

    if (

        market_price < lower_bound - 1e-10

        or market_price > upper_bound + 1e-10

    ):

        return np.nan


    def objective(sigma):

        return (

            pricing_function(

                S0,

                K,

                T,

                q,

                sigma

            )

            - market_price

        )


    try:

        f_min = objective(

            sigma_min

        )

        f_max = objective(

            sigma_max

        )


        if (

            not np.isfinite(f_min)

            or not np.isfinite(f_max)

            or f_min * f_max > 0

        ):

            return np.nan


        return brentq(

            objective,

            sigma_min,

            sigma_max,

            xtol=1e-12,

            rtol=1e-12,

            maxiter=200

        )


    except (

        ValueError,

        RuntimeError,

        OverflowError

    ):

        return np.nan


# ============================================

# 14. MARKET IMPLIED VOLATILITY CALCULATION

# ============================================

df_calibration["IV_market"] = [

    implied_volatility(

        market_price=row["Market_price"],

        option_type=row["Option_type"],

        S0=S0,

        K=row["Strike"],

        T=row["T"],

        q=q

    )

    for _, row in df_calibration.iterrows()

]


valid_iv = (

    np.isfinite(

        df_calibration["IV_market"]

    )

    & (

        df_calibration["IV_market"] > 0

    )

)


print()

print("============================================================")

print("BLACK-SCHOLES INVERSION")

print("GSW / SVENSSON CURVE")

print("============================================================")

print(

    "Successfully computed IVs:",

    int(np.sum(valid_iv)),

    "/",

    len(df_calibration)

)


df_calibration = (

    df_calibration.loc[

        valid_iv

    ]

    .copy()

    .reset_index(drop=True)

)


# =================================

# 15. PRICE RECONSTRUCTION CHECK

# =================================

reconstructed_prices = []


for _, row in df_calibration.iterrows():

    if row["Option_type"] == "Call":

        price = bs_call(

            S0=S0,

            K=row["Strike"],

            T=row["T"],

            q=q,

            sigma=row["IV_market"]

        )

    else:

        price = bs_put(

            S0=S0,

            K=row["Strike"],

            T=row["T"],

            q=q,

            sigma=row["IV_market"]

        )


    reconstructed_prices.append(

        price

    )


df_calibration[

    "BS_reconstructed_price"

] = reconstructed_prices


df_calibration[

    "BS_reconstruction_error"

] = np.abs(

    df_calibration["BS_reconstructed_price"]

    - df_calibration["Market_price"]

)


print()

print("============================================================")

print("INVERSION CHECK")

print("============================================================")

print(

    "Maximum price error:",

    df_calibration[

        "BS_reconstruction_error"

    ].max()

)


# =====================================================

# 16. COMPARISON WITH THE IV PROVIDED BY THE DATASET

# =====================================================

if "IV_BDD" in df_calibration.columns:

    mask_bdd = (

        np.isfinite(

            df_calibration["IV_BDD"]

        )

        & (

            df_calibration["IV_BDD"] > 0

        )

    )


    df_calibration[

        "IV_BDD_error"

    ] = np.nan


    df_calibration.loc[

        mask_bdd,

        "IV_BDD_error"

    ] = np.abs(

        df_calibration.loc[

            mask_bdd,

            "IV_market"

        ]

        - df_calibration.loc[

            mask_bdd,

            "IV_BDD"

        ]

    )


    print()

    print("============================================================")

    print("COMPARISON WITH THE IV PROVIDED BY THE DATASET")

    print("============================================================")

    print(

        "Number of comparisons:",

        int(np.sum(mask_bdd))

    )

    print(

        "Mean absolute difference:",

        df_calibration.loc[

            mask_bdd,

            "IV_BDD_error"

        ].mean()

    )

    print(

        "Median absolute difference:",

        df_calibration.loc[

            mask_bdd,

            "IV_BDD_error"

        ].median()

    )


# ====================

# 17. FINAL SUMMARY

# ====================

print()

print("============================================================")

print("FINAL CALIBRATION DATASET")

print("============================================================")

print(

    "Number of options:",

    len(df_calibration)

)

print(

    "Number of calendar horizons:",

    df_calibration["T_days"].nunique()

)

print(

    "Number of effective maturities T:",

    df_calibration["T"].nunique()

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


print(
    "Moneyness used: K/F0(T)"
)

print(
    "K/F0(T) min/max in the sample:",
    f"{df_calibration['K_over_F0'].min():.6f}",
    "/",
    f"{df_calibration['K_over_F0'].max():.6f}"
)


final_summary = (

    df_calibration

    .groupby(

        [

            "T_days",

            "Root",

            "T"

        ],

        as_index=False

    )

    .size()

    .rename(

        columns={

            "size": "Nombre_options"

        }

    )

)


print()

print(

    final_summary.to_string(

        index=False

    )

)


# ===============================
# 18. FIGURE: SELECTED OPTIONS
# ===============================

plt.figure(

    figsize=(10, 6)

)


for root, marker in [

    ("SPX", "o"),

    ("SPXW", "x")

]:

    subset = df_calibration[

        df_calibration["Root"] == root

    ]

    plt.scatter(

        subset["T"],

        subset["Strike"],

        s=30,

        marker=marker,

        label=root

    )


plt.axhline(

    S0,

    linestyle="--",

    label=r"$S_0$"

)

forward_curve = (
    df_calibration[
        ["T", "Forward"]
    ]
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

    "Effective maturity T (years)"

)

plt.ylabel(

    "Strike K"

)

plt.title(

    "Options selected for calibration"

)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    figure_path,

    dpi=300,

    bbox_inches="tight"

)

plt.show()



# =============
# 19. EXPORT
# =============

df_calibration.to_csv(

    output_path,

    index=False

)

final_summary.to_csv(

    summary_path,

    index=False

)


print()
print("============================================================")
print("EXPORT")
print("============================================================")
print(
    "Final dataset saved to:"
)
print(
    output_path
)
print()
print(
    "Summary by horizon / family saved to:"
)
print(
    summary_path
)
print()
print(
    "Figure saved to:"
)
print(
    figure_path
)
