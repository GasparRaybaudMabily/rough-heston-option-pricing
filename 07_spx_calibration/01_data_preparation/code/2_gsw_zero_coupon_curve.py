"""
GSW / Svensson zero-coupon yield curve.

This script reconstructs the U.S. Treasury zero-coupon curve from the
Gürkaynak, Sack and Wright parameters observed on September 13, 2022,
validates it against published SVENY rates, and exports the resulting curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========
# 1. PATHS
# ==========

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

GSW_FILE = RAW_DIR / "feds200628.csv"

FIGURE_PATH = (
    FIGURES_DIR
    / "02_gsw_zero_coupon_curve.png"
)

CSV_PATH = (
    CSV_DIR
    / "03_gsw_zero_coupon_curve.csv"
)

# ============================================================
# 2. GSW / SVENSSON PARAMETERS ON SEPTEMBER 13, 2022
# ============================================================

# Source:
# Board of Governors of the Federal Reserve System
# Gürkaynak, Sack and Wright (GSW)
# U.S. Treasury yield curve
#
# File: feds200628.csv
# Date: 2022-09-13
#
# SVENYXX rates are continuously compounded zero-coupon yields,
# expressed as percentages.

BETA0 = 4.1325311721141
BETA1 = -0.483330087369049
BETA2 = 249.569929179392
BETA3 = -250.639461221969
TAU1 = 2.72637117177024
TAU2 = 2.74854195681383


# =========================================
# 3. SVENSSON FUNCTION: ZERO-COUPON YIELD
# =========================================

def zero_coupon_rate_percent(T):
    """
    GSW/Svensson zero-coupon rate at maturity T.

    Parameter
    ---------
    T : float or array_like
        Maturity in years.

    Returns
    -------
    z(T) as an annual percentage rate, continuously compounded.
    """

    T = np.asarray(T, dtype=float)

    if np.any(T <= 0.0):
        raise ValueError("Maturities T must be strictly positive.")

    x1 = T / TAU1
    x2 = T / TAU2

    term1 = (1.0 - np.exp(-x1)) / x1
    term2 = term1 - np.exp(-x1)
    term3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)

    z = (
        BETA0
        + BETA1 * term1
        + BETA2 * term2
        + BETA3 * term3
    )

    return z


def zero_coupon_rate(T):
    """
    Same rate as zero_coupon_rate_percent(T),
    but returned as a decimal value.
    """
    return zero_coupon_rate_percent(T) / 100.0


def discount_factor(T):
    """
    Discount factor associated with the GSW curve:
        P(0,T) = exp(-z(T) T)

    with z(T) as a continuously compounded decimal rate.
    """
    T = np.asarray(T, dtype=float)
    return np.exp(-zero_coupon_rate(T) * T)


# ==========================================
# 4. OFFICIAL SVENY POINTS FOR VALIDATION
# ==========================================

# The Fed file directly provides zero-coupon rates
# at integer maturities from 1 to 30 years.
# Maturities from 1 to 10 years are used here to verify that the
# Svensson formula reconstructed from the GSW parameters
# matches the published values.

official_maturities = np.arange(
    1.0,
    11.0,
    1.0,
)

official_sveny = np.array(
    [
        3.7966,
        3.7974,
        3.7356,
        3.6567,
        3.5831,
        3.5246,
        3.4837,
        3.4595,
        3.4495,
        3.4507,
    ],
    dtype=float,
)

reconstructed_sveny = zero_coupon_rate_percent(
    official_maturities
)

absolute_errors = np.abs(
    reconstructed_sveny - official_sveny
)


# =======================
# 5. PARAMETER DISPLAY
# =======================

print()
print("=====================================")
print("GSW / SVENSSON ZERO-COUPON CURVE")
print("DATE: SEPTEMBER 13, 2022")
print("=====================================")
print()

print(f"BETA0 = {BETA0:.15f}")
print(f"BETA1 = {BETA1:.15f}")
print(f"BETA2 = {BETA2:.15f}")
print(f"BETA3 = {BETA3:.15f}")
print(f"TAU1  = {TAU1:.15f}")
print(f"TAU2  = {TAU2:.15f}")

print()
print("========================================================")
print("VALIDATION AGAINST SVENY RATES PUBLISHED BY THE FED")
print("========================================================")
print()

for T, official, reconstructed, error in zip(
    official_maturities,
    official_sveny,
    reconstructed_sveny,
    absolute_errors,
):
    print(
        f"T = {T:>4.1f} year"
        f" | Fed = {official:.4f} %"
        f" | reconstructed = {reconstructed:.6f} %"
        f" | error = {error:.6e}"
    )

print()
print(
    "Maximum absolute reconstruction error: "
    f"{absolute_errors.max():.6e} percentage point(s)"
)


# ================
# 6. CURVE GRID
# ================

# Start from a maturity very close to zero in order to display
# the very short end of the curve as well.
#
# The 10-year upper bound is NOT related to the filtering of the
# option sample: it is only used to display a sufficiently broad
# portion of the term structure.

T_grid = np.linspace(
    1.0 / 365.25,
    10.0,
    2000,
)

z_grid_percent = zero_coupon_rate_percent(
    T_grid
)


# =============
# 7. FIGURE
# =============

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    T_grid,
    z_grid_percent,
    linewidth=2.0,
    label="GSW / Svensson zero-coupon curve",
)

ax.scatter(
    official_maturities,
    official_sveny,
    s=45,
    zorder=3,
    label="SVENY rates published by the Fed",
)

ax.set_title(
    "Courbe des taux zéro-coupon du Trésor américain "
    "au 13 septembre 2022"
)

ax.set_xlabel(
    "Maturity $T$ (years)"
)

ax.set_ylabel(
    "Continuously compounded zero-coupon rate (%)"
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# =====================
# 8. CURVE CSV EXPORT
# =====================

discount_grid = discount_factor(
    T_grid
)

curve_data = np.column_stack(
    [
        T_grid,
        z_grid_percent / 100.0,
        z_grid_percent,
        discount_grid,
    ]
)

header = (
    "Maturity_years,"
    "Zero_coupon_rate_decimal,"
    "Zero_coupon_rate_percent,"
    "Discount_factor"
)

np.savetxt(
    CSV_PATH,
    curve_data,
    delimiter=",",
    header=header,
    comments="",
    fmt="%.12f",
)


# ===================================
# 9. EXAMPLES FOR SHORT MATURITIES
# ===================================

test_days = np.array(
    [
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
        647,
    ],
    dtype=float,
)

test_T = test_days / 365.25

test_rates_percent = zero_coupon_rate_percent(
    test_T
)

test_discount_factors = discount_factor(
    test_T
)

print()
print("============================================================")
print("ZERO-COUPON RATE EXAMPLES FOR SHORT MATURITIES")
print("============================================================")
print()

for days, T, rate, df in zip(
    test_days,
    test_T,
    test_rates_percent,
    test_discount_factors,
):
    print(
        f"{int(days):>3d} days"
        f" | T = {T:.8f}"
        f" | z(T) = {rate:.4f} %"
        f" | P(0,T) = {df:.8f}"
    )


# ==========
# 10. END
# ==========

print()
print("Figure saved to:")
print(FIGURE_PATH)

print()
print("CSV saved to:")
print(CSV_PATH)
