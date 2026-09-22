"""
Black-Scholes framework: Implied volatility smile and surface

This script illustrates key empirical features of implied volatility,
including the volatility smile, negative skew, and term structure.

A synthetic implied volatility surface is generated and calibrated using
nonlinear least squares, then visualized across strikes and maturities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pathlib import Path



# ================
# PROJECT PATHS
# ================

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

FIGURES_DIR = ROOT_DIR / "01_black_scholes" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ===============
# PARAMETERS
# ===============

S0 = 100

strikes = np.linspace(70, 130, 300)
maturities = np.linspace(0.1, 2.0, 80)

K_grid, T_grid = np.meshgrid(strikes, maturities)

k_grid = (K_grid - S0) / S0


# ==========================
# SYNTHETIC MARKET DATA
# ==========================

sigma_market = (
    0.20
    - 0.18 * k_grid
    + 0.55 * k_grid**2
    + 0.05 * np.exp(-1.5 * T_grid)
)


# =============
# CALIBRATION
# =============
def vol_model(params, K, T):

    a, b, c, d = params

    k = (K - S0) / S0

    return (
        a
        + b * k
        + c * k**2
        + d * np.exp(-1.5 * T)
    )


def residuals(params):

    sigma_model = vol_model(
        params,
        K_grid,
        T_grid
    )

    return (
        sigma_model - sigma_market
    ).ravel()


initial_guess = [
    0.18,
    -0.10,
    0.40,
    0.03
]


result = least_squares(
    residuals,
    initial_guess
)


a, b, c, d = result.x


print("Calibrated parameters:")
print(f"a = {a:.4f}")
print(f"b = {b:.4f}")
print(f"c = {c:.4f}")
print(f"d = {d:.4f}")


sigma_calibrated = vol_model(
    result.x,
    K_grid,
    T_grid
)


# ====================================
# FIGURE 1: IMPLIED VOLATILITY SMILE
# ====================================

k = (strikes - S0) / S0

smile = (
    0.20
    + 1.30 * k**2
)


plt.figure(figsize=(8, 5))

plt.plot(
    strikes,
    100 * smile,
    linewidth=2
)

plt.axvline(
    S0,
    linestyle="--",
    linewidth=1.3,
    label=r"At-the-money: $K=S_0$"
)

plt.xlabel("Strike $K$")
plt.ylabel(
    r"Implied volatility $\sigma_{\mathrm{impl}}$ (%)"
)

plt.title(
    "Implied volatility smile"
)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "1_implied_volatility_smile.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ========================================
# FIGURE 2: IMPLIED VOLATILITY SURFACE
# ========================================

fig = plt.figure(
    figsize=(10, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


surface = ax.plot_surface(
    K_grid,
    T_grid,
    100 * sigma_calibrated,
    cmap="viridis",
    edgecolor="none"
)


ax.set_xlabel(
    "Strike $K$"
)

ax.set_ylabel(
    "Maturity $T$ (years)"
)

ax.set_zlabel(
    "Implied volatility (%)"
)

ax.set_title(
    "Implied volatility surface"
)


fig.colorbar(
    surface,
    shrink=0.6,
    label="Implied volatility (%)"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "2_implied_volatility_surface.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =============================================
# FIGURE 3: SYMMETRIC SMILE AND NEGATIVE SKEW
# =============================================

skew = (
    0.20
    * np.exp(-2.85 * k)
)


plt.figure(figsize=(8, 5))

plt.plot(
    strikes,
    100 * smile,
    linestyle="--",
    linewidth=2,
    label="Symmetric smile"
)

plt.plot(
    strikes,
    100 * skew,
    linewidth=2,
    label="Negative skew"
)

plt.axvline(
    S0,
    linestyle=":",
    linewidth=1.3,
    label=r"At-the-money: $K=S_0$"
)

plt.xlabel(
    "Strike $K$"
)

plt.ylabel(
    r"Implied volatility $\sigma_{\mathrm{impl}}$ (%)"
)

plt.title(
    "Symmetric smile vs Negative volatility skew"
)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "3_symmetric_smile_vs_negative_skew.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()