"""
Fractional Brownian Motion: Cholesky Simulation

This script simulates fractional Brownian motion paths using the
Cholesky decomposition of the covariance matrix. It illustrates the
effect of different Hurst parameters on the regularity and persistence
of fractional Brownian motion trajectories.
"""


import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# =================
# PROJECT PATHS
# =================

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

FIGURES_DIR = (
    ROOT_DIR
    / "03_fractional_brownian_motion"
    / "figures"
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ===================
# GENERAL PARAMETERS
# ===================

T = 1.0
n = 500

H_values = [0.1, 0.3, 0.5, 0.7]

# Set seed for reproducible simulations
np.random.seed(42)

# Time grid
t = np.linspace(0, T, n + 1)


# ===============================================
# FRACTIONAL BROWNIAN MOTION COVARIANCE MATRIX
# ===============================================

def covariance_fbm(t, H):
    """
    Builds the covariance matrix of fractional Brownian motion:

    Cov(B_t^H, B_s^H)
    = 1/2 * (t^(2H) + s^(2H) - |t-s|^(2H))
    """

    times = t[1:]  # Exclude t = 0 since B_0^H = 0

    ti = times[:, None] # Column vector ti
    tj = times[None, :] # Row vector tj

    covariance = 0.5 * (
        ti ** (2 * H)
        + tj ** (2 * H)
        - np.abs(ti - tj) ** (2 * H)
    )

    return covariance


# =====================
# CHOLESKY SIMULATION
# =====================

def simulate_fbm_cholesky(t, H):
    """
    Simulates a fractional Brownian motion path with Hurst parameter H
    using Cholesky decomposition.
    """

    covariance = covariance_fbm(t, H)

    # Small numerical regularization term to avoid
    # floating-point issues
    epsilon = 1e-10
    covariance += epsilon * np.eye(len(covariance))

    # Cholesky decomposition
    L = np.linalg.cholesky(covariance)

    # Independent standard normal variables
    Z = np.random.normal(size=len(covariance))

    # Gaussian vector with the target covariance structure
    B = L @ Z

    # B_0^H = 0
    B = np.concatenate(([0.0], B))

    return B


# ==============================
# SIMULATION AND VISUALIZATION
# ==============================

plt.figure(figsize=(11, 7))

for H in H_values:

    B_H = simulate_fbm_cholesky(t, H)

    plt.plot(
        t,
        B_H,
        label=f"H = {H}"
    )


plt.xlabel("Time", fontsize=14)
plt.ylabel(r"$B_t^H$", fontsize=14)

plt.title(
    "Fractional Brownian motion paths\n"
    "for different Hurst parameters",
    fontsize=14
)

plt.legend(fontsize=12)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "1_fbm_cholesky_paths.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)