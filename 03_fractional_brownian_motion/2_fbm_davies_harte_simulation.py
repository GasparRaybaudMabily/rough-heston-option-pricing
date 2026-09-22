"""
Fractional Brownian Motion: Davies-Harte Simulation

This script simulates fractional Brownian motion paths using the
Davies-Harte method. The algorithm relies on a circulant embedding
of the fractional Gaussian noise covariance and uses the Fast Fourier
Transform to efficiently generate trajectories for different Hurst parameters.
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

dt = T / n


# ==========================================
# FRACTIONAL GAUSSIAN NOISE AUTOCOVARIANCE
# ==========================================

def gamma_fgn(k, H, dt):
    """
    Autocovariance function of fractional Brownian motion increments.

    gamma(k)
    = 1/2 * ( |k+1|^(2H)
            + |k-1|^(2H)
            - 2|k|^(2H) )
      * dt^(2H)
    """

    return 0.5 * (
        np.abs(k + 1) ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
    ) * dt ** (2 * H)


# =========================
# DAVIES-HARTE SIMULATION
# =========================

def simulate_fbm_davies_harte(n, T, H):
    """
    Simulates a fractional Brownian motion path
    using the Davies-Harte method.
    """

    dt = T / n


    # 1. Fractional Gaussian noise covariance
    gamma = np.array([
        gamma_fgn(k, H, dt)
        for k in range(n)
    ])


    # 2. Circulant vector construction
    c = np.concatenate([
        gamma,
        [0.0],
        gamma[1:][::-1]
    ])


    # 3. Eigenvalues via FFT
    eigenvalues = np.real(np.fft.fft(c))

    # Remove tiny negative eigenvalues caused by numerical errors
    eigenvalues[eigenvalues < 0] = 0.0


    # 4. Complex Gaussian variables

    m = 2 * n

    Z = np.zeros(m, dtype=complex)

    # Zero frequency
    Z[0] = np.random.normal()

    # Nyquist frequency
    Z[n] = np.random.normal()

    # Intermediate frequencies
    for k in range(1, n):

        a = np.random.normal()
        b = np.random.normal()

        Z[k] = (a + 1j * b) / np.sqrt(2)

        # Conjugate symmetry
        Z[m - k] = np.conj(Z[k])


    # 5. Eigenvalue weighting

    W = np.sqrt(eigenvalues) * Z


    # 6. Inverse FFT

    fgn = np.fft.ifft(W).real * np.sqrt(m)

    # Keep the first n increments
    fgn = fgn[:n]


    # 7. Fractional Brownian motion construction

    fbm = np.concatenate([
        [0.0],
        np.cumsum(fgn)
    ])

    return fbm


# ====================================
# SIMULATION ACROSS HURST PARAMETERS
# ====================================

plt.figure(figsize=(11, 7))

for H in H_values:

    B_H = simulate_fbm_davies_harte(
        n=n,
        T=T,
        H=H
    )

    plt.plot(
        t,
        B_H,
        label=f"H = {H}"
    )


plt.xlabel("Time", fontsize=14)

plt.ylabel(
    r"$B_t^H$",
    fontsize=14
)

plt.title(
    "Fractional Brownian motion paths\n"
    "using the Davies-Harte method",
    fontsize=14
)

plt.legend(fontsize=12)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "2_fbm_davies_harte_paths.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)