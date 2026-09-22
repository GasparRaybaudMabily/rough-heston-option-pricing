"""
Fractional Brownian Motion: Hurst Parameter Estimation

This script estimates the Hurst parameter from a simulated fractional
Brownian motion path. The estimation relies on the scaling behavior of
the second moment of increments and a log-log linear regression, from
which the Hurst parameter is recovered from the estimated slope.
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


# =======================================
# 1. FBM SIMULATION USING DAVIES-HARTE
# =======================================

def gamma_fgn(k, H, dt):
    """
    Autocovariance of fractional Gaussian noise.
    """
    return 0.5 * (
        np.abs(k + 1) ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
    ) * dt ** (2 * H)


def simulate_fbm_davies_harte(n, T, H):
    """
    Simulates a fractional Brownian motion path on [0, T]
    with n increments using the Davies-Harte method.
    """
    dt = T / n

    gamma = np.array([
        gamma_fgn(k, H, dt)
        for k in range(n)
    ])

    # Circulant covariance vector
    c = np.concatenate([
        gamma,
        [0.0],
        gamma[1:][::-1]
    ])

    # Eigenvalues via FFT
    eigenvalues = np.real(np.fft.fft(c))

    # Correct small negative eigenvalues caused by numerical errors
    eigenvalues[eigenvalues < 0] = 0.0

    m = 2 * n

    # Complex Gaussian variables
    Z = np.zeros(m, dtype=complex)

    Z[0] = np.random.normal()
    Z[n] = np.random.normal()

    for k in range(1, n):
        a = np.random.normal()
        b = np.random.normal()

        Z[k] = (a + 1j * b) / np.sqrt(2)
        Z[m - k] = np.conj(Z[k])

    # Spectral weighting
    W = np.sqrt(eigenvalues) * Z

    # Transform back to the time domain
    fgn = np.fft.ifft(W).real * np.sqrt(m)

    # Keep the first n increments
    fgn = fgn[:n]

    # Construct the fractional Brownian motion path
    fbm = np.concatenate([
        [0.0],
        np.cumsum(fgn)
    ])

    return fbm


# =======================================
# 2. EMPIRICAL COMPUTATION OF M2(DELTA)
# =======================================

def empirical_m2(path, lags):
    """
    Computes the empirical second moment m2(Delta)
    for multiple time lags.
    """
    m2_values = []

    for lag in lags:
        increments = path[lag:] - path[:-lag]

        m2 = np.mean(increments ** 2)

        m2_values.append(m2)

    return np.array(m2_values)


# ===========================
# 3. EXPERIMENT PARAMETERS
# ===========================

np.random.seed(42)

H_true = 0.1

T = 1.0
n = 10000

dt = T / n

# Simulation
B_H = simulate_fbm_davies_harte(
    n=n,
    T=T,
    H=H_true
)


# ==========================
# 4. TIME-SCALE SELECTION
# ==========================

# Lags expressed in number of time steps
lags = np.array([
    1, 2, 4, 8, 16, 32, 64, 128, 256
])

# Convert lags into time scales Delta
deltas = lags * dt


# =============================
# 5. COMPUTATION OF M2(DELTA)
# =============================

m2_values = empirical_m2(
    B_H,
    lags
)


# =======================
# 6. LOG-LOG REGRESSION
# =======================

log_delta = np.log(deltas)
log_m2 = np.log(m2_values)

# Regression:
# log(m2) = a + beta * log(Delta)

beta, intercept = np.polyfit(
    log_delta,
    log_m2,
    1
)

H_estimated = beta / 2


# =============
# 7. RESULTS
# =============

print("--------------------------------")
print("Estimation results")
print("--------------------------------")

print(f"True H       = {H_true:.4f}")
print(f"Slope beta   = {beta:.4f}")
print(f"Estimated H  = {H_estimated:.4f}")


# ====================
# 8. REGRESSION LINE
# ====================

regression_line = (
    intercept
    + beta * log_delta
)


# =================
# 9. LOG-LOG PLOT
# =================

plt.figure(figsize=(9, 7))

plt.scatter(
    log_delta,
    log_m2,
    label="Empirical moments"
)

plt.plot(
    log_delta,
    regression_line,
    label=(
    f"Regression: slope = {beta:.3f}\n"
    f"Estimated H = {H_estimated:.3f}"
    )
)

plt.xlabel(
    r"$\log(\Delta)$",
    fontsize=14
)

plt.ylabel(
    r"$\log(\widehat{m}_2(\Delta))$",
    fontsize=14
)

plt.title(
    "Hurst parameter estimation\n"
    "from the scaling law of increments",
    fontsize=14
)

plt.legend(fontsize=11)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "3_hurst_parameter_estimation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)