"""
Heston Model: Implied volatility surface and smiles

This script prices European call options under the Heston stochastic
volatility model using its semi-analytical characteristic function.
Black-Scholes implied volatilities are then recovered from Heston prices
to construct and visualize the implied volatility surface and smiles
across different strikes and maturities.
"""


import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from pathlib import Path


# =================
# PROJECT PATHS
# =================

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

FIGURES_DIR = ROOT_DIR / "02_heston" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. MODEL PARAMETERS
# ============================================================

# Initial spot price
S0 = 100.0

# Risk-free rate
r = 0.02

# Dividend yield
q = 0.00

# Initial variance -> initial volatility = 20%
v0 = 0.04

# Mean-reversion speed
kappa = 1.2

# Long-run variance
theta = 0.05

# Vol-of-vol
xi = 0.90

# Price / variance correlation
rho = -0.85


# ============================================================
# 2. BLACK-SCHOLES
# ============================================================

def black_scholes_call(S, K, T, r, q, sigma):
    """
    European call price under the Black-Scholes model.
    """
    # At maturity, the option value equals its payoff.
    # For T < 0, the option is already expired, so the payoff is used
    # to avoid numerical issues.
    if T <= 0:
        return max(S - K, 0.0)

    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    d1 = (
        np.log(S / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call = (
        S * np.exp(-q * T) * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    return call


# ============================================================
# 3. HESTON CHARACTERISTIC FUNCTION
# ============================================================

def heston_characteristic_function(
    phi,
    T,
    S0,
    r,
    q,
    v0,
    kappa,
    theta,
    xi,
    rho,
    j
):
    """
    Characteristic function used in the semi-analytical Heston formulation.

    j = 1 or 2.
    """

    x = np.log(S0)

    if j == 1:
        u = 0.5
        b = kappa - rho * xi
    else:
        u = -0.5
        b = kappa

    a = kappa * theta

    i = 1j

    d = np.sqrt(
        (rho * xi * i * phi - b)**2
        - xi**2 * (2 * u * i * phi - phi**2)
    )

    g = (
        b - rho * xi * i * phi + d
    ) / (
        b - rho * xi * i * phi - d
    )

    # Little Heston Trap formulation for improved numerical stability
    c = 1.0 / g

    exp_minus_dT = np.exp(-d * T)

    C = (
        (r - q) * i * phi * T
        + (a / xi**2)
        * (
            (b - rho * xi * i * phi - d) * T
            - 2.0
            * np.log(
                (1.0 - c * exp_minus_dT)
                / (1.0 - c)
            )
        )
    )

    D = (
        (b - rho * xi * i * phi - d) / xi**2
        * (
            (1.0 - exp_minus_dT)
            / (1.0 - c * exp_minus_dT)
        )
    )

    return np.exp(
        C
        + D * v0
        + i * phi * x
    )


# ============================================================
# 4. PROBABILITIES P1 AND P2
# ===========================================================

def heston_probability(
    j,
    S0,
    K,
    T,
    r,
    q,
    v0,
    kappa,
    theta,
    xi,
    rho
):
    """
    Computes P1 or P2 in the Heston pricing formula.
    """

    logK = np.log(K)

    def integrand(phi):

        f = heston_characteristic_function(
            phi=phi,
            T=T,
            S0=S0,
            r=r,
            q=q,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            j=j
        )

        value = (
            np.exp(-1j * phi * logK)
            * f
            / (1j * phi)
        )

        return np.real(value)

    integral, error = quad(
        integrand,
        1e-8,
        # we could increase the upper integration bound
        # to obtain smoother smiles
        100.0,
        limit=250,
        epsabs=1e-7,
        epsrel=1e-7
    )

    return 0.5 + integral / np.pi


# ============================================================
# 5. HESTON CALL PRICE
# ============================================================

def heston_call_price(
    S0,
    K,
    T,
    r,
    q,
    v0,
    kappa,
    theta,
    xi,
    rho
):
    """
    European call price under the Heston model.
    """

    P1 = heston_probability(
        1,
        S0,
        K,
        T,
        r,
        q,
        v0,
        kappa,
        theta,
        xi,
        rho
    )

    P2 = heston_probability(
        2,
        S0,
        K,
        T,
        r,
        q,
        v0,
        kappa,
        theta,
        xi,
        rho
    )

    call = (
        S0 * np.exp(-q * T) * P1
        - K * np.exp(-r * T) * P2
    )

    return call


# ============================================================
# 6. IMPLIED VOLATILITY
# ============================================================

def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    q
):
    """
    Returns the Black-Scholes implied volatility corresponding
    to the supplied option price.
    """
    # No-arbitrage bounds
    intrinsic = max(
        S * np.exp(-q * T)
        - K * np.exp(-r * T),
        0.0
    )

    upper_bound = S * np.exp(-q * T)

    # Reject prices that violate no-arbitrage bounds
    if market_price <= intrinsic:
        return np.nan

    if market_price >= upper_bound:
        return np.nan

    def objective(sigma):

        return (
            black_scholes_call(
                S=S,
                K=K,
                T=T,
                r=r,
                q=q,
                sigma=sigma
            )
            - market_price
        )

    try:
        # Brent's method is used for robust root finding
        vol = brentq(
            objective,
            1e-6,
            5.0,
            maxiter=200
        )

        return vol

    except ValueError:
        return np.nan


# ============================================================
# 7. STRIKE / MATURITY GRID
# ============================================================

strikes = np.linspace(
    60,
    140,
    41
)

# We could also use np.linspace for maturities,
# but explicit maturities are kept here for simplicity.
maturities = np.array([
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    3.00
])


# Mesh grids required for the volatility surface
K_grid, T_grid = np.meshgrid(
    strikes,
    maturities
)

IV_grid = np.zeros_like(
    K_grid,
    dtype=float
)


# ============================================================
# 8. IMPLIED VOLATILITY SURFACE COMPUTATION
# ============================================================

for i in range(len(maturities)):

    T = maturities[i]

    print(
        f"Computing maturity T = {T:.2f}"
    )

    for j in range(len(strikes)):

        K = strikes[j]

        # Heston option price
        heston_price = heston_call_price(
            S0=S0,
            K=K,
            T=T,
            r=r,
            q=q,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho
        )

        # Corresponding implied volatility
        IV_grid[i, j] = implied_volatility(
            market_price=heston_price,
            S=S0,
            K=K,
            T=T,
            r=r,
            q=q
        )


# ============================================================
# 9. 3D SURFACE
# ============================================================

fig = plt.figure(
    figsize=(12, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

surface = ax.plot_surface(
    K_grid,
    T_grid,
    IV_grid,
    cmap="viridis",
    edgecolor="none",
    alpha=0.9,
    vmin=0.10,
    vmax=0.45
)

ax.set_xlabel(
    "Strike $K$",
    fontsize=12
)

ax.set_ylabel(
    "Maturity $T$",
    fontsize=12
)

ax.set_zlabel(
    "Implied volatility",
    fontsize=12
)

ax.set_title(
    "Heston implied volatility surface"
)


fig.colorbar(
    surface,
    shrink=0.6,
    aspect=12,
    label="Implied volatility"
)

plt.tight_layout()

# Save figure
fig.savefig(
    FIGURES_DIR / "1_heston_implied_volatility_surface.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. SMILES ACROSS MATURITIES
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(10, 6)
)

for i, T in enumerate(maturities):

    ax2.plot(
        strikes,
        IV_grid[i, :],
        label=f"T = {T:.2f}"
    )

ax2.set_xlabel("Strike $K$")
ax2.set_ylabel("Implied volatility")
ax2.set_title("Implied volatility smiles under Heston")
ax2.legend()
ax2.grid(alpha=0.3)

fig2.tight_layout()

# Save figure
fig2.savefig(
    FIGURES_DIR / "2_heston_implied_volatility_smiles.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()