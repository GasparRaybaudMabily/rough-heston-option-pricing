"""
Rough Heston: Implied volatility smile and ATM Skew

This script prices European call options under the Rough Heston model
using the COS method, converts model prices into Black-Scholes implied
volatilities, and studies the resulting volatility smile. It also
estimates the ATM skew across maturities to analyze its term structure.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma
from scipy.stats import norm
from scipy.optimize import brentq

from pathlib import Path

# =================
# PROJECT PATHS
# =================

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

FIGURES_DIR = (
    ROOT_DIR
    / "06_fourier_pricing"
    / "figures"
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ================================
# 1. RICCATI QUADRATIC FUNCTION
# ================================

def riccati_F(phi, h, kappa, nu, rho):
    """
    Quadratic Riccati function associated with the Rough Heston model.
    """

    return (
        -0.5 * (phi**2 + 1j * phi)
        + (1j * rho * nu * phi - kappa) * h
        + 0.5 * nu**2 * h**2
    )


# =============================
# 2. VECTORIZED ADAMS SOLVER
# =============================

def solve_rough_riccati_adams_vectorized(
    phi_values,
    H,
    kappa,
    nu,
    rho,
    T,
    N
):
    """
    Solves the Riccati-Volterra equation simultaneously
    for multiple values of phi.
    """

    phi_values = np.atleast_1d(
        np.asarray(phi_values, dtype=complex)
    )

    M = len(phi_values)

    alpha = H + 0.5
    dt = T / N

    t = np.linspace(0.0, T, N + 1)

    h = np.zeros(
        (M, N + 1),
        dtype=complex
    )

    for n in range(N):

        j = np.arange(n + 1)


        # Adams-Bashforth predictor

        b = (
            (n + 1 - j)**alpha
            - (n - j)**alpha
        )

        F_history = riccati_F(
            phi_values[:, None],
            h[:, :n + 1],
            kappa,
            nu,
            rho
        )

        predictor_sum = F_history @ b

        h_pred = (
            dt**alpha
            / gamma(alpha + 1)
            * predictor_sum
        )


        # Adams-Moulton corrector

        a = np.empty(n + 1)

        # j = 0
        a[0] = (
            n**(alpha + 1)
            - (n - alpha) * (n + 1)**alpha
        )

        # j >= 1
        if n >= 1:

            jj = np.arange(1, n + 1)

            a[1:] = (
                (n - jj + 2)**(alpha + 1)
                + (n - jj)**(alpha + 1)
                - 2 * (n - jj + 1)**(alpha + 1)
            )

        corrector_sum = F_history @ a

        h[:, n + 1] = (
            dt**alpha
            / gamma(alpha + 2)
            * (
                riccati_F(
                    phi_values,
                    h_pred,
                    kappa,
                    nu,
                    rho
                )
                + corrector_sum
            )
        )

    return t, h


# =========================================
# 3. ROUGH HESTON CHARACTERISTIC FUNCTION
# =========================================

def rough_heston_characteristic(
    phi_values,
    V0,
    kappa,
    theta,
    nu,
    rho,
    H,
    r,
    T,
    N_time
):
    """
    Characteristic function of

        X_T = log(S_T / S0)

    under the Rough Heston model.
    """

    phi_values = np.atleast_1d(
        np.asarray(phi_values, dtype=complex)
    )

    alpha = H + 0.5

    t, h = solve_rough_riccati_adams_vectorized(
        phi_values=phi_values,
        H=H,
        kappa=kappa,
        nu=nu,
        rho=rho,
        T=T,
        N=N_time
    )

    dt = T / N_time

    # ===========================================
    # Classical integral of h: trapezoidal rule
    # ===========================================

    integral_h = dt * (
        0.5 * h[:, 0]
        + np.sum(h[:, 1:-1], axis=1)
        + 0.5 * h[:, -1]
    )

    # ========================================
    # Riemann-Liouville fractional integral
    # ========================================

    t_left = t[:-1]
    t_right = t[1:]

    fractional_weights = (
        (T - t_left)**(1.0 - alpha)
        - (T - t_right)**(1.0 - alpha)
    )

    fractional_integral = (
        h[:, :-1] @ fractional_weights
        / gamma(2.0 - alpha)
    )

    # ==========================
    # Characteristic function
    # ==========================

    exponent = (
        1j * phi_values * r * T
        + V0 * fractional_integral
        + kappa * theta * integral_h
    )

    return np.exp(exponent)


# ============================================
# 4. CUMULANT ESTIMATION FOR THE COS METHOD
# ============================================
def estimate_cumulants(
    V0,
    kappa,
    theta,
    nu,
    rho,
    H,
    r,
    T,
    N_time,
    radius=0.15,
    n_points=9
):
    """
    Numerically estimates c1, c2 and c4 from

        K(z) = log Phi_T(-i z).
    """

    z = np.linspace(
        -radius,
        radius,
        n_points
    )

    phi = -1j * z

    Phi = rough_heston_characteristic(
        phi_values=phi,
        V0=V0,
        kappa=kappa,
        theta=theta,
        nu=nu,
        rho=rho,
        H=H,
        r=r,
        T=T,
        N_time=N_time
    )

    K_values = np.real(
        np.log(Phi)
    )

    coeff = np.polynomial.polynomial.polyfit(
        z,
        K_values,
        deg=6
    )

    c1 = coeff[1]
    c2 = 2.0 * coeff[2]
    c4 = 24.0 * coeff[4]

    return c1, c2, c4


# ==============================================
# 5. CHI AND PSI FUNCTIONS FOR THE COS METHOD
# ==============================================

def chi_cos(j, a, b, c, d):

    u = j * np.pi / (b - a)

    term_d = (
        np.cos(u * (d - a))
        + u * np.sin(u * (d - a))
    )

    term_c = (
        np.cos(u * (c - a))
        + u * np.sin(u * (c - a))
    )

    return (
        np.exp(d) * term_d
        - np.exp(c) * term_c
    ) / (1.0 + u**2)


def psi_cos(j, a, b, c, d):

    u = j * np.pi / (b - a)

    if j == 0:
        return d - c

    return (
        np.sin(u * (d - a))
        - np.sin(u * (c - a))
    ) / u


# =========================================
# 6. ROUGH HESTON CALL PRICING WITH COS
# =========================================

def price_call_cos(
    S0,
    K,
    V0,
    kappa,
    theta,
    nu,
    rho,
    H,
    r,
    T,
    N_time,
    N_cos,
    L,
    cumulants
):
    """
    Prices a European call option under the Rough Heston model
    using the COS method.
    """

    c1, c2, c4 = cumulants

    c2 = max(
        float(np.real(c2)),
        1e-12
    )

    c4_positive = max(
        float(np.real(c4)),
        0.0
    )

    # ======================
    # Truncation interval
    # ======================

    width = L * np.sqrt(
        c2 + np.sqrt(c4_positive)
    )

    a = c1 - width
    b = c1 + width

    # ==============
    # Frequencies
    # ==============

    j_values = np.arange(N_cos)

    u = (
        j_values
        * np.pi
        / (b - a)
    )

    # ==========================
    # Characteristic function
    # ==========================

    Phi = rough_heston_characteristic(
        phi_values=u,
        V0=V0,
        kappa=kappa,
        theta=theta,
        nu=nu,
        rho=rho,
        H=H,
        r=r,
        T=T,
        N_time=N_time
    )

    # ==========
    # Payoff
    # ==========

    k = np.log(K / S0)

    c = max(k, a)
    d = b

    G = np.zeros(N_cos)

    if c < b:

        for j in range(N_cos):

            chi = chi_cos(
                j,
                a,
                b,
                c,
                d
            )

            psi = psi_cos(
                j,
                a,
                b,
                c,
                d
            )

            G[j] = (
                S0 * chi
                - K * psi
            )

    # ===================
    # COS coefficients
    # ===================

    density_coeff = np.real(
        Phi
        * np.exp(-1j * u * a)
    )

    weights = np.ones(N_cos)
    weights[0] = 0.5

    # ========
    # Price
    # ========

    price = (
        np.exp(-r * T)
        * 2.0
        / (b - a)
        * np.sum(
            weights
            * density_coeff
            * G
        )
    )

    return float(np.real(price))


# ==============================
# 7. BLACK-SCHOLES CALL PRICE
# ==============================

def black_scholes_call(
    S0,
    K,
    T,
    r,
    sigma
):
    """
    Black-Scholes European call price.
    """

    d1 = (
        np.log(S0 / K)
        + (r + 0.5 * sigma**2) * T
    ) / (
        sigma * np.sqrt(T)
    )

    d2 = (
        d1
        - sigma * np.sqrt(T)
    )

    return (
        S0 * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


# ========================
# 8. BLACK-SCHOLES VEGA
# ========================

def black_scholes_vega(
    S0,
    K,
    T,
    r,
    sigma
):
    """
    Black-Scholes vega.
    """

    d1 = (
        np.log(S0 / K)
        + (r + 0.5 * sigma**2) * T
    ) / (
        sigma * np.sqrt(T)
    )

    return (
        S0
        * norm.pdf(d1)
        * np.sqrt(T)
    )


# ========================
# 9. IMPLIED VOLATILITY
# ========================

def implied_volatility(
    market_price,
    S0,
    K,
    T,
    r,
    sigma0=0.20,
    tol=1e-10,
    max_iter=100
):
    """
    Computes Black-Scholes implied volatility.

    1. Newton-Raphson
    2. Brent's method as a fallback
    """

    # ----------------------
    # No-arbitrage bounds
    # ----------------------

    lower_bound = max(
        S0 - K * np.exp(-r * T),
        0.0
    )

    upper_bound = S0

    if (
        market_price < lower_bound - 1e-8
        or market_price > upper_bound + 1e-8
    ):
        return np.nan

    # ================
    # Newton-Raphson
    # ================
    sigma = sigma0

    for _ in range(max_iter):

        if sigma <= 0:
            break

        price_bs = black_scholes_call(
            S0,
            K,
            T,
            r,
            sigma
        )

        error = (
            price_bs
            - market_price
        )

        if abs(error) < tol:
            return sigma

        vega = black_scholes_vega(
            S0,
            K,
            T,
            r,
            sigma
        )

        # Vega is too small
        if abs(vega) < 1e-10:
            break

        sigma_new = (
            sigma
            - error / vega
        )

        # Newton step enters a non-physical region
        if (
            sigma_new <= 0.0
            or sigma_new > 5.0
        ):
            break

        sigma = sigma_new

    # =========================
    # Fallback method: Brent
    # =========================

    def objective(sigma):

        return (
            black_scholes_call(
                S0,
                K,
                T,
                r,
                sigma
            )
            - market_price
        )

    try:

        return brentq(
            objective,
            1e-6,
            5.0,
            xtol=tol
        )

    except ValueError:

        return np.nan


# ==============================
# 10. ROUGH HESTON PARAMETERS
# ==============================

S0 = 100.0
# Dividend yield assumption
q=0

V0 = 0.04
theta = 0.04

kappa = 1.5
nu = 0.30
rho = -0.70

H = 0.10

r = 0.02
T = 1.0


# ==========================
# 11. NUMERICAL PARAMETERS
# ==========================

# Adams time resolution
N_time = 250

# Number of COS terms
N_cos = 256

# COS truncation width
L = 10.0


# =================
# 12. STRIKE GRID
# =================

K_values = np.arange(
    70.0,
    131.0,
    5.0
)


# ===========================
# 13. CUMULANT COMPUTATION
# ===========================

print()
print("Computing cumulants...")

cumulants = estimate_cumulants(
    V0=V0,
    kappa=kappa,
    theta=theta,
    nu=nu,
    rho=rho,
    H=H,
    r=r,
    T=T,
    N_time=N_time
)

print()
print("Cumulants:")
print(f"c1 = {cumulants[0]:.8f}")
print(f"c2 = {cumulants[1]:.8f}")
print(f"c4 = {cumulants[2]:.8f}")


# ========================================
# 14. IMPLIED VOLATILITY SMILE AND SKEW
# ========================================

prices_rough = []
implied_vols = []

print()
print("================================")
print("SMILE / SKEW ROUGH HESTON")
print("================================")
print()

print(
    f"{'K':>10}"
    f"{'RH price':>18}"
    f"{'Implied vol':>20}"
)

print("-" * 48)


for K in K_values:

    # ---------------------
    # Rough Heston price
    # ---------------------

    price_rh = price_call_cos(
        S0=S0,
        K=K,
        V0=V0,
        kappa=kappa,
        theta=theta,
        nu=nu,
        rho=rho,
        H=H,
        r=r,
        T=T,
        N_time=N_time,
        N_cos=N_cos,
        L=L,
        cumulants=cumulants
    )

    # ---------------------
    # Implied volatility
    # ---------------------

    sigma_impl = implied_volatility(
        market_price=price_rh,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma0=0.20
    )

    prices_rough.append(
        price_rh
    )

    implied_vols.append(
        sigma_impl
    )

    print(
        f"{K:10.2f}"
        f"{price_rh:18.8f}"
        f"{sigma_impl:20.8f}"
    )


# Conversion to NumPy arrays
prices_rough = np.array(
    prices_rough
)

implied_vols = np.array(
    implied_vols
)


# ===========================================
# 15. BLACK-SCHOLES RECONSTRUCTION CHECK
# ===========================================

reconstructed_prices = np.array(
    [
        black_scholes_call(
            S0,
            K,
            T,
            r,
            sigma
        )
        if np.isfinite(sigma)
        else np.nan

        for K, sigma in zip(
            K_values,
            implied_vols
        )
    ]
)

reconstruction_errors = np.abs(
    reconstructed_prices
    - prices_rough
)

print()
print("=================================")
print("BLACK-SCHOLES INVERSION CHECK")
print("=================================")

print()
print(
    "Maximum reconstruction error:",
    np.nanmax(reconstruction_errors)
)


# ===============================
# 16. ROUGH HESTON CALL PRICES
# ===============================

plt.figure(figsize=(8, 5))

plt.plot(
    K_values,
    prices_rough,
    marker="o"
)

plt.axvline(
    S0,
    linestyle="--",
    label=r"$K=S_0$"
)

plt.xlabel("Strike $K$")
plt.ylabel("Call price")
plt.title(
    "Call prices under Rough Heston"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "3_rough_heston_call_prices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)

# =======================================
# 17. ROUGH HESTON IMPLIED VOLATILITY
# =======================================

plt.figure(figsize=(8, 5))

plt.plot(
    K_values,
    implied_vols,
    marker="o"
)

plt.axvline(
    S0,
    linestyle="--",
    label=r"$K=S_0$"
)

plt.xlabel("Strike $K$")
plt.ylabel("Implied volatility")
plt.title(
    "Implied volatility under Rough Heston"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "4_rough_heston_implied_volatility.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)

# ============================================
# 18. ROUGH HESTON ATM SKEW TERM STRUCTURE
# ============================================

maturities_days = np.array([
    3, 7, 15, 31, 60, 90, 180, 365
], dtype=float)

T_values = maturities_days / 365.0

# Local log-moneyness around the ATM forward:
# k = log(K / F0(T)), with k = 0 at the forward ATM level.
k_local = np.linspace(
    -0.03,
    0.03,
    7
)

atm_skews = []

print()
print("===========================================")
print("ROUGH HESTON ATM SKEW TERM STRUCTURE")
print("===========================================")
print()
print(
    f"{'T (days)':>12}"
    f"{'ATM skew':>18}"
)
print("-" * 30)

for T_skew, days in zip(T_values, maturities_days):

    # Dividend yield q = 0 in this study
    F0 = S0 * np.exp(r * T_skew)

    # Strikes corresponding to local log-moneyness values
    K_local = F0 * np.exp(k_local)

    # Rough Heston cumulants for this maturity
    cumulants_skew = estimate_cumulants(
        V0=V0,
        kappa=kappa,
        theta=theta,
        nu=nu,
        rho=rho,
        H=H,
        r=r,
        T=T_skew,
        N_time=N_time
    )

    local_implied_vols = []

    for K_skew in K_local:

        price_rh_skew = price_call_cos(
            S0=S0,
            K=K_skew,
            V0=V0,
            kappa=kappa,
            theta=theta,
            nu=nu,
            rho=rho,
            H=H,
            r=r,
            T=T_skew,
            N_time=N_time,
            N_cos=N_cos,
            L=L,
            cumulants=cumulants_skew
        )
        # Convert Rough Heston prices into implied volatilities
        sigma_skew = implied_volatility(
            market_price=price_rh_skew,
            S0=S0,
            K=K_skew,
            T=T_skew,
            r=r,
            sigma0=0.20
        )

        local_implied_vols.append(
            sigma_skew
        )

    local_implied_vols = np.asarray(
        local_implied_vols,
        dtype=float
    )

    # Local linear approximation:
    # sigma_impl(k,T) ~= a(T) + S(T) k
    # where S(T) estimates the ATM skew.
    valid = np.isfinite(
        local_implied_vols
    )

    if np.sum(valid) >= 3:

        slope, intercept = np.polyfit(
            k_local[valid],
            local_implied_vols[valid],
            deg=1
        )

        skew_atm = slope

    else:

        skew_atm = np.nan

    atm_skews.append(
        skew_atm
    )
    # S(3d), S(7d), ...
    print(
        f"{days:12.0f}"
        f"{skew_atm:18.8f}"
    )

# T -> S(T)
atm_skews = np.asarray(
    atm_skews,
    dtype=float
)


# ====================================
# 19. ATM SKEW TERM STRUCTURE PLOT
# ====================================

fig_skew, ax_skew = plt.subplots(
    figsize=(8, 5)
)

ax_skew.plot(
    maturities_days,
    atm_skews,
    marker="o"
)

ax_skew.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0
)

ax_skew.set_xlabel(
    "Maturity $T$ (days)"
)

ax_skew.set_ylabel(
    r"Skew ATM $\left.\partial_k \sigma_{\mathrm{impl}}(k,T)\right|_{k=0}$"
)

ax_skew.set_title(
    "ATM skew term structure under Rough Heston"
)

ax_skew.grid(
    True
)

fig_skew.tight_layout()

fig_skew.savefig(
    FIGURES_DIR / "5_rough_heston_atm_skew_term_structure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print(
    "Figure saved to:",
    FIGURES_DIR / "5_rough_heston_atm_skew_term_structure.png"
)
