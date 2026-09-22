"""
Rough Heston vs Heston: Implied volatility and ATM Skew

This script compares the Rough Heston and classical Heston models through
their implied volatility smiles and ATM skew term structures. Rough Heston
call prices are computed with the COS method, while classical Heston prices
are obtained from the semi-analytical P1/P2 representation.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad

from pathlib import Path

# ================
# PROJECT PATHS
# ================

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

        # ============================
        # Adams-Bashforth predictor
        # ============================

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

        # ==========================
        # Adams-Moulton corrector
        # ==========================

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


# ==========================================
# 3. ROUGH HESTON CHARACTERISTIC FUNCTION
# ==========================================

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

    # ============================================
    # Classical integral of h: trapezoidal rule
    # ============================================

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


# ========================================
# 6. ROUGH HESTON CALL PRICING WITH COS
# ========================================

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

    # =========
    # Payoff
    # =========

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

    # =======
    # Prix
    # =======

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

    # =================
    # Newton-Raphson
    # =================

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

# ===========================================================
# 10. CLASSICAL HESTON CHARACTERISTIC FUNCTION AND PRICING
# ===========================================================

def heston_characteristic(
    phi,
    v0,
    kappa,
    theta,
    xi,
    rho,
    r,
    T
):
    """
    Characteristic function de X_T = log(S_T / S0)
    sous Classical Heston.
    """

    phi = np.asarray(phi, dtype=complex)

    d = np.sqrt(
        (kappa - 1j * rho * xi * phi)**2
        + xi**2 * (phi**2 + 1j * phi)
    )

    g = (
        kappa - 1j * rho * xi * phi - d
    ) / (
        kappa - 1j * rho * xi * phi + d
    )

    exp_minus_dT = np.exp(-d * T)

    C = (
        1j * phi * r * T
        + (kappa * theta / xi**2)
        * (
            (kappa - 1j * rho * xi * phi - d) * T
            - 2.0 * np.log(
                (1.0 - g * exp_minus_dT)
                / (1.0 - g)
            )
        )
    )

    D = (
        (kappa - 1j * rho * xi * phi - d)
        / xi**2
        * (
            (1.0 - exp_minus_dT)
            / (1.0 - g * exp_minus_dT)
        )
    )

    return np.exp(
        C + D * v0
    )


def heston_call_price(
    S0,
    K,
    v0,
    kappa,
    theta,
    xi,
    rho,
    r,
    T
):
    """
    Prices a European call option under the classical Heston model
    using the semi-analytical P1/P2 representation.
    """

    log_K_over_S0 = np.log(K / S0)

    def cf(u):
        return heston_characteristic(
            phi=u,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            r=r,
            T=T
        )

    phi_minus_i = cf(-1j)

    def integrand_p1(u):

        numerator = (
            np.exp(-1j * u * log_K_over_S0)
            * cf(u - 1j)
        )

        denominator = (
            1j * u * phi_minus_i
        )

        return np.real(
            numerator / denominator
        )

    def integrand_p2(u):

        return np.real(
            np.exp(-1j * u * log_K_over_S0)
            * cf(u)
            / (1j * u)
        )

    integral_p1 = quad(
        integrand_p1,
        1e-8,
        100.0,
        limit=250,
        epsabs=1e-7,
        epsrel=1e-7
    )[0]

    integral_p2 = quad(
        integrand_p2,
        1e-8,
        100.0,
        limit=250,
        epsabs=1e-7,
        epsrel=1e-7
    )[0]

    P1 = (
        0.5
        + integral_p1 / np.pi
    )

    P2 = (
        0.5
        + integral_p2 / np.pi
    )

    return (
        S0 * P1
        - K * np.exp(-r * T) * P2
    )


# =======================
# 11. MODEL PARAMETERS
# =======================

S0 = 100.0
r = 0.02

# Common parameters whenever possible
V0 = 0.04
theta = 0.04
kappa = 1.5
rho = -0.70

# Rough Heston
nu = 0.30
H = 0.10

# Classical Heston
# xi is the volatility-of-volatility parameter in Heston.
# The same numerical value as nu is used for a simple comparison.
xi = 0.30

# Rough Heston numerical parameters
N_time = 250
N_cos = 256
L = 10.0


# ============================================================
# 12. PLOT 1:
#     Rough Heston vs Heston implied volatility
# ============================================================

T_smile = 1.0

# Forward log-moneyness is used:
# k = log(K / F0(T))
k_values = np.linspace(
    -0.30,
    0.30,
    25
)

F0_smile = (
    S0 * np.exp(r * T_smile)
)

K_values = (
    F0_smile * np.exp(k_values)
)

# Rough Heston cumulants for T = T_smile
cumulants_rh = estimate_cumulants(
    V0=V0,
    kappa=kappa,
    theta=theta,
    nu=nu,
    rho=rho,
    H=H,
    r=r,
    T=T_smile,
    N_time=N_time
)

iv_rh = []
iv_heston = []

print()
iv_title = "IMPLIED VOLATILITY: ROUGH HESTON VS HESTON"
print("=" * len(iv_title))
print(iv_title)
print("=" * len(iv_title))
print()

for k, K in zip(
    k_values,
    K_values
):

    # Rough Heston
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
        T=T_smile,
        N_time=N_time,
        N_cos=N_cos,
        L=L,
        cumulants=cumulants_rh
    )

    sigma_rh = implied_volatility(
        market_price=price_rh,
        S0=S0,
        K=K,
        T=T_smile,
        r=r,
        sigma0=0.20
    )

    # Classical Heston
    price_h = heston_call_price(
        S0=S0,
        K=K,
        v0=V0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        r=r,
        T=T_smile
    )

    sigma_h = implied_volatility(
        market_price=price_h,
        S0=S0,
        K=K,
        T=T_smile,
        r=r,
        sigma0=0.20
    )

    iv_rh.append(
        sigma_rh
    )

    iv_heston.append(
        sigma_h
    )


iv_rh = np.asarray(
    iv_rh,
    dtype=float
)

iv_heston = np.asarray(
    iv_heston,
    dtype=float
)


fig_iv, ax_iv = plt.subplots(
    figsize=(8, 5)
)

ax_iv.plot(
    k_values,
    iv_rh,
    marker="o",
    label="Rough Heston"
)

ax_iv.plot(
    k_values,
    iv_heston,
    marker="s",
    label="Classical Heston"
)

ax_iv.axvline(
    0.0,
    linestyle="--",
    linewidth=1.0,
    label="ATM forward"
)

ax_iv.set_xlabel(
    r"Log-moneyness forward $k=\log(K/F_0(T))$"
)

ax_iv.set_ylabel(
    "IMPLIED VOLATILITY"
)

ax_iv.set_title(
    "IMPLIED VOLATILITY - Rough Heston vs Heston"
)

ax_iv.legend()
ax_iv.grid(True)
fig_iv.tight_layout()

fig_iv.savefig(
    FIGURES_DIR / "13_rough_heston_vs_heston_implied_volatility.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. PLOT 2:
#     Rough Heston vs Heston ATM skew term structure
# ============================================================

maturities_days = np.array([
    3, 7, 15, 31, 60, 90, 180, 365
], dtype=float)

T_values = (
    maturities_days / 365.0
)

# Local window around k = 0
k_local = np.linspace(
    -0.03,
    0.03,
    7
)

skew_rh = []
skew_heston = []

print()
skew_title = "ATM SKEW: ROUGH HESTON VS HESTON"
print("=" * len(skew_title))
print(skew_title)
print("=" * len(skew_title))
print()
print(
    f"{'T (days)':>12}"
    f"{'RH skew':>18}"
    f"{'Heston skew':>18}"
)
print("-" * 48)


for T_skew, days in zip(
    T_values,
    maturities_days
):

    F0 = (
        S0 * np.exp(r * T_skew)
    )

    K_local = (
        F0 * np.exp(k_local)
    )

    # Rough Heston cumulants for this maturity
    cumulants_skew_rh = estimate_cumulants(
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

    local_iv_rh = []
    local_iv_heston = []

    for K in K_local:

        # Rough Heston
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
            T=T_skew,
            N_time=N_time,
            N_cos=N_cos,
            L=L,
            cumulants=cumulants_skew_rh
        )

        sigma_rh = implied_volatility(
            market_price=price_rh,
            S0=S0,
            K=K,
            T=T_skew,
            r=r,
            sigma0=0.20
        )

        # Classical Heston
        price_h = heston_call_price(
            S0=S0,
            K=K,
            v0=V0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            r=r,
            T=T_skew
        )

        sigma_h = implied_volatility(
            market_price=price_h,
            S0=S0,
            K=K,
            T=T_skew,
            r=r,
            sigma0=0.20
        )

        local_iv_rh.append(
            sigma_rh
        )

        local_iv_heston.append(
            sigma_h
        )


    local_iv_rh = np.asarray(
        local_iv_rh,
        dtype=float
    )

    local_iv_heston = np.asarray(
        local_iv_heston,
        dtype=float
    )

    # Local regression around k = 0
    valid_rh = np.isfinite(
        local_iv_rh
    )

    valid_h = np.isfinite(
        local_iv_heston
    )

    if np.sum(valid_rh) >= 3:

        slope_rh = np.polyfit(
            k_local[valid_rh],
            local_iv_rh[valid_rh],
            deg=1
        )[0]

    else:

        slope_rh = np.nan


    if np.sum(valid_h) >= 3:

        slope_h = np.polyfit(
            k_local[valid_h],
            local_iv_heston[valid_h],
            deg=1
        )[0]

    else:

        slope_h = np.nan


    skew_rh.append(
        slope_rh
    )

    skew_heston.append(
        slope_h
    )

    print(
        f"{days:12.0f}"
        f"{slope_rh:18.8f}"
        f"{slope_h:18.8f}"
    )


skew_rh = np.asarray(
    skew_rh,
    dtype=float
)

skew_heston = np.asarray(
    skew_heston,
    dtype=float
)


fig_skew, ax_skew = plt.subplots(
    figsize=(8, 5)
)

ax_skew.plot(
    maturities_days,
    skew_rh,
    marker="o",
    label="Rough Heston"
)

ax_skew.plot(
    maturities_days,
    skew_heston,
    marker="s",
    label="Classical Heston"
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
    "ATM skew term structure - Rough Heston vs Heston"
)

ax_skew.legend()
ax_skew.grid(True)
fig_skew.tight_layout()

fig_skew.savefig(
    FIGURES_DIR / "14_rough_heston_vs_heston_atm_skew_term_structure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print()
print("Figures saved to:", FIGURES_DIR)
