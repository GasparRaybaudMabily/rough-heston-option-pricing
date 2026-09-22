"""
06 - Short-term analysis
1 - Short-term implied volatility surfaces - Rough Heston and classical Heston

This script does not recalibrate either model.
It reads the final parameters produced in 05 - Short-term calibration,
then reconstructs smooth Rough Heston and classical Heston implied volatility
surfaces on a common (K/F0(T), T) grid.

Both models use the COS pricing engine and the same Black-Scholes inversion
to maintain a comparable numerical framework.

Outputs:
- results/01_rough_heston_short_term_implied_volatility_surface.csv
- results/02_heston_short_term_implied_volatility_surface.csv
- figures/01_rough_heston_short_term_implied_volatility_surface.png
- figures/02_heston_short_term_implied_volatility_surface.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.special import gamma
from scipy.stats import norm


# ===========
# 1. PATHS
# ===========

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

CALIB_CSV_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "05_short_term_calibration"
    / "results"
)
ANALYSIS_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "06_short_term_analysis"
)
CSV_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RH_PARAM_FILE = (
    CALIB_CSV_DIR
    / "02_rough_heston_short_term_calibration_parameters.csv"
)
HESTON_PARAM_FILE = (
    CALIB_CSV_DIR
    / "04_heston_short_term_calibration_parameters.csv"
)

OUT_RH_CSV = CSV_DIR / "01_rough_heston_short_term_implied_volatility_surface.csv"
OUT_H_CSV = CSV_DIR / "02_heston_short_term_implied_volatility_surface.csv"

OUT_RH_FIG = FIGURES_DIR / "01_rough_heston_short_term_implied_volatility_surface.png"
OUT_H_FIG = FIGURES_DIR / "02_heston_short_term_implied_volatility_surface.png"


# =====================================
# 2. MARKET AND NUMERICAL PARAMETERS
# =====================================

S0 = 3936.710926
q = 0.0114029141

N_time = 80
N_cos = 128
L = 10.0

# GSW / Svensson parameters as of September 13, 2022.
BETA0 = 4.1325311721141
BETA1 = -0.483330087369049
BETA2 = 249.569929179392
BETA3 = -250.639461221969
TAU1 = 2.72637117177024
TAU2 = 2.74854195681383


def zero_coupon_rate(T):
    """Continuously compounded GSW/Svensson zero-coupon rate, in decimal form."""
    T = float(T)
    if T <= 0:
        return np.nan

    x1 = T / TAU1
    x2 = T / TAU2

    a1 = (1.0 - np.exp(-x1)) / x1
    a2 = (1.0 - np.exp(-x2)) / x2

    z_percent = (
        BETA0
        + BETA1 * a1
        + BETA2 * (a1 - np.exp(-x1))
        + BETA3 * (a2 - np.exp(-x2))
    )
    return z_percent / 100.0


def discount_factor(T):
    r_T = zero_coupon_rate(T)
    return np.exp(-r_T * T)


def forward_price(T):
    return S0 * np.exp(-q * T) / discount_factor(T)


# =============================================
# 3. ROBUST LOADING OF CALIBRATED PARAMETERS
# =============================================

def load_parameter_vector(path, aliases):
    """
    Reads a parameter CSV in several common formats:
    - one row with one column per parameter;
    - two columns of the Parameter / Value type.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    p = pd.read_csv(path)

    # Case 1: parameters stored in columns.
    normalized_columns = {str(c).strip().lower(): c for c in p.columns}
    values = []

    ok = True
    for canonical, names in aliases:
        found = None
        for name in names:
            if name.lower() in normalized_columns:
                found = normalized_columns[name.lower()]
                break
        if found is None:
            ok = False
            break
        values.append(float(p[found].iloc[0]))

    if ok:
        return np.asarray(values, dtype=float)

    # Case 2: one name column + one value column.
    name_candidates = [
        c for c in p.columns
        if str(c).strip().lower() in
        {"paramètre", "parametre", "parameter", "param", "nom"}
    ]
    value_candidates = [
        c for c in p.columns
        if str(c).strip().lower() in
        {"valeur", "value", "estimate", "estimation"}
    ]

    if name_candidates and value_candidates:
        name_col = name_candidates[0]
        value_col = value_candidates[0]

        mapping = {
            str(k).strip().lower(): float(v)
            for k, v in zip(p[name_col], p[value_col])
        }

        values = []
        for canonical, names in aliases:
            found = None
            for name in names:
                if name.lower() in mapping:
                    found = mapping[name.lower()]
                    break
            if found is None:
                raise ValueError(
                    f"Parameter {canonical} missing from {path.name}. "
                    f"Available columns: {list(p.columns)}"
                )
            values.append(found)

        return np.asarray(values, dtype=float)

    raise ValueError(
        f"Unrecognized parameter file format: {path}\n"
        f"Available columns: {list(p.columns)}"
    )


RH_ALIASES = [
    ("V0", ["V0", "v0"]),
    ("kappa", ["kappa"]),
    ("theta", ["theta"]),
    ("nu", ["nu"]),
    ("rho", ["rho"]),
    ("H", ["H", "h"]),
]

HESTON_ALIASES = [
    ("v0", ["v0", "V0"]),
    ("kappa", ["kappa"]),
    ("theta", ["theta"]),
    ("xi", ["xi"]),
    ("rho", ["rho"]),
]

THETA_RH = load_parameter_vector(RH_PARAM_FILE, RH_ALIASES)
THETA_H = load_parameter_vector(HESTON_PARAM_FILE, HESTON_ALIASES)

print("\n" + "=" * 76)
print("FINAL PARAMETERS LOADED FROM 05 - SHORT-TERM CALIBRATION")
print("=" * 76)
print("Rough Heston :", THETA_RH)
print("Heston       :", THETA_H)


# =========================
# 4. COMMON SURFACE GRID
# =========================

# Domain intentionally centered on the most economically readable region.
MONEYNESS = np.linspace(0.95, 1.05, 61)

# Same grid for both models.
T_DAYS = np.array([1, 3, 7, 15, 31], dtype=float)

T_GRID = T_DAYS / 365.25


# ======================================
# 5. COMMON COS / BLACK-SCHOLES TOOLS
# ======================================

def chi_cos(j, a, b, c, d):
    u = j * np.pi / (b - a)

    term_d = np.cos(u * (d - a)) + u * np.sin(u * (d - a))
    term_c = np.cos(u * (c - a)) + u * np.sin(u * (c - a))

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


def black_scholes_price(option_type, S0, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan

    d1 = (
        np.log(S0 / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        return (
            S0 * np.exp(-q * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    if option_type == "Put":
        return (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S0 * np.exp(-q * T) * norm.cdf(-d1)
        )

    return np.nan


def implied_volatility(
    market_price,
    option_type,
    S0,
    K,
    T,
    r,
    q,
    sigma_min=1e-8,
    sigma_max=5.0,
):
    if (
        not np.isfinite(market_price)
        or market_price <= 0
        or not np.isfinite(T)
        or T <= 0
    ):
        return np.nan

    if option_type == "Call":
        lower = max(
            S0 * np.exp(-q * T) - K * np.exp(-r * T),
            0.0,
        )
        upper = S0 * np.exp(-q * T)

    elif option_type == "Put":
        lower = max(
            K * np.exp(-r * T) - S0 * np.exp(-q * T),
            0.0,
        )
        upper = K * np.exp(-r * T)

    else:
        return np.nan

    if market_price < lower - 1e-10 or market_price > upper + 1e-10:
        return np.nan

    def objective(sigma):
        return (
            black_scholes_price(
                option_type, S0, K, T, r, q, sigma
            )
            - market_price
        )

    try:
        f_min = objective(sigma_min)
        f_max = objective(sigma_max)

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
            xtol=1e-10,
            rtol=1e-10,
            maxiter=200,
        )

    except (ValueError, RuntimeError, OverflowError):
        return np.nan


# ==================
# 6. ROUGH HESTON
# ==================

def riccati_F(phi, h, kappa, nu, rho):
    return (
        -0.5 * (phi**2 + 1j * phi)
        + (1j * rho * nu * phi - kappa) * h
        + 0.5 * nu**2 * h**2
    )


def solve_rough_riccati_adams_vectorized(
    phi_values, H, kappa, nu, rho, T, N
):
    phi_values = np.atleast_1d(
        np.asarray(phi_values, dtype=complex)
    )

    M = len(phi_values)
    alpha = H + 0.5
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    h = np.zeros((M, N + 1), dtype=complex)

    for n in range(N):
        j = np.arange(n + 1)

        b = (
            (n + 1 - j)**alpha
            - (n - j)**alpha
        )

        F_history = riccati_F(
            phi_values[:, None],
            h[:, :n + 1],
            kappa,
            nu,
            rho,
        )

        h_pred = (
            dt**alpha
            / gamma(alpha + 1)
            * (F_history @ b)
        )

        a = np.empty(n + 1)

        a[0] = (
            n**(alpha + 1)
            - (n - alpha) * (n + 1)**alpha
        )

        if n >= 1:
            jj = np.arange(1, n + 1)
            a[1:] = (
                (n - jj + 2)**(alpha + 1)
                + (n - jj)**(alpha + 1)
                - 2 * (n - jj + 1)**(alpha + 1)
            )

        h[:, n + 1] = (
            dt**alpha
            / gamma(alpha + 2)
            * (
                riccati_F(
                    phi_values,
                    h_pred,
                    kappa,
                    nu,
                    rho,
                )
                + F_history @ a
            )
        )

    return t, h


def rough_heston_characteristic(
    phi_values,
    V0,
    kappa,
    theta,
    nu,
    rho,
    H,
    r,
    q,
    T,
    N_time,
):
    phi_values = np.atleast_1d(
        np.asarray(phi_values, dtype=complex)
    )

    alpha = H + 0.5

    t, h = solve_rough_riccati_adams_vectorized(
        phi_values,
        H,
        kappa,
        nu,
        rho,
        T,
        N_time,
    )

    dt = T / N_time

    integral_h = dt * (
        0.5 * h[:, 0]
        + np.sum(h[:, 1:-1], axis=1)
        + 0.5 * h[:, -1]
    )

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

    exponent = (
        1j * phi_values * (r - q) * T
        + V0 * fractional_integral
        + kappa * theta * integral_h
    )

    return np.exp(exponent)


def estimate_cumulants_rough(
    V0,
    kappa,
    theta,
    nu,
    rho,
    H,
    r,
    q,
    T,
    N_time,
    radius=0.15,
    n_points=9,
):
    z = np.linspace(-radius, radius, n_points)
    phi = -1j * z

    Phi = rough_heston_characteristic(
        phi,
        V0,
        kappa,
        theta,
        nu,
        rho,
        H,
        r,
        q,
        T,
        N_time,
    )

    K_values = np.real(np.log(Phi))

    coeff = np.polynomial.polynomial.polyfit(
        z,
        K_values,
        deg=6,
    )

    return (
        coeff[1],
        2.0 * coeff[2],
        24.0 * coeff[4],
    )


def price_calls_cos_rough(
    S0,
    strikes,
    params,
    r,
    q,
    T,
    N_time,
    N_cos,
    L,
):
    V0, kappa, theta, nu, rho, H = params

    c1, c2, c4 = estimate_cumulants_rough(
        V0,
        kappa,
        theta,
        nu,
        rho,
        H,
        r,
        q,
        T,
        N_time,
    )

    c2 = max(float(np.real(c2)), 1e-12)
    c4_positive = max(float(np.real(c4)), 0.0)

    width = L * np.sqrt(
        c2 + np.sqrt(c4_positive)
    )

    if not np.isfinite(width) or width <= 1e-10:
        return np.full(len(strikes), np.nan)

    a = c1 - width
    b = c1 + width

    j_values = np.arange(N_cos)
    u = j_values * np.pi / (b - a)

    Phi = rough_heston_characteristic(
        u,
        V0,
        kappa,
        theta,
        nu,
        rho,
        H,
        r,
        q,
        T,
        N_time,
    )

    density_coeff = np.real(
        Phi * np.exp(-1j * u * a)
    )

    weights = np.ones(N_cos)
    weights[0] = 0.5

    prices = []

    for K in strikes:
        k = np.log(K / S0)
        c = max(k, a)
        d = b

        G = np.zeros(N_cos)

        if c < b:
            for j in range(N_cos):
                G[j] = (
                    S0 * chi_cos(j, a, b, c, d)
                    - K * psi_cos(j, a, b, c, d)
                )

        price = (
            np.exp(-r * T)
            * 2.0 / (b - a)
            * np.sum(
                weights
                * density_coeff
                * G
            )
        )

        prices.append(float(np.real(price)))

    return np.asarray(prices)


# ======================
# 7. CLASSICAL HESTON
# ======================

def heston_characteristic(
    phi_values,
    v0,
    kappa,
    theta,
    xi,
    rho,
    r,
    q,
    T,
):
    u = np.atleast_1d(
        np.asarray(phi_values, dtype=complex)
    )

    iu = 1j * u
    beta = kappa - rho * xi * iu

    d = np.sqrt(
        beta**2
        + xi**2 * (u**2 + iu)
    )

    denom = beta + d

    g = np.divide(
        beta - d,
        denom,
        out=np.zeros_like(beta - d),
        where=np.abs(denom) > 1e-14,
    )

    exp_dt = np.exp(-d * T)

    log_term = np.log(
        (1.0 - g * exp_dt)
        / (1.0 - g)
    )

    C = (
        iu * (r - q) * T
        + (kappa * theta / xi**2)
        * (
            (beta - d) * T
            - 2.0 * log_term
        )
    )

    D = (
        (beta - d) / xi**2
        * (1.0 - exp_dt)
        / (1.0 - g * exp_dt)
    )

    return np.exp(C + D * v0)


def estimate_cumulants_heston(
    v0,
    kappa,
    theta,
    xi,
    rho,
    r,
    q,
    T,
    radius=0.15,
    n_points=9,
):
    z = np.linspace(-radius, radius, n_points)
    phi = -1j * z

    Phi = heston_characteristic(
        phi,
        v0,
        kappa,
        theta,
        xi,
        rho,
        r,
        q,
        T,
    )

    K_values = np.real(np.log(Phi))

    coeff = np.polynomial.polynomial.polyfit(
        z,
        K_values,
        deg=6,
    )

    return (
        coeff[1],
        2.0 * coeff[2],
        24.0 * coeff[4],
    )


def price_calls_cos_heston(
    S0,
    strikes,
    params,
    r,
    q,
    T,
    N_cos,
    L,
):
    v0, kappa, theta, xi, rho = params

    c1, c2, c4 = estimate_cumulants_heston(
        v0,
        kappa,
        theta,
        xi,
        rho,
        r,
        q,
        T,
    )

    c2 = max(float(np.real(c2)), 1e-12)
    c4_positive = max(float(np.real(c4)), 0.0)

    width = L * np.sqrt(
        c2 + np.sqrt(c4_positive)
    )

    if not np.isfinite(width) or width <= 1e-10:
        return np.full(len(strikes), np.nan)

    a = c1 - width
    b = c1 + width

    j_values = np.arange(N_cos)
    u = j_values * np.pi / (b - a)

    Phi = heston_characteristic(
        u,
        v0,
        kappa,
        theta,
        xi,
        rho,
        r,
        q,
        T,
    )

    density_coeff = np.real(
        Phi * np.exp(-1j * u * a)
    )

    weights = np.ones(N_cos)
    weights[0] = 0.5

    prices = []

    for K in strikes:
        k = np.log(K / S0)
        c = max(k, a)
        d = b

        G = np.zeros(N_cos)

        if c < b:
            for j in range(N_cos):
                G[j] = (
                    S0 * chi_cos(j, a, b, c, d)
                    - K * psi_cos(j, a, b, c, d)
                )

        price = (
            np.exp(-r * T)
            * 2.0 / (b - a)
            * np.sum(
                weights
                * density_coeff
                * G
            )
        )

        prices.append(float(np.real(price)))

    return np.asarray(prices)


# ========================================================
# 8. CONVERSION OF MODEL PRICES TO IMPLIED VOLATILITIES
# ========================================================

def model_surface(model_name):
    surface = np.full(
        (len(T_GRID), len(MONEYNESS)),
        np.nan,
    )

    rows = []

    print("\n" + "=" * 76)
    print(f"IMPLIED VOLATILITY SURFACE - {model_name}")
    print("=" * 76)

    for i, (days, T) in enumerate(
        zip(T_DAYS, T_GRID),
        start=1,
    ):
        r_T = zero_coupon_rate(T)
        df_T = discount_factor(T)
        F0_T = forward_price(T)
        strikes = F0_T * MONEYNESS

        print(
            f"{i:02d}/{len(T_GRID)} | "
            f"{days:6.1f} days | "
            f"T={T:.6f}"
        )

        if model_name == "Rough Heston":
            call_prices = price_calls_cos_rough(
                S0,
                strikes,
                THETA_RH,
                r_T,
                q,
                T,
                N_time,
                N_cos,
                L,
            )
        else:
            call_prices = price_calls_cos_heston(
                S0,
                strikes,
                THETA_H,
                r_T,
                q,
                T,
                N_cos,
                L,
            )

        for j, (m, K, call_price) in enumerate(
            zip(MONEYNESS, strikes, call_prices)
        ):
            if K < F0_T:
                option_type = "Put"
                price = (
                    call_price
                    - S0 * np.exp(-q * T)
                    + K * df_T
                )
            else:
                option_type = "Call"
                price = call_price

            iv = implied_volatility(
                price,
                option_type,
                S0,
                K,
                T,
                r_T,
                q,
            )

            surface[i - 1, j] = iv

            rows.append({
                "Model": model_name,
                "T_days": days,
                "T": T,
                "K_over_F0": m,
                "Forward": F0_T,
                "Strike": K,
                "r_GSW": r_T,
                "Discount_factor": df_T,
                "IV_model": iv,
            })

    return surface, pd.DataFrame(rows)


IV_RH, RH_TABLE = model_surface("Rough Heston")
IV_H, H_TABLE = model_surface("Heston classique")


# =================
# 9. DIAGNOSTICS
# =================

def print_diagnostic(name, surface):
    n_total = surface.size
    n_valid = int(np.isfinite(surface).sum())

    print("\n" + name)
    print("-" * len(name))
    print(f"Valid IVs : {n_valid}/{n_total}")

    if n_valid:
        print(f"Min IV     : {np.nanmin(surface):.6f}")
        print(f"Max IV     : {np.nanmax(surface):.6f}")


print("\n" + "=" * 76)
print("DIAGNOSTICS")
print("=" * 76)

print_diagnostic("Rough Heston", IV_RH)
print_diagnostic("Classical Heston", IV_H)


# ============================
# 10. COMPARABLE 3D FIGURES
# ============================

K_MESH, T_MESH = np.meshgrid(
    MONEYNESS,
    T_GRID,
)

all_valid = np.concatenate([
    IV_RH[np.isfinite(IV_RH)],
    IV_H[np.isfinite(IV_H)],
])

if len(all_valid) == 0:
    raise ValueError(
        "No valid implied volatility found across the two surfaces."
    )

z_min = float(np.min(all_valid))
z_max = float(np.max(all_valid))


def save_surface(surface, title, output_path):
    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        K_MESH,
        T_MESH,
        surface,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )

    ax.set_xlabel(
        r"Forward moneyness $K/F_0(T)$",
        labelpad=10,
    )
    ax.set_ylabel(
        r"Maturity $T$ (years)",
        labelpad=10,
    )
    ax.set_zlabel(
        "Implied volatility",
        labelpad=8,
    )

    ax.set_title(title, pad=18)

    # Same vertical scale to make both figures comparable.
    ax.set_zlim(z_min, z_max)

    ax.view_init(
        elev=27,
        azim=-125,
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


save_surface(
    IV_RH,
    "Implied volatility surface - Rough Heston",
    OUT_RH_FIG,
)

save_surface(
    IV_H,
    "Implied volatility surface - Classical Heston",
    OUT_H_FIG,
)


# ==============
# 11. EXPORTS
# ==============

RH_TABLE.to_csv(
    OUT_RH_CSV,
    index=False,
)

H_TABLE.to_csv(
    OUT_H_CSV,
    index=False,
)

print("\n" + "=" * 76)
print("SAVED FILES")
print("=" * 76)

for path in [
    OUT_RH_CSV,
    OUT_H_CSV,
    OUT_RH_FIG,
    OUT_H_FIG,
]:
    print(" -", path)

print("\nDone.")
