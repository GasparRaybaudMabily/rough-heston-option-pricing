"""
04 - Global analysis
ATM-forward implied volatility term structure:
Market vs Rough Heston vs classical Heston

This script does not recalibrate either model.
It reads:
- the final calibration sample;
- the definitive global Rough Heston parameters;
- the definitive global classical Heston parameters.

For each effective market maturity:
1) market ATM IV is estimated locally as the intercept of a regression of
   IV_market on log(K/F0), using the 7 observations closest to ATM-forward;
2) Rough Heston and classical Heston ATM IV is calculated exactly at
   strike K = F0(T), using the globally calibrated parameters;
3) results are aggregated by calendar horizon using the same weights as
   the local market estimate.

Outputs:
- results/08_atm_forward_iv_term_structure.csv
- figures/14_atm_forward_iv_term_structure.png
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

PREP_CSV_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "01_data_preparation"
    / "results"
)
CALIB_CSV_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "03_global_calibration"
    / "results"
)
ANALYSIS_DIR = ROOT_DIR / "07_spx_calibration" / "04_global_analysis"
CSV_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RH_PARAM_FILE = CALIB_CSV_DIR / "02_rough_heston_global_calibration_parameters.csv"
HESTON_PARAM_FILE = CALIB_CSV_DIR / "04_heston_global_calibration_parameters.csv"

INPUT_SAMPLE = PREP_CSV_DIR / "05_calibration_sample.csv"

OUT_CSV = CSV_DIR / "08_atm_forward_iv_term_structure.csv"
OUT_FIG = FIGURES_DIR / "14_atm_forward_iv_term_structure.png"


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
    Reads a parameter CSV under several common formats:
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
print("GLOBAL PARAMETERS LOADED FROM 03_GLOBAL_CALIBRATION")
print("=" * 76)
print("Rough Heston :", THETA_RH)
print("Heston       :", THETA_H)


# ======================================
# 4. MARKET ATM ESTIMATION PARAMETERS
# ======================================

N_LOCAL = 7
MIN_LOCAL = 3


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


# ===============================
# 8. LOADING THE MARKET SAMPLE
# ===============================

if not INPUT_SAMPLE.exists():
    raise FileNotFoundError(
        f"File not found: {INPUT_SAMPLE}\n"
        "Check that the final calibration sample has been created."
    )

df = pd.read_csv(INPUT_SAMPLE)

required = ["T", "K_over_F0", "IV_market"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        "Missing columns in input file: " + ", ".join(missing)
    )

if "T_days" in df.columns:
    df["Calendar_days"] = df["T_days"].astype(int)
elif "Calendar_days" not in df.columns:
    df["Calendar_days"] = np.rint(df["T"] * 365.25).astype(int)

valid = (
    np.isfinite(df["T"])
    & np.isfinite(df["K_over_F0"])
    & np.isfinite(df["IV_market"])
    & (df["T"] > 0)
    & (df["K_over_F0"] > 0)
    & (df["IV_market"] > 0)
)

df = df.loc[valid].copy()
df["log_forward_moneyness"] = np.log(df["K_over_F0"])


# =========================================
# 9. MARKET ATM IV BY EFFECTIVE MATURITY
# =========================================

def estimate_market_atm_iv(group, n_local=N_LOCAL):
    """
    Locally estimates:
        IV_market(k,T) = a(T) + b(T) * k,
        k = log(K/F0(T)),

    and retains a(T) as the ATM-forward implied volatility.
    """
    g = group[
        np.isfinite(group["log_forward_moneyness"])
        & np.isfinite(group["IV_market"])
    ].copy()

    if len(g) < MIN_LOCAL:
        return np.nan, len(g)

    g["abs_k"] = np.abs(g["log_forward_moneyness"])
    g = g.sort_values("abs_k").head(min(n_local, len(g))).copy()

    x = g["log_forward_moneyness"].to_numpy(dtype=float)
    y = g["IV_market"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    intercept = float(coef[0])
    return intercept, int(len(g))


# ============================
# 10. ATM IV OF BOTH MODELS
# ============================

def model_atm_iv(model_name, T):
    """
    Computes the model ATM-forward IV at strike K = F0(T).
    """
    T = float(T)
    r_T = zero_coupon_rate(T)
    F0_T = forward_price(T)
    K = F0_T

    strikes = np.array([K], dtype=float)

    if model_name == "Rough Heston":
        call_price = price_calls_cos_rough(
            S0,
            strikes,
            THETA_RH,
            r_T,
            q,
            T,
            N_time,
            N_cos,
            L,
        )[0]

    elif model_name == "Heston classique":
        call_price = price_calls_cos_heston(
            S0,
            strikes,
            THETA_H,
            r_T,
            q,
            T,
            N_cos,
            L,
        )[0]

    else:
        raise ValueError("Unknown model.")

    return implied_volatility(
        call_price,
        "Call",
        S0,
        K,
        T,
        r_T,
        q,
    )


# ========================================
# 11. CALCULATION BY EFFECTIVE MATURITY
# ========================================

rows = []

effective_groups = list(df.groupby("T", sort=True))

print("\n" + "=" * 76)
print("ATM-FORWARD IMPLIED VOLATILITY TERM STRUCTURE")
print("=" * 76)

for i, (T, group) in enumerate(effective_groups, start=1):
    calendar_days = int(group["Calendar_days"].iloc[0])

    iv_market_atm, n_local = estimate_market_atm_iv(group)
    iv_rh_atm = model_atm_iv("Rough Heston", T)
    iv_h_atm = model_atm_iv("Heston classique", T)

    print(
        f"{i:02d}/{len(effective_groups)} | "
        f"{calendar_days:3d} days | "
        f"T={T:.8f} | "
        f"Market={iv_market_atm:.6f} | "
        f"RH={iv_rh_atm:.6f} | "
        f"Heston={iv_h_atm:.6f}"
    )

    rows.append({
        "T": float(T),
        "Effective_days": float(T) * 365.25,
        "Calendar_days": calendar_days,
        "N_local": n_local,
        "IV_ATM_market": iv_market_atm,
        "IV_ATM_Rough_Heston": iv_rh_atm,
        "IV_ATM_Heston": iv_h_atm,
    })

effective = pd.DataFrame(rows)


# ======================================
# 12. AGGREGATION BY CALENDAR HORIZON
# ======================================

calendar_rows = []

for days, group in effective.groupby("Calendar_days", sort=True):
    weights = group["N_local"].to_numpy(dtype=float)

    def weighted_average(column):
        values = group[column].to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

        if not np.any(mask):
            return np.nan

        return float(np.average(values[mask], weights=weights[mask]))

    calendar_rows.append({
        "Calendar_days": int(days),
        "N_effective_maturities": int(len(group)),
        "N_local_total": int(np.sum(weights)),
        "IV_ATM_market": weighted_average("IV_ATM_market"),
        "IV_ATM_Rough_Heston": weighted_average("IV_ATM_Rough_Heston"),
        "IV_ATM_Heston": weighted_average("IV_ATM_Heston"),
    })

term_structure = pd.DataFrame(calendar_rows)


# =============
# 13. FIGURE
# =============

x = term_structure["Calendar_days"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(10, 6.2))

ax.plot(
    x,
    100.0 * term_structure["IV_ATM_market"],
    marker="o",
    linewidth=2.0,
    label="Market",
)

ax.plot(
    x,
    100.0 * term_structure["IV_ATM_Rough_Heston"],
    marker="o",
    linewidth=1.9,
    label="Rough Heston",
)

ax.plot(
    x,
    100.0 * term_structure["IV_ATM_Heston"],
    marker="o",
    linestyle="--",
    linewidth=1.8,
    label="Classical Heston",
)

ax.set_xlabel("Calendar horizon (days)")
ax.set_ylabel("ATM-forward implied volatility (%)")
ax.set_title(
    "ATM-forward implied volatility term structure"
)

ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# =============
# 14. EXPORT
# =============

term_structure.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 76)
print("RESULTS BY CALENDAR HORIZON")
print("=" * 76)

print(
    term_structure.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nSaved files:")
print(" -", OUT_CSV)
print(" -", OUT_FIG)
print("\nDone.")
