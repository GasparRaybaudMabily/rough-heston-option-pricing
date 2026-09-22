"""
Rough Heston multi-start TRF calibration test.

This script calibrates the Rough Heston model to the final SPX calibration
sample using the Trust Region Reflective (TRF) least-squares algorithm from
several initial parameter sets. It compares the resulting RMSE and MAE values
to assess the stability of the calibration with respect to initialization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

from scipy.special import gamma
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import least_squares


# ====================================
# 1. LOADING THE CALIBRATION SAMPLE
# ====================================

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

PREP_CSV_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "01_data_preparation"
    / "results"
)

METHOD_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "02_optimization_methods"
)

CSV_DIR = METHOD_DIR / "results"
FIGURES_DIR = METHOD_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

file_path = PREP_CSV_DIR / "05_calibration_sample.csv"
df = pd.read_csv(file_path)

required_columns = ["Strike", "T", "IV_market", "Option_type", "r_GSW", "Discount_factor", "K_over_F0"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in the calibration file: {missing_columns}")

df = df[
    np.isfinite(df["Strike"])
    & np.isfinite(df["T"])
    & np.isfinite(df["IV_market"])
    & (df["Strike"] > 0)
    & (df["T"] > 0)
    & (df["IV_market"] > 0)
    & df["Option_type"].isin(["Call", "Put"])
].copy().reset_index(drop=True)

print()
print("============================================================")
print("CALIBRATION SAMPLE")
print("============================================================")
print()
print("File:", file_path)
print("Number of observations:", len(df))
print("Number of effective maturities:", df["T"].nunique())
# =======================
# 2. MARKET PARAMETERS
# =======================

# Values previously estimated using the GSW curve
S0 = 3936.710926
q = 0.0114029141

# Zero-coupon rates and discount factors are read
# directement dans "05_calibration_sample.csv" :
#     r_GSW
#     Discount_factor
#
# This ensures that the calibration uses exactly the same
# curve as the one used to construct market implied volatilities.

print()
print("Market specification:")
print(f"S0 = {S0:.6f}")
print(f"q  = {100.0 * q:.6f} %")
print(
    "r_GSW min/max = "
    f"{100.0 * df['r_GSW'].min():.6f} % / "
    f"{100.0 * df['r_GSW'].max():.6f} %"
)
print(
    "Discount factor min/max = "
    f"{df['Discount_factor'].min():.10f} / "
    f"{df['Discount_factor'].max():.10f}"
)
print(
    "Construction moneyness = K/F0(T)"
)


# ==========================
# 3. NUMERICAL PARAMETERS
# ==========================

# ------------------------------------------------------------
# IMPORTANT:
#
# We deliberately start with a reasonable resolution
# to test the calibration.
#
# Once the procedure has been validated, N_time
# and N_cos can be increased for the final calibration.
# ------------------------------------------------------------

N_time = 80
N_cos = 128
L = 10.0


# ================================
# 4. QUADRATIC RICCATI FUNCTION
# ================================

def riccati_F(
    phi,
    h,
    kappa,
    nu,
    rho
):

    return (
        -0.5 * (phi**2 + 1j * phi)

        + (
            1j * rho * nu * phi
            - kappa
        ) * h

        + 0.5 * nu**2 * h**2
    )


# =======================================
# 5. ROUGH HESTON SOLVER:
#    VECTORIZED ADAMS PREDICTOR-CORRECTOR
# =======================================

def solve_rough_riccati_adams_vectorized(
    phi_values,
    H,
    kappa,
    nu,
    rho,
    T,
    N
):

    phi_values = np.atleast_1d(
        np.asarray(
            phi_values,
            dtype=complex
        )
    )

    M = len(phi_values)

    alpha = H + 0.5

    dt = T / N

    t = np.linspace(
        0.0,
        T,
        N + 1
    )

    h = np.zeros(
        (M, N + 1),
        dtype=complex
    )


    for n in range(N):

        j = np.arange(
            n + 1
        )


        # ====================================================
        # PREDICTOR
        # ====================================================

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


        predictor_sum = (
            F_history @ b
        )


        h_pred = (
            dt**alpha
            / gamma(alpha + 1)
            * predictor_sum
        )


        # ====================================================
        # CORRECTOR
        # ====================================================

        a = np.empty(
            n + 1
        )


        a[0] = (
            n**(alpha + 1)

            - (
                n - alpha
            ) * (n + 1)**alpha
        )


        if n >= 1:

            jj = np.arange(
                1,
                n + 1
            )

            a[1:] = (
                (n - jj + 2)**(alpha + 1)

                + (n - jj)**(alpha + 1)

                - 2
                * (n - jj + 1)**(alpha + 1)
            )


        corrector_sum = (
            F_history @ a
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
                    rho
                )

                + corrector_sum
            )
        )


    return t, h


# ==========================================
# 6. ROUGH HESTON CHARACTERISTIC FUNCTION
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
    q,
    T,
    N_time
):

    phi_values = np.atleast_1d(
        np.asarray(
            phi_values,
            dtype=complex
        )
    )


    alpha = H + 0.5


    t, h = (
        solve_rough_riccati_adams_vectorized(
            phi_values=phi_values,
            H=H,
            kappa=kappa,
            nu=nu,
            rho=rho,
            T=T,
            N=N_time
        )
    )


    dt = T / N_time


    # ========================================================
    # CLASSICAL INTEGRAL OF h
    # ========================================================

    integral_h = dt * (
        0.5 * h[:, 0]

        + np.sum(
            h[:, 1:-1],
            axis=1
        )

        + 0.5 * h[:, -1]
    )


    # ========================================================
    # FRACTIONAL INTEGRAL
    # ========================================================

    t_left = t[:-1]
    t_right = t[1:]


    fractional_weights = (
        (T - t_left)**(1.0 - alpha)

        - (T - t_right)**(1.0 - alpha)
    )


    fractional_integral = (
        h[:, :-1]
        @ fractional_weights

        / gamma(
            2.0 - alpha
        )
    )


    # ========================================================
    # CHARACTERISTIC FUNCTION
    # ========================================================

    exponent = (
        1j
        * phi_values
        * (r - q)
        * T

        + V0
        * fractional_integral

        + kappa
        * theta
        * integral_h
    )


    return np.exp(
        exponent
    )


# =========================
# 7. CUMULANT ESTIMATION
# =========================

def estimate_cumulants(
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
    n_points=9
):

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
        q=q,
        T=T,
        N_time=N_time
    )


    K_values = np.real(
        np.log(Phi)
    )


    coeff = (
        np.polynomial.polynomial.polyfit(
            z,
            K_values,
            deg=6
        )
    )


    c1 = coeff[1]

    c2 = (
        2.0
        * coeff[2]
    )

    c4 = (
        24.0
        * coeff[4]
    )


    return c1, c2, c4


# ===================================
# 8. CHI AND PSI FUNCTIONS FOR COS
# ===================================

def chi_cos(
    j,
    a,
    b,
    c,
    d
):

    u = (
        j
        * np.pi
        / (b - a)
    )


    term_d = (
        np.cos(
            u * (d - a)
        )

        + u
        * np.sin(
            u * (d - a)
        )
    )


    term_c = (
        np.cos(
            u * (c - a)
        )

        + u
        * np.sin(
            u * (c - a)
        )
    )


    return (
        np.exp(d)
        * term_d

        - np.exp(c)
        * term_c
    ) / (
        1.0 + u**2
    )


def psi_cos(
    j,
    a,
    b,
    c,
    d
):

    u = (
        j
        * np.pi
        / (b - a)
    )


    if j == 0:

        return (
            d - c
        )


    return (
        np.sin(
            u * (d - a)
        )

        - np.sin(
            u * (c - a)
        )
    ) / u


# ======================================
# 9. COS PRICING FOR MULTIPLE STRIKES
#    FOR THE SAME MATURITY
# ======================================

def price_calls_cos_batch(
    S0,
    strikes,
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
    N_cos,
    L
):

    # ========================================================
    # CUMULANTS:
    # computed ONLY ONCE for the maturity
    # ========================================================

    cumulants = estimate_cumulants(
        V0=V0,
        kappa=kappa,
        theta=theta,
        nu=nu,
        rho=rho,
        H=H,
        r=r,
        q=q,
        T=T,
        N_time=N_time
    )


    c1, c2, c4 = cumulants


    c2 = max(
        float(
            np.real(c2)
        ),
        1e-12
    )


    c4_positive = max(
        float(
            np.real(c4)
        ),
        0.0
    )


    width = (
        L
        * np.sqrt(
            c2
            + np.sqrt(
                c4_positive
            )
        )
    )


    if (
        not np.isfinite(width)
        or width <= 1e-10
    ):

        return np.full(
            len(strikes),
            np.nan
        )


    a = c1 - width
    b = c1 + width


    # ========================================================
    # COS FREQUENCIES
    # ========================================================

    j_values = np.arange(
        N_cos
    )


    u = (
        j_values
        * np.pi
        / (b - a)
    )


    # ========================================================
    # CHARACTERISTIC FUNCTION :
    # computed ONLY ONCE for the entire maturity
    # ========================================================

    Phi = rough_heston_characteristic(
        phi_values=u,
        V0=V0,
        kappa=kappa,
        theta=theta,
        nu=nu,
        rho=rho,
        H=H,
        r=r,
        q=q,
        T=T,
        N_time=N_time
    )


    density_coeff = np.real(
        Phi
        * np.exp(
            -1j
            * u
            * a
        )
    )


    weights = np.ones(
        N_cos
    )

    weights[0] = 0.5


    prices = []


    for K in strikes:

        k = np.log(
            K / S0
        )


        c = max(
            k,
            a
        )

        d = b


        G = np.zeros(
            N_cos
        )


        if c < b:

            for j in range(
                N_cos
            ):

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
                    S0
                    * chi

                    - K
                    * psi
                )


        price = (
            np.exp(
                -r * T
            )

            * 2.0
            / (b - a)

            * np.sum(
                weights
                * density_coeff
                * G
            )
        )


        prices.append(
            float(
                np.real(price)
            )
        )


    return np.asarray(
        prices
    )


# ==========================================================
# 10. BLACK-SCHOLES CALL / PUT WITH CONTINUOUS DIVIDEND q
# ==========================================================

def black_scholes_price(option_type, S0, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "Put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return np.nan


# =======================================
# 11. BLACK-SCHOLES IMPLIED VOLATILITY
# =======================================

def implied_volatility(market_price, option_type, S0, K, T, r, q, sigma_min=1e-8, sigma_max=5.0):
    if not np.isfinite(market_price) or market_price <= 0 or not np.isfinite(T) or T <= 0:
        return np.nan

    if option_type == "Call":
        lower_bound = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper_bound = S0 * np.exp(-q * T)
    elif option_type == "Put":
        lower_bound = max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
        upper_bound = K * np.exp(-r * T)
    else:
        return np.nan

    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        return np.nan

    def objective(sigma):
        return black_scholes_price(option_type, S0, K, T, r, q, sigma) - market_price

    try:
        f_min = objective(sigma_min)
        f_max = objective(sigma_max)
        if not np.isfinite(f_min) or not np.isfinite(f_max) or f_min * f_max > 0:
            return np.nan
        return brentq(objective, sigma_min, sigma_max, xtol=1e-10, rtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError, OverflowError):
        return np.nan


# ===========================================
# 12. ROUGH HESTON IV OVER THE FULL SAMPLE
# ===========================================

def rough_heston_iv_surface(params):
    V0, kappa, theta, nu, rho, H = params
    model_iv = np.full(len(df), np.nan)

    for T, group in df.groupby("T"):
        indices = group.index.values
        strikes = group["Strike"].to_numpy(dtype=float)
        option_types = group["Option_type"].to_numpy()

        # An effective maturity must correspond to a single
        # zero-coupon rate and a single discount factor.
        r_values = group["r_GSW"].to_numpy(dtype=float)
        discount_values = group["Discount_factor"].to_numpy(dtype=float)

        if not np.all(np.isfinite(r_values)):
            raise ValueError(f"Non-finite r_GSW for T={T}")

        if not np.all(np.isfinite(discount_values)):
            raise ValueError(f"Non-finite Discount_factor for T={T}")

        r_T = float(r_values[0])
        discount_T = float(discount_values[0])

        if not np.allclose(r_values, r_T, rtol=0.0, atol=1e-12):
            raise ValueError(f"Multiple r_GSW values for T={T}")

        if not np.allclose(
            discount_values,
            discount_T,
            rtol=0.0,
            atol=1e-12
        ):
            raise ValueError(
                f"Multiple discount factors for T={T}"
            )

        # Internal CSV consistency check.
        discount_from_rate = np.exp(-r_T * T)

        if not np.isclose(
            discount_T,
            discount_from_rate,
            rtol=1e-10,
            atol=1e-12
        ):
            raise ValueError(
                f"Inconsistency between r_GSW / Discount_factor for T={T}: "
                f"P_CSV={discount_T:.12f}, "
                f"exp(-rT)={discount_from_rate:.12f}"
            )

        try:
            call_prices = price_calls_cos_batch(
                S0=S0,
                strikes=strikes,
                V0=V0,
                kappa=kappa,
                theta=theta,
                nu=nu,
                rho=rho,
                H=H,
                r=r_T,
                q=q,
                T=T,
                N_time=N_time,
                N_cos=N_cos,
                L=L
            )

            iv_values = []

            for K, option_type, call_price in zip(
                strikes,
                option_types,
                call_prices
            ):
                if option_type == "Call":
                    model_price = call_price

                elif option_type == "Put":
                    # Call-put parity:
                    # P = C - S0 exp(-qT) + K P(0,T)
                    model_price = (
                        call_price
                        - S0 * np.exp(-q * T)
                        + K * discount_T
                    )

                else:
                    model_price = np.nan

                if (
                    np.isfinite(model_price)
                    and -1e-10 < model_price < 0
                ):
                    model_price = 0.0

                iv_values.append(
                    implied_volatility(
                        market_price=model_price,
                        option_type=option_type,
                        S0=S0,
                        K=K,
                        T=T,
                        r=r_T,
                        q=q
                    )
                )

            model_iv[indices] = iv_values

        except Exception:
            model_iv[indices] = np.nan

    return model_iv

# ==========================
# 13. MARKET VOLATILITIES
# ==========================

IV_market = (
    df["IV_market"]
    .to_numpy(
        dtype=float
    )
)


# ==============
# 14. WEIGHTS
# ==============

# Uniform weighting:
#
# w_i = 1

weights = np.ones(
    len(df)
)


# =========================
# 15. EVALUATION COUNTER
# =========================

evaluation_counter = {
    "n": 0
}


# ========================
# 16. RESIDUAL FUNCTION
# ========================

def residuals(
    params
):

    evaluation_counter["n"] += 1

    n_eval = (
        evaluation_counter["n"]
    )


    start = time.time()


    IV_model = (
        rough_heston_iv_surface(
            params
        )
    )


    residual = (
        IV_model
        - IV_market
    )


    # ========================================================
    # HANDLING NUMERICAL FAILURES
    # ========================================================

    invalid = (
        ~np.isfinite(
            residual
        )
    )


    # An invalid model volatility receives a penalty
    residual[
        invalid
    ] = 1.0


    # Weighting
    residual = (
        np.sqrt(weights)
        * residual
    )


    elapsed = (
        time.time()
        - start
    )


    rmse = np.sqrt(
        np.mean(
            residual**2
        )
    )


    V0, kappa, theta, nu, rho, H = params


    print()
    print(
        f"Evaluation {n_eval:4d}"
        f" | RMSE = {rmse:.6f}"
        f" | time = {elapsed:.2f} s"
    )

    print(
        f"V0={V0:.5f}, "
        f"kappa={kappa:.5f}, "
        f"theta={theta:.5f}, "
        f"nu={nu:.5f}, "
        f"rho={rho:.5f}, "
        f"H={H:.5f}"
    )

    if np.any(
        invalid
    ):

        print(
            "IV values not computed:",
            np.sum(invalid)
        )


    return residual


# =======================================
# MULTI-START: MULTIPLE INITIAL POINTS
# =======================================

initial_points = [
    np.array([0.04, 1.50, 0.04, 0.40, -0.70, 0.10]),
    np.array([0.06, 0.80, 0.06, 0.60, -0.50, 0.20]),
    np.array([0.09, 2.00, 0.08, 0.80, -0.80, 0.30]),
    np.array([0.03, 3.00, 0.10, 1.00, -0.40, 0.15]),
    np.array([0.12, 0.40, 0.12, 0.30, -0.90, 0.35])
]


# =========
# BOUNDS
# =========

lower_bounds = np.array([
    0.005,    # V0
    0.05,     # kappa
    0.005,    # theta
    0.05,     # nu
    -0.99,    # rho
    0.02      # H
])

upper_bounds = np.array([
    0.20,     # V0
    5.00,     # kappa
    0.20,     # theta
    2.00,     # nu
    0.99,     # rho
    0.49      # H
])


# ==================
# RESULTS STORAGE
# ==================

multi_start_results = []


# ===================
# MULTI-START LOOP
# ===================

for start_id, x0 in enumerate(initial_points, start=1):

    print()
    print("============================================================")
    print(f"MULTI-START {start_id}/{len(initial_points)}")
    print("============================================================")
    print()

    print("Initial point:")
    print(f"V0     = {x0[0]}")
    print(f"kappa  = {x0[1]}")
    print(f"theta  = {x0[2]}")
    print(f"nu     = {x0[3]}")
    print(f"rho    = {x0[4]}")
    print(f"H      = {x0[5]}")
    print()


    # Reset counter
    evaluation_counter["n"] = 0


    start_time = time.time()


    result = least_squares(
        fun=residuals,
        x0=x0,
        bounds=(
            lower_bounds,
            upper_bounds
        ),
        method="trf",
        jac="2-point",
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
        max_nfev=40,
        verbose=0
    )


    elapsed_time = (
        time.time()
        - start_time
    )


    # ========================================================
    # FINAL RMSE CALCULATION
    # ========================================================

    IV_RH_start = rough_heston_iv_surface(
        result.x
    )


    valid = (
        np.isfinite(IV_RH_start)
        & np.isfinite(IV_market)
    )


    errors = (
        IV_RH_start[valid]
        - IV_market[valid]
    )


    rmse = np.sqrt(
        np.mean(
            errors**2
        )
    )


    mae = np.mean(
        np.abs(
            errors
        )
    )


    # ========================================================
    # STORAGE
    # ========================================================

    multi_start_results.append({
        "Start": start_id,
        "Initial_V0": x0[0], "Initial_kappa": x0[1], "Initial_theta": x0[2],
        "Initial_nu": x0[3], "Initial_rho": x0[4], "Initial_H": x0[5],
        "V0": result.x[0], "kappa": result.x[1], "theta": result.x[2],
        "nu": result.x[3], "rho": result.x[4], "H": result.x[5],
        "RMSE": rmse, "MAE": mae, "Success": result.success,
        "NFEV": result.nfev, "Time_seconds": elapsed_time,
        "N_valid_IV": int(np.sum(valid)), "N_invalid_IV": int(len(df) - np.sum(valid))
    })


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("Result:")
    print()

    print(f"V0     = {result.x[0]:.8f}")
    print(f"kappa  = {result.x[1]:.8f}")
    print(f"theta  = {result.x[2]:.8f}")
    print(f"nu     = {result.x[3]:.8f}")
    print(f"rho    = {result.x[4]:.8f}")
    print(f"H      = {result.x[5]:.8f}")

    print()
    print(f"RMSE   = {rmse:.8f}")
    print(f"MAE    = {mae:.8f}")
    print(f"Time   = {elapsed_time:.2f} s")
    print(f"NFEV   = {result.nfev}")
    print(f"Success = {result.success}")


# ==========================
# FINAL MULTI-START TABLE
# ==========================

df_multi_start = pd.DataFrame(
    multi_start_results
)


df_multi_start = (
    df_multi_start
    .sort_values(
        by="RMSE"
    )
    .reset_index(
        drop=True
    )
)


print()
print("============================================================")
print("MULTI-START RESULTS")
print("============================================================")
print()

print(
    df_multi_start.to_string(
        index=False
    )
)


# ================
# BEST SOLUTION
# ================

best = (
    df_multi_start
    .iloc[0]
)


print()
print("============================================================")
print("BEST SOLUTION")
print("============================================================")
print()

print(
    f"Start  = {int(best['Start'])}"
)

print(
    f"V0     = {best['V0']:.8f}"
)

print(
    f"kappa  = {best['kappa']:.8f}"
)

print(
    f"theta  = {best['theta']:.8f}"
)

print(
    f"nu     = {best['nu']:.8f}"
)

print(
    f"rho    = {best['rho']:.8f}"
)

print(
    f"H      = {best['H']:.8f}"
)

print(
    f"RMSE   = {best['RMSE']:.8f}"
)

print(
    f"MAE    = {best['MAE']:.8f}"
)


# =========
# EXPORT
# =========

multi_start_output = (
    CSV_DIR
    / "01_trf_multistart_results.csv"
)

df_multi_start.to_csv(
    multi_start_output,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Multi-start test results saved to:")
print(multi_start_output)
print()
