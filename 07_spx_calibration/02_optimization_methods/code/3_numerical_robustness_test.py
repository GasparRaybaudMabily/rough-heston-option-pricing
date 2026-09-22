"""
Numerical robustness test for the Rough Heston calibration.

This script evaluates the calibrated Rough Heston parameters under several
numerical configurations. It studies convergence with respect to N_time and
N_cos, as well as sensitivity to the COS truncation parameter L, while keeping
the calibrated model parameters fixed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

from scipy.special import gamma
from scipy.stats import norm
from scipy.optimize import brentq


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

# Final specification retained
S0 = 3936.710926
q = 0.0114029141

# r(T) and P(0,T) are read by maturity from the sample.


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
        if not np.allclose(discount_values, discount_T, rtol=0.0, atol=1e-12):
            raise ValueError(f"Multiple Discount_factor values for T={T}")
        if not np.isclose(discount_T, np.exp(-r_T * T), rtol=1e-10, atol=1e-12):
            raise ValueError(f"Inconsistency between r_GSW / Discount_factor for T={T}")

        try:
            call_prices = price_calls_cos_batch(
                S0=S0, strikes=strikes, V0=V0, kappa=kappa, theta=theta,
                nu=nu, rho=rho, H=H, r=r_T, q=q, T=T,
                N_time=N_time, N_cos=N_cos, L=L
            )

            iv_values = []
            for K, option_type, call_price in zip(strikes, option_types, call_prices):
                if option_type == "Call":
                    model_price = call_price
                elif option_type == "Put":
                    model_price = call_price - S0 * np.exp(-q * T) + K * discount_T
                else:
                    model_price = np.nan

                if np.isfinite(model_price) and -1e-10 < model_price < 0:
                    model_price = 0.0

                iv_values.append(
                    implied_volatility(
                        market_price=model_price,
                        option_type=option_type,
                        S0=S0, K=K, T=T, r=r_T, q=q
                    )
                )

            model_iv[indices] = iv_values
        except Exception:
            model_iv[indices] = np.nan

    return model_iv


# ===========================================
# 13. CALIBRATED PARAMETERS RETAINED (TRF)
# ===========================================

params_TRF = np.array([
    0.07269029,     # V0
    0.90173188,     # kappa
    0.07958936,     # theta
    0.61556845,     # nu
    -0.66309006,    # rho
    0.23675172      # H
], dtype=float)

IV_market = df["IV_market"].to_numpy(dtype=float)


# ==========================================
# 14. NUMERICAL ROBUSTNESS CONFIGURATIONS
# ==========================================
# The test is performed WITH FIXED ROUGH HESTON PARAMETERS.
# The goal is not to obtain a lower RMSE: we verify that IV values and RMSE
# change only slightly when the numerical resolution is refined.
#
# Block A: N_time / N_cos convergence, with L = 10 fixed.
# Block B: sensitivity to L, with N_time = 160 and N_cos = 256 fixed.

CONFIGS = [
    {"label": "base_80_128_L10",       "N_time": 80,  "N_cos": 128, "L": 10.0, "family": "resolution"},
    {"label": "fine_160_256_L10",      "N_time": 160, "N_cos": 256, "L": 10.0, "family": "resolution"},
    {"label": "veryfine_320_512_L10",  "N_time": 320, "N_cos": 512, "L": 10.0, "family": "resolution"},
    {"label": "fine_160_256_L8",       "N_time": 160, "N_cos": 256, "L": 8.0,  "family": "L"},
    {"label": "fine_160_256_L12",      "N_time": 160, "N_cos": 256, "L": 12.0, "family": "L"},
]

REFERENCE_LABEL = "base_80_128_L10"
L_REFERENCE_LABEL = "fine_160_256_L10"


# ====================================
# 15. EVALUATION OF A CONFIGURATION
# ====================================

def evaluate_configuration(config):
    global N_time, N_cos, L

    N_time = int(config["N_time"])
    N_cos = int(config["N_cos"])
    L = float(config["L"])

    print()
    print("============================================================")
    print(
        f"CONFIGURATION: N_time={N_time}, N_cos={N_cos}, L={L:g}"
    )
    print("============================================================")

    start = time.time()
    model_iv = rough_heston_iv_surface(params_TRF)
    elapsed = time.time() - start

    valid = np.isfinite(model_iv) & np.isfinite(IV_market)
    n_valid = int(valid.sum())
    n_invalid = int(len(df) - n_valid)

    if n_valid == 0:
        rmse = mae = max_error = np.nan
    else:
        errors = model_iv[valid] - IV_market[valid]
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        max_error = float(np.max(np.abs(errors)))

    print(f"Valid observations   : {n_valid} / {len(df)}")
    print(f"RMSE IV              : {rmse:.8f}")
    print(f"MAE IV               : {mae:.8f}")
    print(f"Max IV error         : {max_error:.8f}")
    print(f"Time                  : {elapsed:.2f} s")

    stats = {
        "Label": config["label"],
        "Family": config["family"],
        "N_time": N_time,
        "N_cos": N_cos,
        "L": L,
        "N_valid_IV": n_valid,
        "N_invalid_IV": n_invalid,
        "RMSE": rmse,
        "MAE": mae,
        "Max_error": max_error,
        "Time_seconds": elapsed,
    }

    return model_iv, stats


# ========================================
# 16. COMPUTATION OF ALL CONFIGURATIONS
# ========================================

print()
print("============================================================")
print("ROUGH HESTON NUMERICAL ROBUSTNESS TEST")
print("============================================================")
print("File:", file_path)
print("Number of observations:", len(df))
print("Number of effective maturities:", df["T"].nunique())
print("Fixed TRF parameters:", params_TRF)

iv_by_label = {}
summary_rows = []

for config in CONFIGS:
    iv, stats = evaluate_configuration(config)
    iv_by_label[config["label"]] = iv
    summary_rows.append(stats)

summary = pd.DataFrame(summary_rows)


# ======================================
# 17. COMPARISONS WITH THE REFERENCES
# ======================================

ref_iv = iv_by_label[REFERENCE_LABEL]
l_ref_iv = iv_by_label[L_REFERENCE_LABEL]

for i, row in summary.iterrows():
    label = row["Label"]
    current_iv = iv_by_label[label]

    valid_ref = np.isfinite(current_iv) & np.isfinite(ref_iv)
    if valid_ref.any():
        delta = current_iv[valid_ref] - ref_iv[valid_ref]
        summary.loc[i, "Mean_abs_delta_IV_vs_base"] = np.mean(np.abs(delta))
        summary.loc[i, "RMS_delta_IV_vs_base"] = np.sqrt(np.mean(delta**2))
        summary.loc[i, "Max_abs_delta_IV_vs_base"] = np.max(np.abs(delta))
    else:
        summary.loc[i, [
            "Mean_abs_delta_IV_vs_base",
            "RMS_delta_IV_vs_base",
            "Max_abs_delta_IV_vs_base"
        ]] = np.nan

    valid_l = np.isfinite(current_iv) & np.isfinite(l_ref_iv)
    if valid_l.any():
        delta_l = current_iv[valid_l] - l_ref_iv[valid_l]
        summary.loc[i, "Mean_abs_delta_IV_vs_L10fine"] = np.mean(np.abs(delta_l))
        summary.loc[i, "Max_abs_delta_IV_vs_L10fine"] = np.max(np.abs(delta_l))
    else:
        summary.loc[i, [
            "Mean_abs_delta_IV_vs_L10fine",
            "Max_abs_delta_IV_vs_L10fine"
        ]] = np.nan

base_rmse = float(summary.loc[summary["Label"] == REFERENCE_LABEL, "RMSE"].iloc[0])
summary["Delta_RMSE_vs_base"] = summary["RMSE"] - base_rmse


# ====================================
# 18. RESULTS BY EFFECTIVE MATURITY
# ====================================

maturity_rows = []

for T, group in df.groupby("T", sort=True):
    idx = group.index.to_numpy()
    market = IV_market[idx]

    row = {
        "T": float(T),
        "T_days_effective": float(T * 365.25),
        "N": int(len(group)),
    }

    for config in CONFIGS:
        label = config["label"]
        model = iv_by_label[label][idx]
        valid = np.isfinite(model) & np.isfinite(market)
        if valid.any():
            err = model[valid] - market[valid]
            row[f"RMSE__{label}"] = float(np.sqrt(np.mean(err**2)))
        else:
            row[f"RMSE__{label}"] = np.nan

    base_local = iv_by_label[REFERENCE_LABEL][idx]
    fine_local = iv_by_label[L_REFERENCE_LABEL][idx]
    valid_num = np.isfinite(base_local) & np.isfinite(fine_local)
    if valid_num.any():
        d = fine_local[valid_num] - base_local[valid_num]
        row["Mean_abs_delta_IV_160_256_vs_80_128"] = float(np.mean(np.abs(d)))
        row["Max_abs_delta_IV_160_256_vs_80_128"] = float(np.max(np.abs(d)))
    else:
        row["Mean_abs_delta_IV_160_256_vs_80_128"] = np.nan
        row["Max_abs_delta_IV_160_256_vs_80_128"] = np.nan

    maturity_rows.append(row)

maturity_results = pd.DataFrame(maturity_rows)


# ===========================
# 19. FULL TABLE BY OPTION
# ===========================

full_results = df.copy()
for config in CONFIGS:
    label = config["label"]
    full_results[f"IV_RH__{label}"] = iv_by_label[label]
    full_results[f"Residual__{label}"] = iv_by_label[label] - IV_market

full_results["Delta_IV__160_256_vs_80_128"] = (
    iv_by_label["fine_160_256_L10"] - iv_by_label["base_80_128_L10"]
)
full_results["Delta_IV__320_512_vs_160_256"] = (
    iv_by_label["veryfine_320_512_L10"] - iv_by_label["fine_160_256_L10"]
)
full_results["Delta_IV__L8_vs_L10_fine"] = (
    iv_by_label["fine_160_256_L8"] - iv_by_label["fine_160_256_L10"]
)
full_results["Delta_IV__L12_vs_L10_fine"] = (
    iv_by_label["fine_160_256_L12"] - iv_by_label["fine_160_256_L10"]
)


# ======================
# 20. SUMMARY DISPLAY
# ======================

print()
print("============================================================")
print("GLOBAL SUMMARY")
print("============================================================")
print(
    summary[[
        "Label", "N_time", "N_cos", "L", "N_valid_IV", "N_invalid_IV",
        "RMSE", "MAE", "Max_error", "Delta_RMSE_vs_base",
        "Mean_abs_delta_IV_vs_base", "Max_abs_delta_IV_vs_base", "Time_seconds"
    ]].to_string(index=False)
)

print()
print("============================================================")
print("CONVERGENCE 80/128 -> 160/256 -> 320/512 (L=10)")
print("============================================================")
for label in [
    "base_80_128_L10",
    "fine_160_256_L10",
    "veryfine_320_512_L10",
]:
    row = summary.loc[summary["Label"] == label].iloc[0]
    print(
        f"{label:24s} | RMSE={row['RMSE']:.8f} "
        f"| ΔRMSE/base={row['Delta_RMSE_vs_base']:+.8f} "
        f"| max |ΔIV|/base={row['Max_abs_delta_IV_vs_base']:.8f}"
    )

print()
print("============================================================")
print("SENSITIVITY TO L (N_time=160, N_cos=256)")
print("============================================================")
for label in [
    "fine_160_256_L8",
    "fine_160_256_L10",
    "fine_160_256_L12",
]:
    row = summary.loc[summary["Label"] == label].iloc[0]
    print(
        f"{label:24s} | RMSE={row['RMSE']:.8f} "
        f"| mean |ΔIV| vs L10={row['Mean_abs_delta_IV_vs_L10fine']:.10f} "
        f"| max |ΔIV| vs L10={row['Max_abs_delta_IV_vs_L10fine']:.10f}"
    )


# ==============
# 21. FIGURES
# ==============

# Figure 1: global RMSE by configuration
fig, ax = plt.subplots(figsize=(10, 6))
labels_plot = summary["Label"].tolist()
ax.bar(np.arange(len(summary)), summary["RMSE"].to_numpy())
ax.set_xticks(np.arange(len(summary)))
ax.set_xticklabels(labels_plot, rotation=30, ha="right")
ax.set_ylabel("Implied volatility RMSE")
ax.set_title("Numerical robustness of the Rough Heston calibration")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "04_numerical_robustness_rmse_by_configuration.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Figure 2: numerical impact by maturity of 160/256 vs 80/128
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    maturity_results["T_days_effective"],
    maturity_results["Max_abs_delta_IV_160_256_vs_80_128"],
    marker="o"
)
ax.set_xlabel("Effective maturity (days)")
ax.set_ylabel("Max |Δ IV| : 160/256 vs 80/128")
ax.set_title("Impact of numerical refinement by maturity")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "05_numerical_robustness_impact_by_maturity.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


# ==================
# 22. CSV EXPORTS
# ==================

summary_path = (
    CSV_DIR
    / "04_numerical_robustness_summary.csv"
)

maturity_path = (
    CSV_DIR
    / "05_numerical_robustness_by_maturity.csv"
)

full_path = (
    CSV_DIR
    / "06_numerical_robustness_full_results.csv"
)

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

maturity_results.to_csv(
    maturity_path,
    index=False,
    encoding="utf-8-sig"
)

full_results.to_csv(
    full_path,
    index=False,
    encoding="utf-8-sig"
)

print()
print("============================================================")
print("EXPORTS")
print("============================================================")
print("Global summary       :", summary_path)
print("By maturity          :", maturity_path)
print("Full results         :", full_path)
print("RMSE figure          :", FIGURES_DIR / "04_numerical_robustness_rmse_by_configuration.png")
print("Figure by maturity   :", FIGURES_DIR / "05_numerical_robustness_impact_by_maturity.png")
print()
print("Numerical robustness test completed.")
