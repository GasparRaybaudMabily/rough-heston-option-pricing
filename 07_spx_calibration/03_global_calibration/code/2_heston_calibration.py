"""
Global classical Heston calibration on SPX options.

This script calibrates the classical Heston model to the final SPX calibration
sample using a multi-start Trust Region Reflective (TRF) least-squares
procedure. Option prices are computed with the COS method and converted into
implied volatilities to evaluate the fit across maturities and forward
moneyness.
"""


import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq, least_squares
from scipy.stats import norm


# ====================================
# 1. LOADING THE CALIBRATION SAMPLE
# ====================================

# Chapter 7 root path
ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

# Path to the calibration sample
PREP_CSV_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "01_data_preparation"
    / "results"
)

# Path to the global calibration outputs
CALIB_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "03_global_calibration"
)

CSV_DIR = CALIB_DIR / "results"
FIGURES_DIR = CALIB_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load the calibration sample
file_path = PREP_CSV_DIR / "05_calibration_sample.csv"
df = pd.read_csv(file_path)

required_columns = [
    "Strike",
    "T",
    "IV_market",
    "Option_type",
    "r_GSW",
    "Discount_factor",
    "K_over_F0"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in the calibration file: {missing_columns}"
    )

df = df[
    np.isfinite(df["Strike"])
    & np.isfinite(df["T"])
    & np.isfinite(df["IV_market"])
    & np.isfinite(df["r_GSW"])
    & np.isfinite(df["Discount_factor"])
    & (df["Strike"] > 0)
    & (df["T"] > 0)
    & (df["IV_market"] > 0)
    & (df["Discount_factor"] > 0)
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

# Values estimated from call-put parity
# using the GSW zero-coupon curve.
S0 = 3936.710926
q = 0.0114029141


# ==========================
# 3. NUMERICAL PARAMETERS
# ==========================

# Same COS configuration as for the Rough Heston calibration,
# to make the numerical comparison as consistent as possible.
N_cos = 128
L = 10.0


# ===================================
# 4. CHI AND PSI FUNCTIONS FOR COS
# ===================================

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


# ==============================================
# 5. CLASSICAL HESTON CHARACTERISTIC FUNCTION
# ==============================================

def heston_characteristic(
    phi_values,
    v0,
    kappa,
    theta,
    xi,
    rho,
    r,
    q,
    T
):
    """
    Characteristic function of the log-return X_T = log(S_T / S_0)
    under the classical Heston model.
    """

    u = np.atleast_1d(
        np.asarray(phi_values, dtype=complex)
    )

    iu = 1j * u

    beta = kappa - rho * xi * iu
    d = np.sqrt(
        beta**2
        + xi**2 * (u**2 + iu)
    )

    # Stable form.
    denom = beta + d

    g = np.divide(
        beta - d,
        denom,
        out=np.zeros_like(beta - d),
        where=np.abs(denom) > 1e-14
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

    return np.exp(
        C + D * v0
    )


# ================================
# 6. HESTON CUMULANT ESTIMATION
# ================================

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
    n_points=9
):
    # Same numerical cumulant estimation procedure as the one
    # used in the Rough Heston COS engine.

    z = np.linspace(
        -radius,
        radius,
        n_points
    )

    phi = -1j * z

    Phi = heston_characteristic(
        phi_values=phi,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        r=r,
        q=q,
        T=T
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


# ========================================
# 7. HESTON PRICING WITH THE COS METHOD
# ========================================

# The classical Heston model is the one presented in Chapter 2.
#
# In this chapter, Heston prices are nevertheless computed with
# the COS method, which is also used for Rough Heston.
# This choice makes it possible to compare both models with a common
# numerical engine and to limit the influence of the pricing
# method on the comparison of calibration performance.

def price_calls_cos_heston_batch(
    S0,
    strikes,
    v0,
    kappa,
    theta,
    xi,
    rho,
    r,
    q,
    T,
    N_cos,
    L
):
    c1, c2, c4 = estimate_cumulants_heston(
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        r=r,
        q=q,
        T=T
    )

    c2 = max(
        float(np.real(c2)),
        1e-12
    )

    c4_positive = max(
        float(np.real(c4)),
        0.0
    )

    width = L * np.sqrt(
        c2 + np.sqrt(c4_positive)
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

    j_values = np.arange(N_cos)
    u = j_values * np.pi / (b - a)

    Phi = heston_characteristic(
        phi_values=u,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        r=r,
        q=q,
        T=T
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
                chi = chi_cos(
                    j, a, b, c, d
                )

                psi = psi_cos(
                    j, a, b, c, d
                )

                G[j] = (
                    S0 * chi
                    - K * psi
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

        prices.append(
            float(np.real(price))
        )

    return np.asarray(prices)


# =========================================================
# 8. BLACK-SCHOLES CALL / PUT WITH CONTINUOUS DIVIDEND q
# =========================================================

def black_scholes_price(
    option_type,
    S0,
    K,
    T,
    r,
    q,
    sigma
):
    if T <= 0 or sigma <= 0:
        return np.nan

    d1 = (
        np.log(S0 / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (
        sigma * np.sqrt(T)
    )

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


# ======================================
# 9. BLACK-SCHOLES IMPLIED VOLATILITY
# ======================================

def implied_volatility(
    market_price,
    option_type,
    S0,
    K,
    T,
    r,
    q,
    sigma_min=1e-8,
    sigma_max=5.0
):
    if (
        not np.isfinite(market_price)
        or market_price <= 0
        or not np.isfinite(T)
        or T <= 0
    ):
        return np.nan

    if option_type == "Call":
        lower_bound = max(
            S0 * np.exp(-q * T)
            - K * np.exp(-r * T),
            0.0
        )
        upper_bound = S0 * np.exp(-q * T)

    elif option_type == "Put":
        lower_bound = max(
            K * np.exp(-r * T)
            - S0 * np.exp(-q * T),
            0.0
        )
        upper_bound = K * np.exp(-r * T)

    else:
        return np.nan

    if (
        market_price < lower_bound - 1e-10
        or market_price > upper_bound + 1e-10
    ):
        return np.nan

    def objective(sigma):
        return (
            black_scholes_price(
                option_type,
                S0,
                K,
                T,
                r,
                q,
                sigma
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
            maxiter=200
        )

    except (
        ValueError,
        RuntimeError,
        OverflowError
    ):
        return np.nan


# =====================================
# 10. HESTON IV OVER THE FULL SAMPLE
# =====================================

def heston_iv_surface(params):
    v0, kappa, theta, xi, rho = params

    model_iv = np.full(
        len(df),
        np.nan
    )

    for T, group in df.groupby("T"):
        indices = group.index.values
        strikes = group["Strike"].to_numpy(dtype=float)
        option_types = group["Option_type"].to_numpy()

        r_values = group["r_GSW"].to_numpy(dtype=float)
        discount_values = group["Discount_factor"].to_numpy(dtype=float)

        if not np.all(np.isfinite(r_values)):
            raise ValueError(
                f"Non-finite r_GSW pour T={T}"
            )

        if not np.all(np.isfinite(discount_values)):
            raise ValueError(
                f"Non-finite Discount_factor pour T={T}"
            )

        r_T = float(r_values[0])
        discount_T = float(discount_values[0])

        if not np.allclose(
            r_values,
            r_T,
            rtol=0.0,
            atol=1e-12
        ):
            raise ValueError(
                f"Multiple r_GSW values pour T={T}"
            )

        if not np.allclose(
            discount_values,
            discount_T,
            rtol=0.0,
            atol=1e-12
        ):
            raise ValueError(
                f"Multiple Discount_factor values pour T={T}"
            )

        if not np.isclose(
            discount_T,
            np.exp(-r_T * T),
            rtol=1e-10,
            atol=1e-12
        ):
            raise ValueError(
                f"Inconsistency between r_GSW / Discount_factor pour T={T}"
            )

        try:
            call_prices = price_calls_cos_heston_batch(
                S0=S0,
                strikes=strikes,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                r=r_T,
                q=q,
                T=T,
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


# ======================================
# 11. MARKET VOLATILITIES AND METRICS
# ======================================

IV_market = df["IV_market"].to_numpy(
    dtype=float
)


def metrics(market, model):
    valid = (
        np.isfinite(market)
        & np.isfinite(model)
    )

    errors = (
        model[valid]
        - market[valid]
    )

    if len(errors) == 0:
        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAX_ERROR": np.nan
        }

    return {
        "N": int(valid.sum()),
        "RMSE": float(
            np.sqrt(np.mean(errors**2))
        ),
        "MAE": float(
            np.mean(np.abs(errors))
        ),
        "MAX_ERROR": float(
            np.max(np.abs(errors))
        )
    }


# ================================
# 12. BOUNDS AND INITIAL POINTS
# ================================

# Heston parameters:
# (v0, kappa, theta, xi, rho)

lower_bounds = np.array([
    0.005,    # v0
    0.05,     # kappa
    0.005,    # theta
    0.05,     # xi
    -0.99     # rho
])

upper_bounds = np.array([
    0.20,     # v0
    50.00,    # kappa
    0.20,     # theta
    8.00,     # xi
    0.99      # rho
])

# Several starting points are used to reduce the
# dependence of the Heston calibration on a single initialization.
HESTON_STARTS = [
    np.array([0.04, 1.50, 0.04, 0.40, -0.70]),
    np.array([0.08, 5.00, 0.06, 1.50, -0.50]),
    np.array([0.08, 20.0, 0.06, 3.00, -0.40]),
    np.array([0.08, 45.0, 0.06, 5.00, -0.40])
]


# ====================================================
# 13. CLASSICAL HESTON CALIBRATION - MULTISTART TRF
# ====================================================

def calibrate_heston_from(x0, start_number):
    evaluation_counter = {"n": 0}

    def residuals_heston(params):
        evaluation_counter["n"] += 1

        IV_model = heston_iv_surface(
            params
        )

        residual = (
            IV_model
            - IV_market
        )

        invalid = ~np.isfinite(
            residual
        )

        residual[invalid] = 1.0

        if (
            evaluation_counter["n"] == 1
            or evaluation_counter["n"] % 10 == 0
        ):
            rmse_current = np.sqrt(
                np.mean(residual**2)
            )

            print(
                f"Heston start {start_number}"
                f" | evaluation {evaluation_counter['n']:3d}"
                f" | RMSE = {rmse_current:.6f}"
                f" | invalid IV = {np.sum(invalid)}"
            )

        return residual

    start_time = time.time()

    result = least_squares(
        fun=residuals_heston,
        x0=x0,
        bounds=(
            lower_bounds,
            upper_bounds
        ),
        method="trf",
        jac="2-point",
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        max_nfev=120,
        verbose=0
    )

    elapsed = (
        time.time()
        - start_time
    )

    IV_model = heston_iv_surface(
        result.x
    )

    model_metrics = metrics(
        IV_market,
        IV_model
    )

    return (
        result,
        elapsed,
        IV_model,
        model_metrics
    )


print()
print("==========================================")
print("GLOBAL CLASSICAL HESTON CALIBRATION")
print("==========================================")
print()
print("S0 used:", f"{S0:.6f}")
print("q used   :", f"{100.0 * q:.6f} %")
print(
    "Rate range over the sample: "
    f"{100.0 * df['r_GSW'].min():.6f} % à "
    f"{100.0 * df['r_GSW'].max():.6f} %"
)
print("Moneyness  : K/F0(T) for sample construction")
print("N_cos      :", N_cos)
print("L          :", L)
print()

heston_runs = []

calibration_start = time.time()

for start_number, x0 in enumerate(
    HESTON_STARTS,
    start=1
):
    result, elapsed, IV_model, model_metrics = (
        calibrate_heston_from(
            x0=x0,
            start_number=start_number
        )
    )

    heston_runs.append(
        (
            result,
            elapsed,
            IV_model,
            model_metrics
        )
    )

    print(
        f"Start {start_number} completed"
        f" | RMSE = {model_metrics['RMSE']:.8f}"
        f" | MAE = {model_metrics['MAE']:.8f}"
        f" | time = {elapsed:.2f} s"
        f" | success = {result.success}"
    )

calibration_time = (
    time.time()
    - calibration_start
)

best_idx = int(
    np.nanargmin([
        run[3]["RMSE"]
        for run in heston_runs
    ])
)

best_result, best_time, IV_Heston, MET_H = (
    heston_runs[best_idx]
)

v0_star = best_result.x[0]
kappa_star = best_result.x[1]
theta_star = best_result.x[2]
xi_star = best_result.x[3]
rho_star = best_result.x[4]


# ==========================
# 14. CALIBRATION RESULTS
# ==========================

print()
print("================================")
print("CALIBRATION RESULTS")
print("================================")
print()

print(f"v0     = {v0_star:.8f}")
print(f"kappa  = {kappa_star:.8f}")
print(f"theta  = {theta_star:.8f}")
print(f"xi     = {xi_star:.8f}")
print(f"rho    = {rho_star:.8f}")

print()
print("Best starting point:", best_idx + 1)
print("Convergence :", best_result.success)
print("Message :", best_result.message)
print("Number of evaluations:", best_result.nfev)
print("Best start time:", f"{best_time:.2f} seconds")
print("Total multistart time:", f"{calibration_time:.2f} seconds")


# =======================
# 15. ERROR STATISTICS
# =======================

df["IV_Heston"] = IV_Heston

df["Residual_IV_Heston"] = (
    df["IV_Heston"]
    - df["IV_market"]
)

df["Abs_error_IV_Heston"] = np.abs(
    df["Residual_IV_Heston"]
)

valid = (
    np.isfinite(df["IV_Heston"])
    & np.isfinite(df["IV_market"])
)

RMSE = MET_H["RMSE"]
MAE = MET_H["MAE"]
MAX_ERROR = MET_H["MAX_ERROR"]

print()
print("============================")
print("FIT QUALITY")
print("============================")
print()
print(f"Number of valid IV = {MET_H['N']}")
print(f"RMSE IV             = {RMSE:.8f}")
print(f"MAE IV              = {MAE:.8f}")
print(f"Max IV error       = {MAX_ERROR:.8f}")


# ========================
# 16. BOUNDS DIAGNOSTIC
# ========================

print()
print("==========================")
print("BOUNDS DIAGNOSTIC")
print("==========================")
print()

for name, value, lower, upper in zip(
    ["v0", "kappa", "theta", "xi", "rho"],
    best_result.x,
    lower_bounds,
    upper_bounds
):
    print(
        f"{name:6s} = {value:.8f}"
        f" | lower bound = {lower:.4f}"
        f" | upper bound = {upper:.4f}"
        f" | écart upper bound = {upper - value:.8f}"
    )


# =======================================
# 17. MARKET IV / HESTON IV COMPARISON
# =======================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    df.loc[valid, "IV_market"],
    df.loc[valid, "IV_Heston"],
    s=25
)

iv_min = min(
    df.loc[valid, "IV_market"].min(),
    df.loc[valid, "IV_Heston"].min()
)

iv_max = max(
    df.loc[valid, "IV_market"].max(),
    df.loc[valid, "IV_Heston"].max()
)

plt.plot(
    [iv_min, iv_max],
    [iv_min, iv_max],
    linestyle="--",
    label="Perfect equality"
)

plt.xlabel(
    "Market implied volatility"
)

plt.ylabel(
    "Classical Heston implied volatility"
)

plt.title(
    "Market vs classical Heston"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

market_model_figure_path = (
    FIGURES_DIR
    / "04_market_vs_heston_global_calibration.png"
)

plt.savefig(
    market_model_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================
# 18. RESIDUALS BY MATURITY
# ============================

plt.figure(
    figsize=(9, 5)
)

plt.scatter(
    df.loc[valid, "T"],
    df.loc[valid, "Residual_IV_Heston"],
    s=25
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel(
    "Maturity T (years)"
)

plt.ylabel(
    "Heston IV - market IV"
)

plt.title(
    "Heston calibration residuals by maturity"
)

plt.grid(True)
plt.tight_layout()

maturity_figure_path = (
    FIGURES_DIR
    / "05_residuals_by_maturity_heston_global_calibration.png"
)

plt.savefig(
    maturity_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================
# 19. RESIDUALS BY FORWARD MONEYNESS
# =====================================

plt.figure(
    figsize=(9, 5)
)

plt.scatter(
    df.loc[valid, "K_over_F0"],
    df.loc[valid, "Residual_IV_Heston"],
    s=25
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel(
    r"Forward moneyness $K/F_0(T)$"
)

plt.ylabel(
    "Heston IV - market IV"
)

plt.title(
    "Heston calibration residuals by forward moneyness"
)

plt.grid(True)
plt.tight_layout()

moneyness_figure_path = (
    FIGURES_DIR
    / "06_residuals_by_forward_moneyness_heston_global_calibration.png"
)

plt.savefig(
    moneyness_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================
# 20. RESULTS EXPORT
# =====================

output_path = (
    CSV_DIR
    / "03_heston_global_calibration_results.csv"
)

parameters_path = (
    CSV_DIR
    / "04_heston_global_calibration_parameters.csv"
)

multistart_path = (
    CSV_DIR
    / "05_heston_multistart_results.csv"
)

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

parameters_df = pd.DataFrame([{
    "v0": v0_star,
    "kappa": kappa_star,
    "theta": theta_star,
    "xi": xi_star,
    "rho": rho_star,
    "RMSE": RMSE,
    "MAE": MAE,
    "MAX_ERROR": MAX_ERROR,
    "N_valid_IV": MET_H["N"],
    "Success": best_result.success,
    "NFEV": best_result.nfev,
    "Best_start": best_idx + 1,
    "Best_start_time_seconds": best_time,
    "Total_multistart_time_seconds": calibration_time,
    "N_cos": N_cos,
    "L": L
}])

parameters_df.to_csv(
    parameters_path,
    index=False,
    encoding="utf-8-sig"
)

multistart_rows = []

for start_number, (
    result,
    elapsed,
    IV_model,
    model_metrics
) in enumerate(heston_runs, start=1):

    multistart_rows.append({
        "Start": start_number,
        "v0_initial": HESTON_STARTS[start_number - 1][0],
        "kappa_initial": HESTON_STARTS[start_number - 1][1],
        "theta_initial": HESTON_STARTS[start_number - 1][2],
        "xi_initial": HESTON_STARTS[start_number - 1][3],
        "rho_initial": HESTON_STARTS[start_number - 1][4],
        "v0": result.x[0],
        "kappa": result.x[1],
        "theta": result.x[2],
        "xi": result.x[3],
        "rho": result.x[4],
        "RMSE": model_metrics["RMSE"],
        "MAE": model_metrics["MAE"],
        "MAX_ERROR": model_metrics["MAX_ERROR"],
        "N_valid_IV": model_metrics["N"],
        "Success": result.success,
        "NFEV": result.nfev,
        "Time_seconds": elapsed
    })

multistart_df = pd.DataFrame(
    multistart_rows
).sort_values(
    "RMSE"
).reset_index(
    drop=True
)

multistart_df.to_csv(
    multistart_path,
    index=False,
    encoding="utf-8-sig"
)

print()
print("===================")
print("EXPORT")
print("===================")
print()
print(
    "Detailed results saved to:",
    output_path
)
print(
    "Parameters and metrics saved to:",
    parameters_path
)
print(
    "Multistart results saved to:",
    multistart_path
)
print(
    "Figures saved to:",
    FIGURES_DIR
)
