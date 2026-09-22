"""
Differential Evolution and TRF calibration test for the Rough Heston model.

This script first performs a global search with Differential Evolution and
then refines the resulting parameter set with the Trust Region Reflective
(TRF) least-squares algorithm. The final Rough Heston implied volatilities
are compared with market implied volatilities and calibration residuals are
analyzed across maturity and forward moneyness.
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
from scipy.optimize import differential_evolution


# ====================================
# 1. LOADING THE CALIBRATION SAMPLE
# ====================================

from pathlib import Path

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

S0 = 3936.710926

# r is read by maturity from the r_GSW column

q = 0.0114029141


print()
print("GSW market specification:")
print(f"S0 = {S0:.6f}")
print(f"q  = {100.0 * q:.6f} %")
print(
    f"r_GSW min/max = {100.0 * df['r_GSW'].min():.6f} % / "
    f"{100.0 * df['r_GSW'].max():.6f} %"
)
print("Construction moneyness = K/F0(T)")
print()

# ==========================
# 3. NUMERICAL PARAMETERS
# ==========================

# Deliberately moderate resolution for the optimization stage.
#
# A check with a finer discretization will then be performed.

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
    # CUMULANTS
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
    # CHARACTERISTIC FUNCTION
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


    weights_cos = np.ones(
        N_cos
    )

    weights_cos[0] = 0.5


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
                weights_cos
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
                    model_price = (
                        call_price
                        - S0 * np.exp(-q * T)
                        + K * discount_T
                    )
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


# ===========================
# 15. RESIDUAL CALCULATION
# ===========================

def compute_residuals(
    params
):

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
    # PENALTY IN CASE OF NUMERICAL FAILURE
    # ========================================================

    invalid = (
        ~np.isfinite(
            residual
        )
    )


    residual[
        invalid
    ] = 1.0


    residual = (
        np.sqrt(weights)
        * residual
    )


    return residual


# =======================================
# 16. SCALAR OBJECTIVE FUNCTION FOR DE
# =======================================

de_counter = {
    "n": 0
}


def objective_de(
    params
):

    de_counter["n"] += 1

    n_eval = (
        de_counter["n"]
    )


    start = (
        time.time()
    )


    residual = (
        compute_residuals(
            params
        )
    )


    mse = np.mean(
        residual**2
    )


    rmse = np.sqrt(
        mse
    )


    elapsed = (
        time.time()
        - start
    )


    # Avoid displaying every single evaluation.
    # Display one evaluation out of 10.

    if (
        n_eval == 1
        or n_eval % 10 == 0
    ):

        V0, kappa, theta, nu, rho, H = params

        print()
        print(
            f"DE - evaluation {n_eval:4d}"
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


    return mse


# =======================
# 17. NUMERICAL BOUNDS
# =======================

# WARNING:
#
# These are NUMERICAL search bounds,
# not additional theoretical constraints of the model.

bounds_de = [

    (0.005, 0.20),   # V0

    (0.05, 5.00),    # kappa

    (0.005, 0.20),   # theta

    (0.05, 2.00),    # nu

    (-0.99, 0.99),   # rho

    (0.02, 0.49)     # H
]


lower_bounds = np.array(
    [
        bound[0]
        for bound in bounds_de
    ]
)


upper_bounds = np.array(
    [
        bound[1]
        for bound in bounds_de
    ]
)


# =============================
# 18. DIFFERENTIAL EVOLUTION
# =============================

print()
print("============================================================")
print("STEP 1: DIFFERENTIAL EVOLUTION")
print("============================================================")
print()

print(
    "Global search over the numerical domain..."
)

print()


de_start = (
    time.time()
)


result_de = differential_evolution(
    func=objective_de,

    bounds=bounds_de,

    strategy="best1bin",

    # --------------------------------------------------------
    # FIRST TEST:
    # keep the computational cost deliberately reasonable.
    # --------------------------------------------------------

    maxiter=8,

    popsize=5,

    tol=1e-3,

    mutation=(0.5, 1.0),

    recombination=0.7,

    # No automatic local refinement:
    # TRF is performed explicitly just afterwards.
    polish=False,

    seed=123,

    workers=1,

    updating="immediate",

    disp=True
)


de_time = (
    time.time()
    - de_start
)


theta_de = (
    result_de.x
)


rmse_de = np.sqrt(
    result_de.fun
)


print()
print("============================================================")
print("DIFFERENTIAL EVOLUTION RESULT")
print("============================================================")
print()

print(
    f"V0     = {theta_de[0]:.8f}"
)

print(
    f"kappa  = {theta_de[1]:.8f}"
)

print(
    f"theta  = {theta_de[2]:.8f}"
)

print(
    f"nu     = {theta_de[3]:.8f}"
)

print(
    f"rho    = {theta_de[4]:.8f}"
)

print(
    f"H      = {theta_de[5]:.8f}"
)

print()

print(
    f"RMSE DE = {rmse_de:.8f}"
)

print(
    f"DE time = {de_time:.2f} seconds"
)

print(
    f"Number of DE calls = {de_counter['n']}"
)


# ==================
# 19. TRF COUNTER
# ==================

trf_counter = {
    "n": 0
}


# ========================
# 20. RESIDUALS FOR TRF
# ========================

def residuals_trf(
    params
):

    trf_counter["n"] += 1

    n_eval = (
        trf_counter["n"]
    )


    start = (
        time.time()
    )


    residual = (
        compute_residuals(
            params
        )
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
        f"TRF - evaluation {n_eval:4d}"
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


    return residual


# ================================
# 21. LOCAL REFINEMENT WITH TRF
# ================================

print()
print("============================================================")
print("STEP 2: LOCAL TRF REFINEMENT")
print("============================================================")
print()

print(
    "Initial point provided by Differential Evolution:"
)

print(
    theta_de
)


trf_start = (
    time.time()
)


result_trf = least_squares(
    fun=residuals_trf,

    x0=theta_de,

    bounds=(
        lower_bounds,
        upper_bounds
    ),

    method="trf",

    jac="2-point",

    xtol=1e-6,

    ftol=1e-6,

    gtol=1e-6,

    max_nfev=40,

    verbose=2
)


trf_time = (
    time.time()
    - trf_start
)


theta_star = (
    result_trf.x
)


# =======================
# 22. FINAL PARAMETERS
# =======================

V0_star = theta_star[0]

kappa_star = theta_star[1]

theta_star_value = theta_star[2]

nu_star = theta_star[3]

rho_star = theta_star[4]

H_star = theta_star[5]


print()
print("============================================================")
print("FINAL DE + TRF PARAMETERS")
print("============================================================")
print()

print(
    f"V0     = {V0_star:.8f}"
)

print(
    f"kappa  = {kappa_star:.8f}"
)

print(
    f"theta  = {theta_star_value:.8f}"
)

print(
    f"nu     = {nu_star:.8f}"
)

print(
    f"rho    = {rho_star:.8f}"
)

print(
    f"H      = {H_star:.8f}"
)

print()

print(
    "TRF convergence:",
    result_trf.success
)

print(
    "TRF message:",
    result_trf.message
)

print(
    "TRF NFEV:",
    result_trf.nfev
)

print(
    f"TRF time: {trf_time:.2f} seconds"
)

print(
    f"Total DE + TRF time: "
    f"{de_time + trf_time:.2f} seconds"
)


# =================================
# 23. FINAL ROUGH HESTON SURFACE
# =================================

IV_RH = (
    rough_heston_iv_surface(
        theta_star
    )
)


df[
    "IV_RH"
] = IV_RH


df[
    "Residual_IV"
] = (
    df["IV_RH"]
    - df["IV_market"]
)


df[
    "Abs_error_IV"
] = np.abs(
    df["Residual_IV"]
)


# =======================
# 24. ERROR STATISTICS
# =======================

valid = (
    np.isfinite(
        df["IV_RH"]
    )

    & np.isfinite(
        df["IV_market"]
    )
)


errors = (
    df.loc[
        valid,
        "Residual_IV"
    ]
    .to_numpy()
)


RMSE = np.sqrt(
    np.mean(
        errors**2
    )
)


MAE = np.mean(
    np.abs(
        errors
    )
)


MAX_ERROR = np.max(
    np.abs(
        errors
    )
)


MSE = np.mean(
    errors**2
)


print()
print("============================================================")
print("FINAL FIT QUALITY")
print("============================================================")
print()

print(
    f"MSE IV         = {MSE:.8f}"
)

print(
    f"RMSE IV        = {RMSE:.8f}"
)

print(
    f"MAE IV         = {MAE:.8f}"
)

print(
    f"Max IV error    = {MAX_ERROR:.8f}"
)


# ======================================
# 25. COMPARISON WITH MULTI-START TRF
# ======================================

multi_start_path = (
    CSV_DIR
    / "01_trf_multistart_results.csv"
)

print()
print("============================================================")
print("COMPARISON WITH MULTI-START TRF")
print("============================================================")
print()

if multi_start_path.exists():
    df_previous = pd.read_csv(multi_start_path)
    if "RMSE" in df_previous.columns and len(df_previous) > 0:
        previous_rmse = df_previous["RMSE"].min()
        print(f"Best multi-start RMSE       = {previous_rmse:.8f}")
        print(f"RMSE DE + TRF              = {RMSE:.8f}")
        print(f"Difference                  = {RMSE - previous_rmse:.8f}")
    else:
        print("The multi-start file does not contain a usable RMSE.")
else:
    print("Multi-start file not found: comparison skipped.")
# =============================================
# 26. MARKET IV / ROUGH HESTON IV COMPARISON
# =============================================

plt.figure(
    figsize=(7, 7)
)


plt.scatter(
    df.loc[
        valid,
        "IV_market"
    ],

    df.loc[
        valid,
        "IV_RH"
    ],

    s=25
)


iv_min = min(
    df.loc[
        valid,
        "IV_market"
    ].min(),

    df.loc[
        valid,
        "IV_RH"
    ].min()
)


iv_max = max(
    df.loc[
        valid,
        "IV_market"
    ].max(),

    df.loc[
        valid,
        "IV_RH"
    ].max()
)


plt.plot(
    [
        iv_min,
        iv_max
    ],

    [
        iv_min,
        iv_max
    ],

    linestyle="--",
    label="Perfect equality"
)


plt.xlabel(
    "Market implied volatility"
)

plt.ylabel(
    "Rough Heston implied volatility"
)

plt.title(
    "Market vs Rough Heston"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "01_market_vs_rough_heston_de_trf.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================
# 27. RESIDUALS BY MATURITY
# ============================

plt.figure(
    figsize=(9, 5)
)


plt.scatter(
    df.loc[
        valid,
        "T"
    ],

    df.loc[
        valid,
        "Residual_IV"
    ],

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
    "Rough Heston IV - market IV"
)

plt.title(
    "Calibration residuals by maturity"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "02_residuals_by_maturity_de_trf.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================
# 28. RESIDUALS BY STRIKE
# ==========================

plt.figure(
    figsize=(9, 5)
)


plt.scatter(
    df.loc[valid, "K_over_F0"],

    df.loc[
        valid,
        "Residual_IV"
    ],

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
    "Rough Heston IV - market IV"
)

plt.title(
    "Calibration residuals by forward moneyness"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "03_residuals_by_forward_moneyness_de_trf.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ========================================
# 29. EXPORT OF CALIBRATED OBSERVATIONS
# ========================================

output_path = (
    CSV_DIR
    / "02_de_trf_detailed_results.csv"
)




df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)


# =======================
# 30. PARAMETER EXPORT
# =======================

parameters_output_path = (
    CSV_DIR
    / "03_de_trf_parameters.csv"
)


df_parameters = pd.DataFrame(
    {
        "Parameter": [
            "V0",
            "kappa",
            "theta",
            "nu",
            "rho",
            "H"
        ],

        "DE": [
            theta_de[0],
            theta_de[1],
            theta_de[2],
            theta_de[3],
            theta_de[4],
            theta_de[5]
        ],

        "DE_TRF": [
            V0_star,
            kappa_star,
            theta_star_value,
            nu_star,
            rho_star,
            H_star
        ]
    }
)


df_parameters.to_csv(
    parameters_output_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("============================================================")
print("EXPORT")
print("============================================================")
print()

print(
    "Detailed results saved to:"
)

print(
    output_path
)

print()

print(
    "Parameters saved to:"
)

print(
    parameters_output_path
)
