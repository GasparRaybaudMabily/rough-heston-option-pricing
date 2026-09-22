"""
Rough Heston: Gil-Pelaez and COS pricing comparison

This script prices European call options under the Rough Heston model
using two Fourier-based approaches: the Gil-Pelaez inversion formula
and the COS method. Both methods are compared in terms of numerical
stability, pricing accuracy and computational performance.
"""


import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.special import gamma

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


# =================================
# 1. RICCATI QUADRATIC FUNCTION
# =================================

def riccati_F(phi, h, kappa, nu, rho):
    """
    Quadratic Riccati function associated with the Rough Heston model.

    phi may be either a complex scalar or a complex array.
    """

    return (
        -0.5 * (phi**2 + 1j * phi)
        + (1j * rho * nu * phi - kappa) * h
        + 0.5 * nu**2 * h**2
    )


# ==============================
# 2. VECTORIZED ADAMS SOLVER
# ==============================

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

    Returns
    -------
    t : ndarray
        Time grid.
    h : ndarray
        Complex-valued array of shape (number of phi values, N + 1).
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

    # ----------------------
    # Time-stepping loop
    # ----------------------

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

        # Coefficient for j = 0
        a[0] = (
            n**(alpha + 1)
            - (n - alpha) * (n + 1)**alpha
        )

        # Coefficients for j >= 1
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
    S0,
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
    Computes Phi_T(phi) for multiple Fourier arguments.

    Convention
    ----------
    X_T = log(S_T / S0)

    so that

    Phi_T(phi) = E[exp(i phi X_T)].
    """

    if not (0.0 < H < 0.5):
        raise ValueError(
            "This characteristic-function implementation "
            "is intended for the rough regime 0 < H < 1/2."
            )

    phi_values = np.atleast_1d(
        np.asarray(phi_values, dtype=complex)
    )

    alpha = H + 0.5

    # ------------------
    # Solve h(t, phi)
    # ------------------

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

    # ================================
    # 3.1 Classical integral of h
    #     trapezoidal rule
    # ================================

    integral_h = dt * (
        0.5 * h[:, 0]
        + np.sum(h[:, 1:-1], axis=1)
        + 0.5 * h[:, -1]
    )

    # ============================================
    # 3.2 Riemann-Liouville fractional integral
    # ============================================

    # We use:
    #
    # 1 / Gamma(2-alpha)
    # sum h_j [
    #   (T-t_j)^(1-alpha)
    #   - (T-t_{j+1})^(1-alpha)
    # ]

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

    # ===============================
    # 3.3 Characteristic function
    # ===============================

    exponent = (
        1j * phi_values * r * T
        + V0 * fractional_integral
        + kappa * theta * integral_h
    )

    Phi = np.exp(exponent)

    return Phi


# ========================
# 4. GIL-PELAEZ PRICING
# ========================

def price_call_gil_pelaez(
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
    u_max,
    n_u
):
    """
    Prices a European call option using the Gil-Pelaez inversion formula.
    """

    start = time.perf_counter()

    k = np.log(K / S0)

    # Avoid u = 0 because the formulas contain 1 / u
    u_min = 1e-4

    u = np.linspace(
        u_min,
        u_max,
        n_u
    )

    # ------------------------------
    # Required Fourier arguments
    # ------------------------------

    phi_all = np.concatenate(
        [
            u.astype(complex),
            u.astype(complex) - 1j,
            np.array([-1j])
        ]
    )

    Phi_all = rough_heston_characteristic(
        phi_values=phi_all,
        S0=S0,
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

    # ---------------------------------------
    # Split characteristic-function outputs
    # ---------------------------------------

    Phi_u = Phi_all[:n_u]

    Phi_u_minus_i = Phi_all[
        n_u:2 * n_u
    ]

    Phi_minus_i = Phi_all[-1]

    # ----------------
    # P2 integrand
    # ----------------

    integrand_P2 = np.imag(
        np.exp(-1j * u * k)
        * Phi_u
        / u
    )

    # ----------------
    # P1 integrand
    # ----------------

    integrand_P1 = np.imag(
        np.exp(-1j * u * k)
        * Phi_u_minus_i
        / (
            u * Phi_minus_i
        )
    )

    # ------------
    # Quadrature
    # ------------

    P1 = (
        0.5
        + (1.0 / np.pi)
        * np.trapz(
            integrand_P1,
            u
        )
    )

    P2 = (
        0.5
        + (1.0 / np.pi)
        * np.trapz(
            integrand_P2,
            u
        )
    )

    # -------------
    # Call price
    # -------------

    price = (
        S0 * P1
        - K * np.exp(-r * T) * P2
    )

    elapsed = time.perf_counter() - start

    return {
        "price": float(np.real(price)),
        "P1": float(np.real(P1)),
        "P2": float(np.real(P2)),
        "time": elapsed,
        "Phi_minus_i": Phi_minus_i
    }


# ====================================
# 5. NUMERICAL CUMULANT ESTIMATION
# ====================================

def estimate_cumulants(
    S0,
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
    Estimates c1, c2 and c4 from

        K(z) = log Phi_T(-i z)

    using a local polynomial fit around z = 0.

    This step is used only to construct the truncation interval [a, b]
    for the COS method.
    """

    z = np.linspace(
        -radius,
        radius,
        n_points
    )

    phi = -1j * z

    Phi = rough_heston_characteristic(
        phi_values=phi,
        S0=S0,
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

    # Cumulant-generating function
    K_values = np.real(
        np.log(Phi)
    )

    # Polynomial fit:
    #
    # K(z) =
    # c1 z
    # + c2 z^2 / 2!
    # + c3 z^3 / 3!
    # + c4 z^4 / 4!
    # + ...
    #
    coeff = np.polynomial.polynomial.polyfit(
        z,
        K_values,
        deg=6
    )

    c1 = coeff[1]
    c2 = 2.0 * coeff[2]
    c4 = 24.0 * coeff[4]

    return c1, c2, c4


# =============================================
# 6. CHI AND PSI FUNCTIONS FOR THE COS PAYOFF
# =============================================

def chi_cos(j, a, b, c, d):
    """
    chi_j(c,d)
    """

    u = j * np.pi / (b - a)

    numerator_d = (
        np.cos(u * (d - a))
        + u * np.sin(u * (d - a))
    )

    numerator_c = (
        np.cos(u * (c - a))
        + u * np.sin(u * (c - a))
    )

    return (
        np.exp(d) * numerator_d
        - np.exp(c) * numerator_c
    ) / (1.0 + u**2)


def psi_cos(j, a, b, c, d):
    """
    psi_j(c,d)
    """

    u = j * np.pi / (b - a)

    if j == 0:
        return d - c

    return (
        np.sin(u * (d - a))
        - np.sin(u * (c - a))
    ) / u


# ================
# 7. COS PRICING
# ================

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
    L=10.0,
    cumulants=None
):
    """
    Prices a European call option using the COS method.
    """

    start = time.perf_counter()

    k = np.log(K / S0)

    # ===============
    # 7.1 Cumulants
    # ===============

    if cumulants is None:

        c1, c2, c4 = estimate_cumulants(
            S0=S0,
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

    else:

        c1, c2, c4 = cumulants

    # -----------------------
    # Numerical safeguards
    # -----------------------

    c2 = max(
        float(np.real(c2)),
        1e-12
    )

    c4_positive = max(
        float(np.real(c4)),
        0.0
    )

    width = L * np.sqrt(
        c2
        + np.sqrt(c4_positive)
    )

    a = c1 - width
    b = c1 + width

    # ======================
    # 7.2 COS frequencies
    # ======================

    j_values = np.arange(
        N_cos
    )

    u = (
        j_values
        * np.pi
        / (b - a)
    )

    # ===============================
    # 7.3 Characteristic function
    # ===============================

    Phi = rough_heston_characteristic(
        phi_values=u,
        S0=S0,
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

    # =========================
    # 7.4 Payoff coefficients
    # =========================

    G = np.zeros(
        N_cos
    )

    # The call payoff is positive for x > k
    c = max(k, a)
    d = b

    if c >= b:

        # The payoff is zero over the entire interval [a, b]
        G[:] = 0.0

    else:

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

    # ===========================
    # 7.5 Density coefficients
    # ===========================
    density_coeff = np.real(
        Phi
        * np.exp(
            -1j * u * a
        )
    )

    # Primed summation:
    # the j = 0 term is multiplied by 1/2
    weights = np.ones(
        N_cos
    )

    weights[0] = 0.5

    # ===========
    # 7.6 Price
    # ===========

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

    elapsed = time.perf_counter() - start

    return {
        "price": float(np.real(price)),
        "time": elapsed,
        "a": a,
        "b": b,
        "c1": c1,
        "c2": c2,
        "c4": c4
    }


# ==========================
# 8. EXPERIMENT PARAMETERS
# ==========================

S0 = 100.0
K = 100.0

V0 = 0.04
theta = 0.04

kappa = 1.5
nu = 0.30
rho = -0.70

H = 0.10

r = 0.02
T = 1.0

# Number of Adams time steps
N_time = 250


# ==========================
# 9. CUMULANT COMPUTATION
# ==========================

print()
print("Computing COS cumulants...")
cumulants = estimate_cumulants(
    S0=S0,
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

c1, c2, c4 = cumulants

print()
print("Estimated cumulants:")
print(f"c1 = {c1:.8f}")
print(f"c2 = {c2:.8f}")
print(f"c4 = {c4:.8f}")


# ===============================
# 10. GIL-PELAEZ STABILITY TEST
# ===============================

u_max_values = [
    40.0,
    60.0,
    80.0,
    100.0
]

gp_results = []

print()
print("==========================")
print("GIL-PELAEZ STABILITY")
print("==========================")
print()

print(
    f"{'u_max':>10}"
    f"{'Price':>18}"
    f"{'Time (s)':>18}"
)

print("-" * 46)

for u_max in u_max_values:

    # Keep approximately the same Fourier-grid spacing
    # as u_max increases.
    n_u = int(
        4 * u_max
    )

    result = price_call_gil_pelaez(
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
        u_max=u_max,
        n_u=n_u
    )

    gp_results.append(
        result
    )

    print(
        f"{u_max:10.1f}"
        f"{result['price']:18.10f}"
        f"{result['time']:18.4f}"
    )


# =========================
# 11. COS STABILITY TEST
# =========================

N_cos_values = [
    32,
    64,
    128,
    256
]

cos_results = []

print()
print("===================")
print("COS STABILITY")
print("===================")
print()

print(
    f"{'N_COS':>10}"
    f"{'Price':>18}"
    f"{'Time (s)':>18}"
)

print("-" * 46)

for N_cos in N_cos_values:

    result = price_call_cos(
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
        L=10.0,
        cumulants=cumulants
    )

    cos_results.append(
        result
    )

    print(
        f"{N_cos:10d}"
        f"{result['price']:18.10f}"
        f"{result['time']:18.4f}"
    )


# ========================
# 12. FINAL COMPARISON
# ========================

C_GP = gp_results[-1]["price"]
C_COS = cos_results[-1]["price"]

time_GP = gp_results[-1]["time"]
time_COS = cos_results[-1]["time"]

absolute_error = abs(
    C_GP - C_COS
)

relative_error = (
    absolute_error
    / abs(C_GP)
)


print()
print("======================")
print("FINAL COMPARISON")
print("======================")

print()
print(f"Gil-Pelaez price : {C_GP:.10f}")
print(f"COS price        : {C_COS:.10f}")

print()
print(
    f"Absolute error   : "
    f"{absolute_error:.10e}"
)

print(
    f"Relative error   : "
    f"{relative_error:.10e}"
)

print()
print(
    f"Gil-Pelaez time  : "
    f"{time_GP:.4f} s"
)

print(
    f"COS time          : "
    f"{time_COS:.4f} s"
)


# ========================
# 13. MARTINGALE CHECK
# ========================

Phi_minus_i = gp_results[-1][
    "Phi_minus_i"
]

martingale_target = np.exp(
    r * T
)

martingale_error = abs(
    Phi_minus_i
    - martingale_target
)

print()
print("===================")
print("MARTINGALE CHECK")
print("===================")

print()
print(
    "Phi_T(-i)        =",
    Phi_minus_i
)

print(
    "exp(rT)          =",
    martingale_target
)

print(
    "Martingale error  =",
    martingale_error
)


# ================================
# 14. GIL-PELAEZ STABILITY PLOT
# ================================
plt.figure(figsize=(8, 5))

plt.plot(
    u_max_values,
    [result["price"] for result in gp_results],
    marker="o"
)

plt.xlabel(r"Truncation bound $u_{\max}$")
plt.ylabel("Call price")
plt.title(
    "Numerical stability of the Gil-Pelaez method"
)

# Avoid scientific-offset formatting on the y-axis "1e-5 + 8.5619"
plt.ticklabel_format(
    axis="y",
    style="plain",
    useOffset=False
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "1_gil_pelaez_stability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)


# =========================
# 15. COS STABILITY PLOT
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    N_cos_values,
    [result["price"] for result in cos_results],
    marker="o"
)

plt.xlabel(r"Number of terms $N_{\mathrm{COS}}$")
plt.ylabel("Call price")
plt.title(
    "Numerical stability of the COS method"
)

# Use the same presentation for both figures
plt.ticklabel_format(
    axis="y",
    style="plain",
    useOffset=False
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "2_cos_stability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)