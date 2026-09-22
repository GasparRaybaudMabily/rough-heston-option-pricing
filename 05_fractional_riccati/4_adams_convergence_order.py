"""
Rough Heston: Convergence order of the Adams Scheme

This script studies the numerical convergence of the Adams
predictor-corrector scheme used to solve the Rough Heston
fractional Riccati equation. The observed convergence order is
estimated from solutions computed on successively refined grids
and compared with the corresponding theoretical order.
"""


import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma

from pathlib import Path

# =================
# PROJECT PATHS
# =================

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

FIGURES_DIR = (
    ROOT_DIR
    / "05_fractional_riccati"
    / "figures"
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# 1. RICCATI QUADRATIC FUNCTION
# ===============================

def riccati_F(phi, h, kappa, nu, rho):
    """
    Quadratic Riccati function associated with the Rough Heston model.
    """

    return (
        -0.5 * (phi**2 + 1j * phi)
        + (1j * rho * nu * phi - kappa) * h
        + 0.5 * nu**2 * h**2
    )


# ==============================================
# 2. ROUGH HESTON RICCATI SOLVER: ADAMS SCHEME
# ==============================================

def solve_rough_riccati_adams(phi, H, kappa, nu, rho, T, N):
    """
    Solves the Rough Heston Riccati-Volterra equation
    using the Adams predictor-corrector scheme.
    """

    alpha = H + 0.5
    dt = T / N

    t = np.linspace(0.0, T, N + 1)

    # Complex-valued solution
    h = np.zeros(N + 1, dtype=complex)

    # Initial condition
    h[0] = 0.0

    for n in range(N):


        # Predictor step: Adams-Bashforth

        predictor_sum = 0.0 + 0.0j

        for j in range(n + 1):

            b = (
                (n + 1 - j)**alpha
                - (n - j)**alpha
            )

            predictor_sum += (
                b
                * riccati_F(
                    phi,
                    h[j],
                    kappa,
                    nu,
                    rho
                )
            )

        h_pred = (
            dt**alpha
            / gamma(alpha + 1)
            * predictor_sum
        )


        # Corrector step: Adams-Moulton

        corrector_sum = 0.0 + 0.0j

        for j in range(n + 1):

            if j == 0:

                a = (
                    n**(alpha + 1)
                    - (n - alpha) * (n + 1)**alpha
                )

            else:

                a = (
                    (n - j + 2)**(alpha + 1)
                    + (n - j)**(alpha + 1)
                    - 2 * (n - j + 1)**(alpha + 1)
                )

            corrector_sum += (
                a
                * riccati_F(
                    phi,
                    h[j],
                    kappa,
                    nu,
                    rho
                )
            )

        h[n + 1] = (
            dt**alpha
            / gamma(alpha + 2)
            * (
                riccati_F(
                    phi,
                    h_pred,
                    kappa,
                    nu,
                    rho
                )
                + corrector_sum
            )
        )

    return t, h


# ================================
# 3. OBSERVED CONVERGENCE ORDER
# ================================

def compute_observed_order(
    phi,
    H,
    kappa,
    nu,
    rho,
    T,
    N
):
    """
    Computes the observed convergence order using three grids:

        N
        2N
        4N

    according to

        p_obs = log(D_N / D_2N) / log(2)
    """

    # Solution on grid N

    t_N, h_N = solve_rough_riccati_adams(
        phi,
        H,
        kappa,
        nu,
        rho,
        T,
        N
    )


    # Solution on grid 2N

    t_2N, h_2N = solve_rough_riccati_adams(
        phi,
        H,
        kappa,
        nu,
        rho,
        T,
        2 * N
    )


    # Solution on grid 4N

    t_4N, h_4N = solve_rough_riccati_adams(
        phi,
        H,
        kappa,
        nu,
        rho,
        T,
        4 * N
    )

    # =====================================
    # 4. Comparison at common time points
    # =====================================

    # On the 2N grid, the points corresponding to the N grid are:
    #
    #     0, 2, 4, ..., 2N

    h_2N_on_N = h_2N[::2]

    # On the 4N grid, the points corresponding to the N grid are:
    #
    #     0, 4, 8, ..., 4N

    h_4N_on_N = h_4N[::4]

    # ========================
    # 5. Computation of D_N
    # ========================

    D_N = np.max(
        np.abs(
            h_N
            - h_2N_on_N
        )
    )

    # ========================
    # 6. Computation of D_2N
    # ========================

    # Compare h^(2N) and h^(4N) on the N-grid points
    # so that both norms are evaluated at identical time points.

    D_2N = np.max(
        np.abs(
            h_2N_on_N
            - h_4N_on_N
        )
    )

    # ====================
    # 7. Observed order
    # ====================

    p_obs = (
        np.log(D_N / D_2N)
        / np.log(2.0)
    )

    return D_N, D_2N, p_obs


# ======================
# 8. MODEL PARAMETERS
# ======================

phi = 1.0

kappa = 1.5
nu = 0.4
rho = -0.7

T = 1.0


# =====================
# 9. HURST PARAMETERS
# =====================

H_values = [
    0.1,
    0.5,
    0.7
]


# =================
# 10. GRID SIZES
# =================

N_values = [
    25,
    50,
    100,
    200,
    400
]


# =====================
# 11. RESULT STORAGE
# =====================

results = {}


# ===================
# 12. COMPUTATIONS
# ===================

for H in H_values:

    alpha = H + 0.5

    # Theoretical order
    p_theoretical = min(
        2.0,
        1.0 + alpha
    )

    results[H] = {
        "N": [],
        "D_N": [],
        "D_2N": [],
        "p_obs": [],
        "p_theoretical": p_theoretical
    }

    print()
    print("====================================================")
    print(f"H = {H}")
    print(f"alpha = {alpha}")
    print(f"Theoretical order = {p_theoretical}")
    print("====================================================")

    print()
    print(
        f"{'N':>8}"
        f"{'D_N':>20}"
        f"{'D_2N':>20}"
        f"{'p_obs':>15}"
    )

    print("-" * 63)

    for N in N_values:

        D_N, D_2N, p_obs = compute_observed_order(
            phi=phi,
            H=H,
            kappa=kappa,
            nu=nu,
            rho=rho,
            T=T,
            N=N
        )

        results[H]["N"].append(N)
        results[H]["D_N"].append(D_N)
        results[H]["D_2N"].append(D_2N)
        results[H]["p_obs"].append(p_obs)

        print(
            f"{N:8d}"
            f"{D_N:20.8e}"
            f"{D_2N:20.8e}"
            f"{p_obs:15.6f}"
        )


# ======================================
# 13. SUMMARY OF FINAL OBSERVED ORDERS
# ======================================

print()
print()
print("===========")
print("SUMMARY")
print("===========")

print()

print(
    f"{'H':>8}"
    f"{'alpha':>12}"
    f"{'p_th':>12}"
    f"{'p_obs final':>18}"
)

print("-" * 50)

for H in H_values:

    alpha = H + 0.5

    p_theoretical = results[H]["p_theoretical"]

    p_final = results[H]["p_obs"][-1]

    print(
        f"{H:8.2f}"
        f"{alpha:12.2f}"
        f"{p_theoretical:12.4f}"
        f"{p_final:18.6f}"
    )


# ====================================================
# 14. OBSERVED CONVERGENCE ORDER AS A FUNCTION OF N
# ====================================================

plt.figure(figsize=(8, 5))

for H in H_values:

    plt.plot(
        results[H]["N"],
        results[H]["p_obs"],
        marker="o",
        label=f"H = {H}"
    )

plt.xlabel("N")
plt.ylabel(r"$p_{\mathrm{obs}}$")
plt.title("Observed convergence order of the Adams scheme")

plt.legend()
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "9_adams_observed_convergence_order.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)


# ====================================
# 15. OBSERVED VS THEORETICAL ORDER
# ====================================

plt.figure(figsize=(8, 5))

for H in H_values:

    N_array = np.array(
        results[H]["N"]
    )

    p_theoretical = results[H]["p_theoretical"]

    plt.plot(
        N_array,
        results[H]["p_obs"],
        marker="o",
        label=f"Observed, H = {H}"
    )

    plt.plot(
        N_array,
        np.full_like(
            N_array,
            p_theoretical,
            dtype=float
        ),
        linestyle="--",
        label=f"Theoretical, H = {H}"
    )

plt.xlabel("N")
plt.ylabel(r"Order $p$")
plt.title("Observed and theoretical convergence orders")

plt.legend()
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "10_observed_vs_theoretical_convergence_order.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)