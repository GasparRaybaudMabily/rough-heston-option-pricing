"""
Rough Heston: Adams Scheme validation in the Heston limit

This script validates the Adams predictor-corrector scheme used for the
Rough Heston fractional Riccati equation by considering the limiting
case H = 0.5. In this regime, the Rough Heston equation reduces to the
classical Heston Riccati equation, which is solved numerically with
solve_ivp and used as a reference solution.
"""


import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma
from scipy.integrate import solve_ivp

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
    Quadratic Riccati function common to the Heston
    and Rough Heston models.
    """

    return (
        -0.5 * (phi**2 + 1j * phi)
        + (1j * rho * nu * phi - kappa) * h
        + 0.5 * nu**2 * h**2
    )


# ====================================================
# 2. ROUGH HESTON: ADAMS PREDICTOR-CORRECTOR SCHEME
# ====================================================

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


# =======================================
# 3. CLASSICAL HESTON: RICCATI EQUATION
# =======================================

def solve_heston_riccati(phi, kappa, nu, rho, T, t_eval):
    """
    Solves the classical Heston Riccati equation

        B'(t) = F(phi, B(t)),
        B(0) = 0.

    The solution obtained with solve_ivp is used as the
    reference solution.
    """

    def heston_ode(t, y):

        # Reconstruct the complex-valued variable B
        B = y[0] + 1j * y[1]

        dB = riccati_F(
            phi,
            B,
            kappa,
            nu,
            rho
        )

        # solve_ivp operates here on two real-valued components
        return [dB.real, dB.imag]

    solution = solve_ivp(
        heston_ode,
        [0.0, T],
        [0.0, 0.0],
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12
    )

    B = solution.y[0] + 1j * solution.y[1]

    return B


# ===========================
# 4. EXPERIMENT PARAMETERS
# ===========================

phi = 1.0

kappa = 1.5
nu = 0.4
rho = -0.7

T = 1.0
N = 400

# Limiting case: Rough Heston -> classical Heston
H = 0.5


# =====================================
# 5. ROUGH HESTON SOLUTION WITH ADAMS
# =====================================

t, h_adams = solve_rough_riccati_adams(
    phi=phi,
    H=H,
    kappa=kappa,
    nu=nu,
    rho=rho,
    T=T,
    N=N
)


# =========================================
# 6. CLASSICAL HESTON REFERENCE SOLUTION
# =========================================

B_heston = solve_heston_riccati(
    phi=phi,
    kappa=kappa,
    nu=nu,
    rho=rho,
    T=T,
    t_eval=t
)


# =======================
# 7. ERROR COMPUTATION
# =======================

error = np.abs(h_adams - B_heston)

max_error = np.max(error)
final_error = error[-1]


print("============================================")
print("Adams validation for H = 1/2")
print("============================================")

print()
print("Final Adams value:")
print(h_adams[-1])

print()
print("Final classical Heston value:")
print(B_heston[-1])

print()
print("Error at maturity T:")
print(final_error)

print()
print("Maximum error over [0, T]:")
print(max_error)


# ==========================
# 8. REAL-PART COMPARISON
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    h_adams.real,
    label="Rough Heston - Adams (H=0.5)"
)

plt.plot(
    t,
    B_heston.real,
    "--",
    label="Classical Heston"
)

plt.xlabel("t")
plt.ylabel(r"$\Re(h(t,\varphi))$")
plt.title("Adams scheme validation: real part")

plt.legend()
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "3_adams_heston_limit_real_part.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)


# ===============================
# 9. IMAGINARY-PART COMPARISON
# ===============================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    h_adams.imag,
    label="Rough Heston - Adams (H=0.5)"
)

plt.plot(
    t,
    B_heston.imag,
    "--",
    label="Classical Heston"
)

plt.xlabel("t")
plt.ylabel(r"$\Im(h(t,\varphi))$")
plt.title("Adams scheme validation: imaginary part")

plt.legend()
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "4_adams_heston_limit_imaginary_part.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)

# ====================
# 10. ABSOLUTE ERROR
# ====================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    error
)

plt.xlabel("t")
plt.ylabel(r"$|h_{\mathrm{Adams}}(t)-B_{\mathrm{Heston}}(t)|$")
plt.title("Adams scheme validation: absolute error")

plt.grid(True)

plt.savefig(
    FIGURES_DIR / "5_adams_heston_limit_absolute_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)