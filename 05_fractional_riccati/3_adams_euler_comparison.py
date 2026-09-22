"""
Rough Heston: Adams, Fractional Euler and Heston comparison

This script compares two numerical schemes for the Rough Heston
fractional Riccati equation: the Adams predictor-corrector method
and a fractional Euler scheme. In the limiting case H = 0.5,
both numerical solutions are compared with the classical Heston
Riccati solution used as a reference.
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


# ===================================================
# 2. ROUGH HESTON: ADAMS PREDICTOR-CORRECTOR SCHEME
# ===================================================

def solve_rough_riccati_adams(phi, H, kappa, nu, rho, T, N):

    alpha = H + 0.5
    dt = T / N

    t = np.linspace(0.0, T, N + 1)

    h = np.zeros(N + 1, dtype=complex)
    h[0] = 0.0

    for n in range(N):

        # Adams-Bashforth predictor

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


        # Adams-Moulton corrector

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


# ===========================================
# 3. ROUGH HESTON: FRACTIONAL EULER SCHEME
# ===========================================

def solve_rough_riccati_euler(phi, H, kappa, nu, rho, T, N):

    alpha = H + 0.5
    dt = T / N

    t = np.linspace(0.0, T, N + 1)

    h = np.zeros(N + 1, dtype=complex)
    h[0] = 0.0

    for n in range(N):

        euler_sum = 0.0 + 0.0j

        for j in range(n + 1):

            b = (
                (n + 1 - j)**alpha
                - (n - j)**alpha
            )

            euler_sum += (
                b
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
            / gamma(alpha + 1)
            * euler_sum
        )

    return t, h


# ==========================================
# 4. CLASSICAL HESTON: REFERENCE SOLUTION
# ==========================================

def solve_heston_riccati(phi, kappa, nu, rho, T, t_eval):

    def heston_ode(t, y):

        B = y[0] + 1j * y[1]

        dB = riccati_F(
            phi,
            B,
            kappa,
            nu,
            rho
        )

        return [dB.real, dB.imag]

    solution = solve_ivp(
        heston_ode,
        [0.0, T],
        [0.0, 0.0],
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-13
    )

    B = solution.y[0] + 1j * solution.y[1]

    return B


# ===========================
# 5. EXPERIMENT PARAMETERS
# ===========================

phi = 1.0

kappa = 1.5
nu = 0.4
rho = -0.7

T = 1.0
N = 400

# Limiting case: Rough Heston -> classical Heston
H = 0.5


# ===================
# 6. ADAMS SOLUTION
# ===================

t, h_adams = solve_rough_riccati_adams(
    phi=phi,
    H=H,
    kappa=kappa,
    nu=nu,
    rho=rho,
    T=T,
    N=N
)


# ==============================
# 7. FRACTIONAL EULER SOLUTION
# ==============================

t_euler, h_euler = solve_rough_riccati_euler(
    phi=phi,
    H=H,
    kappa=kappa,
    nu=nu,
    rho=rho,
    T=T,
    N=N
)


# ===============================
# 8. CLASSICAL HESTON SOLUTION
# ===============================

B_heston = solve_heston_riccati(
    phi=phi,
    kappa=kappa,
    nu=nu,
    rho=rho,
    T=T,
    t_eval=t
)


# =======================
# 9. ERROR COMPUTATION
# =======================

error_adams = np.abs(h_adams - B_heston)
error_euler = np.abs(h_euler - B_heston)

max_error_adams = np.max(error_adams)
max_error_euler = np.max(error_euler)

final_error_adams = error_adams[-1]
final_error_euler = error_euler[-1]


print("====================================================")
print("Adams / Euler / classical Heston comparison")
print("====================================================")

print()
print("Parameters:")
print(f"H     = {H}")
print(f"phi   = {phi}")
print(f"N     = {N}")
print(f"T     = {T}")

print()
print("Final classical Heston value:")
print(B_heston[-1])

print()
print("Final Adams value:")
print(h_adams[-1])

print()
print("Final fractional Euler value:")
print(h_euler[-1])

print()
print("----------------------------------------------------")

print("Maximum Adams error:")
print(max_error_adams)

print("Maximum Euler error:")
print(max_error_euler)

print()
print("Final Adams error:")
print(final_error_adams)

print("Final Euler error:")
print(final_error_euler)

print()
print("Euler / Adams error ratio:")
print(max_error_euler / max_error_adams)


# ==========================
# 10. REAL-PART COMPARISON
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    B_heston.real,
    "--",
    label="Classical Heston"
)

plt.plot(
    t,
    h_adams.real,
    label="Adams"
)

plt.plot(
    t,
    h_euler.real,
    label="Fractional Euler"
)

plt.xlabel("t")
plt.ylabel(r"$\Re(h(t,\varphi))$")
plt.title("Adams, Euler and Heston comparison: real part")

plt.legend()
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "6_adams_euler_heston_real_part.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)


# ===============================
# 11. IMAGINARY-PART COMPARISON
# ===============================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    B_heston.imag,
    "--",
    label="Classical Heston"
)

plt.plot(
    t,
    h_adams.imag,
    label="Adams"
)

plt.plot(
    t,
    h_euler.imag,
    label="Fractional Euler"
)

plt.xlabel("t")
plt.ylabel(r"$\Im(h(t,\varphi))$")
plt.title("Adams, Euler and Heston comparison: imaginary part")

plt.legend()
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "7_adams_euler_heston_imaginary_part.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)


# =============================================
# 12. ERROR COMPARISON ON A LOGARITHMIC SCALE
# =============================================

plt.figure(figsize=(8, 5))

# Exclude t = 0 since both errors are exactly zero
plt.semilogy(
    t[1:],
    error_adams[1:],
    label="Adams error"
)

plt.semilogy(
    t[1:],
    error_euler[1:],
    label="Euler error"
)

plt.xlabel("t")
plt.ylabel("Absolute error")
plt.title("Numerical error comparison")

plt.legend()
plt.grid(True)

plt.savefig(
    FIGURES_DIR / "8_adams_euler_heston_absolute_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)