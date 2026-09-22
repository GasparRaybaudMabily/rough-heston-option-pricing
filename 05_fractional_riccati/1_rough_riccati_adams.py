"""
Rough Heston: Fractional Riccati Equation with the Adams Scheme

This script solves the fractional Riccati-Volterra equation arising
in the Rough Heston model using an Adams predictor-corrector scheme.
The real and imaginary parts of the solution are then analyzed for
different values of the Hurst parameter.
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
    Solves the Rough Heston Riccati-Volterra equation using
    an Adams predictor-corrector scheme.

    Parameters
    ----------
    phi : float
        Fourier variable.
    H : float
        Hurst parameter.
    kappa : float
        Mean-reversion speed.
    nu : float
        Vol-of-vol.
    rho : float
        Price-variance correlation.
    T : float
        Maturity.
    N : int
        Number of time steps.

    Returns
    -------
    t : ndarray
        Time grid.
    h : ndarray
        Complex-valued approximation of h(t, phi).
    """

    alpha = H + 0.5
    dt = T / N

    t = np.linspace(0.0, T, N + 1)

    # The solution h is complex-valued because phi enters through i * phi
    h = np.zeros(N + 1, dtype=complex)

    # Initial condition
    h[0] = 0.0

    for n in range(N):

        # ================
        # Predictor step
        # ================

        predictor_sum = 0.0 + 0.0j

        for j in range(n + 1):
            b = (
                (n + 1 - j)**alpha
                - (n - j)**alpha
            )

            predictor_sum += b * riccati_F(
                phi,
                h[j],
                kappa,
                nu,
                rho
            )

        h_pred = (
            dt**alpha
            / gamma(alpha + 1)
            * predictor_sum
        )

        # ================
        # Corrector step
        # ================

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

            corrector_sum += a * riccati_F(
                phi,
                h[j],
                kappa,
                nu,
                rho
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


# =====================
# 3. MODEL PARAMETERS
# =====================

phi = 1.0
kappa = 1.5
nu = 0.4
rho = -0.7

T = 1.0
N = 500

H_values = [0.1, 0.3, 0.5, 0.7]

# ===============
# 4. REAL PART
# ===============

plt.figure(figsize=(8, 5))

for H in H_values:

    t, h = solve_rough_riccati_adams(
        phi=phi,
        H=H,
        kappa=kappa,
        nu=nu,
        rho=rho,
        T=T,
        N=N
    )

    plt.plot(
        t,
        np.real(h),
        label=rf"$H={H}$"
    )

plt.xlabel("t")
plt.ylabel(r"$\Re(h(t,\varphi))$")
plt.title(
    rf"Real part of $h(t,\varphi)$ for $\varphi={phi}$"
)

plt.legend()
plt.grid()

plt.savefig(
    FIGURES_DIR / "1_rough_riccati_real_part.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)


# ====================
# 5. IMAGINARY PART
# ====================

plt.figure(figsize=(8, 5))

for H in H_values:

    t, h = solve_rough_riccati_adams(
        phi=phi,
        H=H,
        kappa=kappa,
        nu=nu,
        rho=rho,
        T=T,
        N=N
    )

    plt.plot(
        t,
        np.imag(h),
        label=rf"$H={H}$"
    )

plt.xlabel("t")
plt.ylabel(r"$\Im(h(t,\varphi))$")
plt.title(
    rf"Imaginary part of $h(t,\varphi)$ for $\varphi={phi}$"
)

plt.legend()
plt.grid()

plt.savefig(
    FIGURES_DIR / "2_rough_riccati_imaginary_part.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure saved to:", FIGURES_DIR)


# ==================
# 6. FINAL VALUES
# ==================

print("Final values of h(T, phi):\n")

for H in H_values:

    t, h = solve_rough_riccati_adams(
        phi=phi,
        H=H,
        kappa=kappa,
        nu=nu,
        rho=rho,
        T=T,
        N=N
    )

    print(
        f"H = {H:.1f}  -->  "
        f"h(T,phi) = {h[-1]}"
    )