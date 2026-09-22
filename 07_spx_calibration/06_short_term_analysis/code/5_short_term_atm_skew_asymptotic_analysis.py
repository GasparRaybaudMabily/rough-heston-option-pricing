"""
Chapter 7 - Short-term analysis
Short-term ATM skew asymptotic analysis.

This script does not recalibrate any model. It reads the short-term ATM skew
produced by the previous analysis and the Rough Heston short-term calibration
parameters. It studies the log-log relation |Skew(T)| ~ C T^(H-1/2), excluding
the atypical one-day market observation from the regressions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========
# 1. PATHS
# ===========

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

CALIB_CSV_DIR = ROOT_DIR / "07_spx_calibration" / "05_short_term_calibration" / "results"

ANALYSIS_DIR = ROOT_DIR / "07_spx_calibration" / "06_short_term_analysis"
CSV_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SKEW_FILE = (
    CSV_DIR
    / "04_short_term_atm_skew_by_calendar_horizon.csv"
)

RH_PARAMS_FILE = (
    CALIB_CSV_DIR
    / "02_rough_heston_short_term_calibration_parameters.csv"
)

OUT_CSV = (
    CSV_DIR
    / "07_short_term_atm_skew_asymptotic_analysis.csv"
)

OUT_FIG = (
    FIGURES_DIR
    / "09_short_term_atm_skew_asymptotic_analysis.png"
)


# =========================
# 2. ANALYSIS PARAMETERS
# =========================

# Same window for all three regressions to preserve comparability.
FIT_DAYS = [3, 7, 15, 31]

# The one-day point is retained in the figure for information only,
# but is not included in any regression.
EXCLUDED_DAY = 1


# ===================
# 3. LOAD ATM SKEW
# ===================

if not SKEW_FILE.exists():
    raise FileNotFoundError(
        f"ATM skew file not found: {SKEW_FILE}"
    )

df = pd.read_csv(SKEW_FILE)

required = [
    "Calendar_days",
    "Skew_Market",
    "Skew_Rough_Heston",
    "Skew_Heston",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns in ATM skew file: "
        + ", ".join(missing)
    )

df = df[required].copy()

df["Calendar_days"] = (
    pd.to_numeric(df["Calendar_days"], errors="coerce")
    .round()
    .astype("Int64")
)

for col in [
    "Skew_Market",
    "Skew_Rough_Heston",
    "Skew_Heston",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

df = (
    df.dropna()
    .copy()
)

df["Calendar_days"] = df["Calendar_days"].astype(int)

actual_days = sorted(
    df["Calendar_days"].unique().tolist()
)

expected_days = [1, 3, 7, 15, 31]

if actual_days != expected_days:
    raise ValueError(
        f"Unexpected horizons: {actual_days}. "
        f"Expected: {expected_days}"
    )


# =======================================
# 4. LOAD THE ROUGH HESTON H PARAMETER
# =======================================

def load_hurst_parameter(path):
    """
    Read H from the usual parameter CSV formats:
    - a row containing an H column;
    - two columns of the Parameter / Value type.
    """

    if not path.exists():
        return np.nan

    p = pd.read_csv(path)

    normalized = {
        str(c).strip().lower(): c
        for c in p.columns
    }

    # Case 1: H is directly available as a column.
    if "h" in normalized:
        value = pd.to_numeric(
            p[normalized["h"]],
            errors="coerce",
        ).dropna()

        if not value.empty:
            return float(value.iloc[0])

    # Case 2: Parameter / Value format.
    name_candidates = [
        c for c in p.columns
        if str(c).strip().lower()
        in {
            "paramètre",
            "parametre",
            "parameter",
            "param",
            "nom",
        }
    ]

    value_candidates = [
        c for c in p.columns
        if str(c).strip().lower()
        in {
            "valeur",
            "value",
            "estimate",
            "estimation",
        }
    ]

    if name_candidates and value_candidates:
        name_col = name_candidates[0]
        value_col = value_candidates[0]

        names = (
            p[name_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mask = names == "h"

        if mask.any():
            value = pd.to_numeric(
                p.loc[mask, value_col],
                errors="coerce",
            ).dropna()

            if not value.empty:
                return float(value.iloc[0])

    return np.nan


H_CALIBRATED = load_hurst_parameter(
    RH_PARAMS_FILE
)

BETA_THEORETICAL_RH = (
    H_CALIBRATED - 0.5
    if np.isfinite(H_CALIBRATED)
    else np.nan
)


# ========================
# 5. LOG-LOG REGRESSION
# ========================

def loglog_fit(data, skew_col, model_name):
    """
    Fit:

        log |Skew(T)| = intercept + beta * log(T)

    over the FIT_DAYS horizons.

    T is expressed in years. Using days instead does not change beta;
    it only changes the intercept.
    """

    g = (
        data[
            data["Calendar_days"].isin(FIT_DAYS)
        ]
        .copy()
        .sort_values("Calendar_days")
    )

    g = g[
        np.isfinite(g[skew_col])
        & (np.abs(g[skew_col]) > 0)
    ].copy()

    if len(g) < 3:
        raise ValueError(
            f"Insufficient number of points for {model_name}."
        )

    g["T_years"] = (
        g["Calendar_days"].astype(float)
        / 365.25
    )

    g["Abs_Skew"] = np.abs(
        g[skew_col].astype(float)
    )

    g["Log_T"] = np.log(
        g["T_years"]
    )

    g["Log_Abs_Skew"] = np.log(
        g["Abs_Skew"]
    )

    x = g["Log_T"].to_numpy(dtype=float)
    y = g["Log_Abs_Skew"].to_numpy(dtype=float)

    X = np.column_stack([
        np.ones(len(x)),
        x,
    ])

    coef, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    intercept = float(coef[0])
    beta = float(coef[1])

    fitted = X @ coef

    ss_res = float(
        np.sum(
            (y - fitted) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    # Implied H associated with a beta slope when imposing
    # the relation beta = H - 1/2.
    h_implied = beta + 0.5

    return {
        "Modele": model_name,
        "N": int(len(g)),
        "Jours_utilises": ",".join(
            str(int(x))
            for x in g["Calendar_days"]
        ),
        "Intercept": intercept,
        "Beta_estime": beta,
        "R2": float(r2),
        "H_implicite_beta_plus_1_2": float(h_implied),
        "C_estime": float(np.exp(intercept)),
    }, g


summary_market, fit_market = loglog_fit(
    df,
    "Skew_Market",
    "Marché",
)

summary_rh, fit_rh = loglog_fit(
    df,
    "Skew_Rough_Heston",
    "Rough Heston",
)

summary_heston, fit_heston = loglog_fit(
    df,
    "Skew_Heston",
    "Heston classique",
)


# ========================================================
# 6. COMPARISON WITH THE THEORETICAL ROUGH HESTON SLOPE
# ========================================================

summary = pd.DataFrame([
    summary_market,
    summary_rh,
    summary_heston,
])

summary["H_calibre_Rough_Heston"] = np.nan
summary["Beta_theorique_H_moins_1_2"] = np.nan
summary["Ecart_beta_estime_vs_theorique"] = np.nan

if np.isfinite(H_CALIBRATED):
    rh_mask = (
        summary["Modele"]
        == "Rough Heston"
    )

    summary.loc[
        rh_mask,
        "H_calibre_Rough_Heston"
    ] = H_CALIBRATED

    summary.loc[
        rh_mask,
        "Beta_theorique_H_moins_1_2"
    ] = BETA_THEORETICAL_RH

    summary.loc[
        rh_mask,
        "Ecart_beta_estime_vs_theorique"
    ] = (
        summary.loc[
            rh_mask,
            "Beta_estime"
        ]
        - BETA_THEORETICAL_RH
    )


# ================
# 7. CSV EXPORT
# ================

summary.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ====================
# 8. LOG-LOG FIGURE
# ====================

fig, ax = plt.subplots(
    figsize=(9.5, 6.5),
)

series = [
    (
        "Marché",
        "Skew_Market",
        summary_market,
        fit_market,
    ),
    (
        "Rough Heston",
        "Skew_Rough_Heston",
        summary_rh,
        fit_rh,
    ),
    (
        "Heston classique",
        "Skew_Heston",
        summary_heston,
        fit_heston,
    ),
]

for model_name, skew_col, result, g in series:

    T = g["T_years"].to_numpy(dtype=float)
    abs_skew = g["Abs_Skew"].to_numpy(dtype=float)

    ax.scatter(
        T,
        abs_skew,
        s=55,
        label=f"{model_name} - observations",
        zorder=4,
    )

    T_line = np.geomspace(
        T.min(),
        T.max(),
        200,
    )

    fitted_line = np.exp(
        result["Intercept"]
        + result["Beta_estime"]
        * np.log(T_line)
    )

    ax.plot(
        T_line,
        fitted_line,
        linewidth=2.0,
        label=(
            f"{model_name} - fit "
            f"(β={result['Beta_estime']:.3f}, "
            f"R²={result['R2']:.3f})"
        ),
    )


# Theoretical slope from calibrated H: same normalization as the
# Rough Heston fit, so that only the slopes are compared.
if np.isfinite(BETA_THEORETICAL_RH):

    T_rh = fit_rh["T_years"].to_numpy(dtype=float)

    # Pass the theoretical curve through the mean log level
    # of the RH regression to avoid comparing intercepts.
    x_mean = float(
        np.mean(
            np.log(T_rh)
        )
    )

    y_mean = float(
        np.mean(
            np.log(
                fit_rh["Abs_Skew"]
            )
        )
    )

    theoretical_intercept = (
        y_mean
        - BETA_THEORETICAL_RH * x_mean
    )

    T_theory = np.geomspace(
        T_rh.min(),
        T_rh.max(),
        200,
    )

    theoretical_curve = np.exp(
        theoretical_intercept
        + BETA_THEORETICAL_RH
        * np.log(T_theory)
    )

    ax.plot(
        T_theory,
        theoretical_curve,
        linestyle=":",
        linewidth=2.0,
        label=(
            "Theoretical Rough Heston slope "
            f"(H={H_CALIBRATED:.4f}, "
            f"β=H-1/2={BETA_THEORETICAL_RH:.3f})"
        ),
    )


# One-day market point displayed separately, outside the fit.
one_day = df[
    df["Calendar_days"]
    == EXCLUDED_DAY
].copy()

if not one_day.empty:

    skew_1d = float(
        one_day["Skew_Market"].iloc[0]
    )

    if np.isfinite(skew_1d) and skew_1d != 0:

        T_1d = (
            EXCLUDED_DAY
            / 365.25
        )

        ax.scatter(
            [T_1d],
            [abs(skew_1d)],
            s=85,
            marker="x",
            zorder=6,
            label="Market 1 day - excluded from fit",
        )


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    "Maturity $T$ (years, logarithmic scale)"
)

ax.set_ylabel(
    r"$|\mathcal{S}(T)|$ "
    "(logarithmic scale)"
)

ax.set_title(
    "Log-log analysis of short-term ATM skew"
)

ax.grid(
    True,
    which="both",
    alpha=0.25,
)

ax.legend(
    fontsize=9,
)

fig.tight_layout()

fig.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =====================
# 9. CONSOLE SUMMARY
# =====================

print()
print("=" * 86)
print("ATM SKEW ASYMPTOTIC ANALYSIS - LOG-LOG REGRESSION")
print("=" * 86)

print()
print(
    "Regression performed over horizons:",
    FIT_DAYS,
)

print(
    "The one-day maturity is excluded from the fits."
)

print()

for _, row in summary.iterrows():

    print(
        f"{row['Modele']:<18s} | "
        f"beta = {row['Beta_estime']:+.6f} | "
        f"R² = {row['R2']:.6f} | "
        f"H implicite = {row['H_implicite_beta_plus_1_2']:.6f}"
    )

if np.isfinite(H_CALIBRATED):

    print()
    print("ROUGH HESTON - COMPARISON WITH THEORY")
    print("-" * 86)

    print(
        f"Calibrated H                       = {H_CALIBRATED:.6f}"
    )

    print(
        f"theoretical beta = H - 1/2        = {BETA_THEORETICAL_RH:+.6f}"
    )

    print(
        f"estimated beta over 3-31 days     = {summary_rh['Beta_estime']:+.6f}"
    )

    print(
        "estimated - theoretical beta gap  = "
        f"{summary_rh['Beta_estime'] - BETA_THEORETICAL_RH:+.6f}"
    )

print()
print("Saved files:")
print(" -", OUT_CSV)
print(" -", OUT_FIG)

print()
print(
    "Interpretation: the log-log regression is an empirical diagnostic "
    "of the skew term structure over a limited number of maturities. "
    "It should not be interpreted as empirical proof of the asymptotic "
    "law."
)

print()
print("Done.")
