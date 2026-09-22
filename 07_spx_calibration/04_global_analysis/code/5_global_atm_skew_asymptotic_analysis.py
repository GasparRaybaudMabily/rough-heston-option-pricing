"""
Global asymptotic analysis of the ATM skew.

This script does not recalibrate either model. It reads the ATM skew by
calendar horizon produced in the global analysis section and the H parameter
from the global Rough Heston calibration.

It studies the log-log relationship

    log |S(T)| = a + beta * log(T),

over four increasing maturity windows: 3–31, 3–94, 3–185, and 3–647 days.
For Rough Heston, the estimated slope beta is compared with the theoretical
short-maturity reference

    beta_theoretical = H - 1/2.

The relation S(T) ~ C T^(H-1/2) is asymptotic as T -> 0. The wider maturity
windows are therefore used as descriptive diagnostics of the stability of the
term structure rather than as strict tests of the short-maturity asymptotic.
The 1-day maturity is excluded from all regressions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========
# 1. PATHS
# ===========

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

CALIB_CSV_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "03_global_calibration"
    / "results"
)

ANALYSIS_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "04_global_analysis"
)

CSV_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SKEW_FILE = (
    CSV_DIR
    / "04_atm_skew_by_calendar_horizon.csv"
)

RH_PARAMS_FILE = (
    CALIB_CSV_DIR
    / "02_rough_heston_global_calibration_parameters.csv"
)

OUT_CSV = (
    CSV_DIR
    / "07_global_atm_skew_asymptotic_analysis.csv"
)

OUT_FIG_GLOBAL = (
    FIGURES_DIR
    / "09_global_atm_skew_loglog_analysis.png"
)


# ======================
# 2. ANALYSIS WINDOWS
# ======================

WINDOWS = [
    ("3–31 days", 3, 31),
    ("3–94 days", 3, 94),
    ("3–185 days", 3, 185),
    ("3–647 days", 3, 647),
]

SERIES = [
    ("Market", "Skew_Market"),
    ("Rough Heston", "Skew_Rough_Heston"),
    ("Classical Heston", "Skew_Heston"),
]


# =========================
# 3. LOADING GLOBAL SKEW
# =========================

if not SKEW_FILE.exists():
    raise FileNotFoundError(
        f"Skew file not found: {SKEW_FILE}"
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
        "Missing columns in the skew file: "
        + ", ".join(missing)
    )

df = df[required].copy()

df["Calendar_days"] = pd.to_numeric(
    df["Calendar_days"],
    errors="coerce",
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
    df.dropna(subset=["Calendar_days"])
    .copy()
)

df["Calendar_days"] = (
    df["Calendar_days"]
    .round()
    .astype(int)
)

df = (
    df.sort_values("Calendar_days")
    .reset_index(drop=True)
)

if df["Calendar_days"].duplicated().any():
    duplicated = df.loc[
        df["Calendar_days"].duplicated(keep=False),
        "Calendar_days",
    ].tolist()

    raise ValueError(
        "Multiple rows exist for some calendar horizons: "
        f"{duplicated}"
    )

available_days = sorted(
    df["Calendar_days"].unique().tolist()
)

print()
print("=" * 88)
print("GLOBAL ASYMPTOTIC ANALYSIS OF THE ATM SKEW")
print("=" * 88)
print("Available horizons:", available_days)


# ==============================================
# 4. ROBUST READING OF THE GLOBAL H PARAMETER
# ==============================================

def load_hurst_parameter(path):
    """
    Reads H from the two usual formats:
    1. one row with an H column;
    2. two columns of the Parameter / Value type.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Rough Heston parameter file not found: {path}"
        )

    p = pd.read_csv(path)

    normalized = {
        str(c).strip().lower(): c
        for c in p.columns
    }

    # Case 1: H is directly a column.
    if "h" in normalized:
        values = pd.to_numeric(
            p[normalized["h"]],
            errors="coerce",
        ).dropna()

        if not values.empty:
            return float(values.iloc[0])

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
            values = pd.to_numeric(
                p.loc[mask, value_col],
                errors="coerce",
            ).dropna()

            if not values.empty:
                return float(values.iloc[0])

    raise ValueError(
        f"Unable to read the H parameter from {path.name}. "
        f"Available columns: {list(p.columns)}"
    )


H_GLOBAL = load_hurst_parameter(
    RH_PARAMS_FILE
)

BETA_THEORETICAL_RH = (
    H_GLOBAL - 0.5
)

print(f"Global Rough Heston H = {H_GLOBAL:.6f}")
print(
    "Theoretical reference slope "
    f"H - 1/2 = {BETA_THEORETICAL_RH:+.6f}"
)


# ========================
# 5. LOG-LOG REGRESSION
# ========================

def loglog_fit(data, skew_col, model_name, min_days, max_days):
    """
    Fits:

        log |S(T)| = a + beta log(T)

    over horizons min_days <= Calendar_days <= max_days.

    T is expressed in years. Changing the unit modifies the intercept,
    but not the beta slope.
    """

    g = data[
        (data["Calendar_days"] >= min_days)
        & (data["Calendar_days"] <= max_days)
    ].copy()

    g = g[
        np.isfinite(g[skew_col])
        & (np.abs(g[skew_col]) > 0)
    ].copy()

    g = (
        g.sort_values("Calendar_days")
        .reset_index(drop=True)
    )

    if len(g) < 3:
        raise ValueError(
            f"Insufficient number of points for {model_name} "
            f"over {min_days}–{max_days} days: {len(g)}"
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

    h_implied = (
        beta + 0.5
    )

    result = {
        "Modele": model_name,
        "N": int(len(g)),
        "Jours_min": int(min_days),
        "Jours_max": int(max_days),
        "Jours_utilises": ",".join(
            str(int(x))
            for x in g["Calendar_days"]
        ),
        "Intercept": intercept,
        "C_estime": float(np.exp(intercept)),
        "Beta_estime": beta,
        "R2": float(r2),
        "H_implicite_beta_plus_1_2": float(h_implied),
    }

    return result, g


# ================================
# 6. COMPUTING THE FOUR WINDOWS
# ================================

results = []
fit_data = {}

for window_label, min_days, max_days in WINDOWS:

    fit_data[window_label] = {}

    for model_name, skew_col in SERIES:

        result, g = loglog_fit(
            df,
            skew_col,
            model_name,
            min_days,
            max_days,
        )

        result["Fenetre"] = window_label
        result["H_global_Rough_Heston"] = (
            H_GLOBAL
            if model_name == "Rough Heston"
            else np.nan
        )
        result["Beta_theorique_H_moins_1_2"] = (
            BETA_THEORETICAL_RH
            if model_name == "Rough Heston"
            else np.nan
        )
        result["Ecart_beta_estime_vs_theorique"] = (
            result["Beta_estime"]
            - BETA_THEORETICAL_RH
            if model_name == "Rough Heston"
            else np.nan
        )

        results.append(result)

        fit_data[window_label][model_name] = {
            "result": result,
            "data": g,
        }


summary = pd.DataFrame(
    results
)

summary = summary[
    [
        "Fenetre",
        "Modele",
        "N",
        "Jours_min",
        "Jours_max",
        "Jours_utilises",
        "Intercept",
        "C_estime",
        "Beta_estime",
        "R2",
        "H_implicite_beta_plus_1_2",
        "H_global_Rough_Heston",
        "Beta_theorique_H_moins_1_2",
        "Ecart_beta_estime_vs_theorique",
    ]
]

summary.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ====================
# 7. PLOTTING TOOLS
# ====================

def add_window_plot(
    ax,
    window_label,
    min_days,
    max_days,
    show_legend=True,
):
    """
    Plots the three series and their log-log regressions over a given window.
    Also adds the theoretical Rough Heston slope, normalized to the
    mean level of the Rough Heston series over the window.
    """

    window_results = fit_data[
        window_label
    ]

    for model_name, _ in SERIES:

        item = window_results[
            model_name
        ]

        result = item["result"]
        g = item["data"]

        T = g[
            "T_years"
        ].to_numpy(dtype=float)

        abs_skew = g[
            "Abs_Skew"
        ].to_numpy(dtype=float)

        ax.scatter(
            T,
            abs_skew,
            s=42,
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
            linewidth=1.9,
            label=(
                f"{model_name} "
                f"(β={result['Beta_estime']:.3f}, "
                f"R²={result['R2']:.3f})"
            ),
        )

    # Theoretical RH slope = H - 1/2.
    rh_item = window_results[
        "Rough Heston"
    ]

    rh_g = rh_item[
        "data"
    ]

    log_T_rh = np.log(
        rh_g["T_years"].to_numpy(dtype=float)
    )

    log_abs_rh = np.log(
        rh_g["Abs_Skew"].to_numpy(dtype=float)
    )

    # Normalization: the theoretical line passes through the center
    # of the Rough Heston observations in log-log coordinates.
    x_mean = float(
        np.mean(log_T_rh)
    )

    y_mean = float(
        np.mean(log_abs_rh)
    )

    theoretical_intercept = (
        y_mean
        - BETA_THEORETICAL_RH * x_mean
    )

    T_theory = np.geomspace(
        rh_g["T_years"].min(),
        rh_g["T_years"].max(),
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
            "Theoretical RH slope "
            f"(H={H_GLOBAL:.3f}, "
            f"β={BETA_THEORETICAL_RH:.3f})"
        ),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"Maturity $T$ (years)"
    )

    ax.set_ylabel(
        r"$|\mathcal{S}(T)|$"
    )

    ax.set_title(
        window_label
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    if show_legend:
        ax.legend(
            fontsize=8,
        )


# =========================
# 8. GLOBAL 2 x 2 FIGURE
# =========================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10.5),
)

axes = axes.ravel()

for ax, (
    window_label,
    min_days,
    max_days,
) in zip(
    axes,
    WINDOWS,
):
    add_window_plot(
        ax,
        window_label,
        min_days,
        max_days,
        show_legend=True,
    )

fig.suptitle(
    "Global log-log analysis of the ATM skew by maturity window",
    fontsize=16,
    y=0.995,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.965]
)

fig.savefig(
    OUT_FIG_GLOBAL,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ========================
# 9. INDIVIDUAL FIGURES
# ========================

individual_paths = []

for window_label, min_days, max_days in WINDOWS:

    fig, ax = plt.subplots(
        figsize=(9.5, 6.5),
    )

    add_window_plot(
        ax,
        window_label,
        min_days,
        max_days,
        show_legend=True,
    )

    ax.set_title(
        "Log-log analysis of the ATM skew - "
        + window_label
    )

    fig.tight_layout()

    clean_label = (
        window_label
        .replace("–", "_")
        .replace(" days", "_days")
    )

    figure_number = {
        "3–31 days": "10",
        "3–94 days": "11",
        "3–185 days": "12",
        "3–647 days": "13",
    }[window_label]

    out_fig = (
        FIGURES_DIR
        / (
            f"{figure_number}_global_atm_skew_loglog_"
            f"{clean_label}.png"
        )
    )

    fig.savefig(
        out_fig,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    individual_paths.append(
        out_fig
    )


# ====================
# 10. CONSOLE TABLE
# ====================

print()
print("=" * 88)
print("LOG-LOG REGRESSION RESULTS")
print("=" * 88)

display_columns = [
    "Fenetre",
    "Modele",
    "N",
    "Beta_estime",
    "R2",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("=" * 88)
print("ROUGH HESTON: COMPARISON WITH THE THEORETICAL SLOPE")
print("=" * 88)

rh_summary = summary[
    summary["Modele"] == "Rough Heston"
].copy()

print(
    rh_summary[
        [
            "Fenetre",
            "N",
            "Beta_estime",
            "Beta_theorique_H_moins_1_2",
            "Ecart_beta_estime_vs_theorique",
            "R2",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print(
    "Reminder: beta = H - 1/2 is a short-maturity asymptotic. "
    "The 3–31 day window is therefore the closest comparison "
    "to this theoretical interpretation. The wider windows "
    "are mainly used to study the descriptive stability "
    "of the slope as the horizon is progressively extended."
)

print()
print("=" * 88)
print("FILES SAVED")
print("=" * 88)
print(" -", OUT_CSV)
print(" -", OUT_FIG_GLOBAL)

for path in individual_paths:
    print(" -", path)

print()
print("Done.")
