"""
Chapter 7 - Section 7.1.4
ATM implied volatility skew observed in the market.

The script reads the final calibration sample (398 options) constructed
using the GSW / Svensson curve and forward moneyness K/F0(T).

For each effective maturity, the ATM skew is estimated locally by:

    IV(k,T) = a(T) + Skew(T) * k,
    k = log(K/F0(T)),

using the N_LOCAL observations closest to k=0.

For calendar horizons containing two effective maturities
(SPX / SPXW), the skews are then aggregated using a weighted average based
on the number of local points used.

The main plot retains the 1-day maturity but displays it
separately, without connecting it to the other maturities, so as not to hide
its atypical nature.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ===========
# 1. PATHS
# ===========

ROOT_DIR = Path.cwd() / "rough-heston-option-pricing"

RAW_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "00_raw_data"
)

PREP_DIR = (
    ROOT_DIR
    / "07_spx_calibration"
    / "01_data_preparation"
)

CSV_DIR = PREP_DIR / "results"
FIGURES_DIR = PREP_DIR / "figures"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


file_path = (
    CSV_DIR
    / "05_calibration_sample.csv"
)

effective_path = (
    CSV_DIR
    / "07_market_atm_skew_by_effective_maturity.csv"
)

calendar_path = (
    CSV_DIR
    / "08_market_atm_skew_by_calendar_horizon.csv"
)

figure_path = (
    FIGURES_DIR
    / "09_market_atm_skew.png"
)


# ================
# 2. PARAMETERS
# ================

# Maximum number of points closest to forward ATM used
# for the local regression at each effective maturity.
N_LOCAL = 7

# Minimum number of points required to estimate a slope.
MIN_LOCAL = 3


# ==============================
# 3. LOADING THE FINAL SAMPLE
# ==============================

if not file_path.exists():
    raise FileNotFoundError(
        f"File not found: {file_path}\n"
        "Check that the final forward sample has been created."
    )

df = pd.read_csv(file_path)

required = ["T", "K_over_F0", "IV_market"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        "Missing columns in the input file: " + ", ".join(missing)
    )

# Calendar horizon: T_days is used when available.
if "T_days" in df.columns:
    df["Calendar_days"] = df["T_days"].astype(int)
elif "Calendar_days" not in df.columns:
    df["Calendar_days"] = np.rint(df["T"] * 365.25).astype(int)

if "Root" not in df.columns:
    df["Root"] = "NA"

valid = (
    np.isfinite(df["T"])
    & np.isfinite(df["K_over_F0"])
    & np.isfinite(df["IV_market"])
    & (df["T"] > 0)
    & (df["K_over_F0"] > 0)
    & (df["IV_market"] > 0)
)

df = df.loc[valid].copy()
df["log_forward_moneyness"] = np.log(df["K_over_F0"])

print("=" * 72)
print("MARKET ATM SKEW - FINAL CALIBRATION SAMPLE")
print("=" * 72)
print("File:", file_path)
print("Valid observations:", len(df))
print("Calendar horizons:", df["Calendar_days"].nunique())
print("Effective maturities:", df["T"].nunique())
print("Maximum number of local points:", N_LOCAL)


# ===============================
# 4. LOCAL ATM SKEW ESTIMATION
# ===============================

def estimate_local_skew(group, n_local=N_LOCAL):
    """
    Locally estimates:
        sigma_impl = intercept + skew * log(K/F0)
    using the points closest to forward ATM.
    """

    g = group[
        np.isfinite(group["log_forward_moneyness"])
        & np.isfinite(group["IV_market"])
    ].copy()

    if len(g) < MIN_LOCAL:
        return {
            "Skew_ATM": np.nan,
            "IV_ATM_estimee": np.nan,
            "R2_local": np.nan,
            "N_local": len(g),
            "K_over_F0_min_local": np.nan,
            "K_over_F0_max_local": np.nan,
            "Brackets_ATM": False,
        }

    g["abs_k"] = np.abs(g["log_forward_moneyness"])
    g = g.sort_values("abs_k").head(min(n_local, len(g))).copy()
    g = g.sort_values("log_forward_moneyness")

    x = g["log_forward_moneyness"].to_numpy(dtype=float)
    y = g["IV_market"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = coef

    fitted = X @ coef
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "Skew_ATM": float(slope),
        "IV_ATM_estimee": float(intercept),
        "R2_local": float(r2) if np.isfinite(r2) else np.nan,
        "N_local": int(len(g)),
        "K_over_F0_min_local": float(np.exp(np.min(x))),
        "K_over_F0_max_local": float(np.exp(np.max(x))),
        "Brackets_ATM": bool(np.min(x) <= 0.0 <= np.max(x)),
    }


rows = []

for T, group in df.groupby("T", sort=True):
    result = estimate_local_skew(group)

    calendar_days = int(group["Calendar_days"].iloc[0])
    roots = ",".join(sorted(group["Root"].astype(str).unique()))

    rows.append({
        "T": float(T),
        "Effective_days": float(T) * 365.25,
        "Calendar_days": calendar_days,
        "Root": roots,
        "N_total": int(len(group)),
        **result,
    })

skew_effective = pd.DataFrame(rows)

print()
print("=" * 72)
print("ATM SKEW BY EFFECTIVE MATURITY")
print("=" * 72)
print(
    skew_effective[
        [
            "Effective_days",
            "Calendar_days",
            "Root",
            "N_local",
            "K_over_F0_min_local",
            "K_over_F0_max_local",
            "Brackets_ATM",
            "Skew_ATM",
            "R2_local",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
)


# =====================================
# 5. AGGREGATION BY CALENDAR HORIZON
# =====================================

# When a horizon contains two effective maturities (SPX and SPXW),
# weighted average by the number of points used in each regression.
calendar_rows = []

for days, group in skew_effective.groupby("Calendar_days", sort=True):
    values = group["Skew_ATM"].to_numpy(dtype=float)
    weights = group["N_local"].to_numpy(dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if np.any(mask):
        skew_calendar = float(np.average(values[mask], weights=weights[mask]))
    else:
        skew_calendar = np.nan

    calendar_rows.append({
        "Calendar_days": int(days),
        "N_effective_maturities": int(len(group)),
        "N_local_total": int(np.sum(weights[mask])) if np.any(mask) else 0,
        "Skew_ATM": skew_calendar,
        "Abs_Skew_ATM": abs(skew_calendar) if np.isfinite(skew_calendar) else np.nan,
    })

skew_calendar = pd.DataFrame(calendar_rows)

print()
print("=" * 72)
print("ATM SKEW BY CALENDAR HORIZON")
print("=" * 72)
print(
    skew_calendar.to_string(index=False, float_format=lambda x: f"{x:.6f}")
)


# ===========================================
# 6. FIGURES - MARKET ATM SKEW BY MATURITY
# ===========================================

regular = skew_calendar[
    np.isfinite(skew_calendar["Skew_ATM"])
    & (skew_calendar["Calendar_days"] >= 3)
].copy()

short = skew_calendar[
    np.isfinite(skew_calendar["Skew_ATM"])
    & (skew_calendar["Calendar_days"] <= 31)
].copy()

one_day = short[
    short["Calendar_days"] == 1
].copy()

short_regular = short[
    short["Calendar_days"] >= 3
].copy()


# ------------------------------------------------
# Figure 1: global structure from 3 to 647 days
# ------------------------------------------------

fig_global, ax_global = plt.subplots(figsize=(8, 5.8))

ax_global.plot(
    regular["Calendar_days"],
    regular["Skew_ATM"],
    marker="o",
    linewidth=1.8,
)

ax_global.axhline(
    0.0,
    linewidth=0.9,
    linestyle="--",
    alpha=0.7,
)

ax_global.set_xlabel("Calendar horizon (days)")
ax_global.set_ylabel(
    r"Skew ATM $\left.\partial \sigma_{\mathrm{impl}}/\partial"
    r"\log(K/F_0)\right|_{k=0}$"
)
ax_global.set_title("Global ATM skew structure (3--647 days)")
ax_global.grid(alpha=0.25)

fig_global.tight_layout()

fig_global.savefig(
    FIGURES_DIR / "09_market_atm_skew_global_structure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -------------------------------------
# Figure 2: zoom on short maturities
# -------------------------------------

fig_short, ax_short = plt.subplots(figsize=(8, 5.8))

ax_short.plot(
    short_regular["Calendar_days"],
    short_regular["Skew_ATM"],
    marker="o",
    linewidth=1.8,
    label="3--31 days",
)

if not one_day.empty:
    ax_short.scatter(
        one_day["Calendar_days"],
        one_day["Skew_ATM"],
        s=70,
        marker="o",
        label="1 day (atypical)",
        zorder=5,
    )

    x1 = float(one_day["Calendar_days"].iloc[0])
    y1 = float(one_day["Skew_ATM"].iloc[0])

    ax_short.annotate(
        "1 day",
        xy=(x1, y1),
        xytext=(8, 8),
        textcoords="offset points",
    )

ax_short.axhline(
    0.0,
    linewidth=0.9,
    linestyle="--",
    alpha=0.7,
)

ax_short.set_xlabel("Calendar horizon (days)")
ax_short.set_ylabel("Skew ATM")
ax_short.set_title("Zoom on short maturities")
ax_short.set_xticks([1, 3, 7, 15, 31])
ax_short.grid(alpha=0.25)
ax_short.legend()

fig_short.tight_layout()

fig_short.savefig(
    FIGURES_DIR / "10_market_atm_skew_short_maturities.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============
# 7. EXPORT
# ============

skew_effective.to_csv(
    effective_path,
    index=False,
    encoding="utf-8-sig"
)

skew_calendar.to_csv(
    calendar_path,
    index=False,
    encoding="utf-8-sig"
)

print(" -", effective_path)
print(" -", calendar_path)
print(" -", figure_path)
