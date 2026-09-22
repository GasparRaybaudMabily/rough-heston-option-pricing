"""
SPX dataset overview.

This script provides a descriptive overview of the raw SPX option dataset,
including expiration dates, strikes, option roots, quote availability,
and the distribution of observations across expiration horizons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ===========================
# 1. SPX DATASET OVERVIEW
# ===========================

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

INPUT_PATH = RAW_DIR / "spx_quotedata.csv"

SUMMARY_PATH = CSV_DIR / "01_spx_dataset_summary.csv"
EXPIRATIONS_PATH = CSV_DIR / "02_spx_expiration_summary.csv"
FIGURE_PATH = FIGURES_DIR / "01_quotes_by_expiration.png"

REFERENCE_DATE = pd.Timestamp("2022-09-13")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Raw file not found:\n"
        f"{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "Expiration Date",
    "Strike",
    "Calls",
    "Puts",
    "BidC",
    "AskC",
    "BidP",
    "AskP",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        "Missing columns in spx_quotedata.csv: "
        + ", ".join(missing_columns)
    )

df["Expiration Date"] = pd.to_datetime(
    df["Expiration Date"],
    format="%d-%m-%y",
    errors="coerce"
)

df["T_days"] = (df["Expiration Date"] - REFERENCE_DATE).dt.days

df["RootC"] = (
    df["Calls"]
    .astype(str)
    .str.extract(r"^(SPXW|SPX)", expand=False)
)

df["RootP"] = (
    df["Puts"]
    .astype(str)
    .str.extract(r"^(SPXW|SPX)", expand=False)
)

df["Root"] = df["RootC"].fillna(df["RootP"])

n_rows = len(df)

valid_expirations = df["Expiration Date"].dropna()
n_expirations = valid_expirations.nunique()

expiration_min = valid_expirations.min() if len(valid_expirations) > 0 else pd.NaT
expiration_max = valid_expirations.max() if len(valid_expirations) > 0 else pd.NaT

n_unique_strikes = df["Strike"].nunique(dropna=True)
strike_min = df["Strike"].min()
strike_max = df["Strike"].max()

n_spx = int((df["Root"] == "SPX").sum())
n_spxw = int((df["Root"] == "SPXW").sum())
n_unknown_root = int(df["Root"].isna().sum())

n_calls_named = int(df["Calls"].notna().sum())
n_puts_named = int(df["Puts"].notna().sum())

n_valid_call_quotes = int(
    (
        (df["BidC"] > 0)
        & (df["AskC"] > 0)
        & (df["AskC"] >= df["BidC"])
    ).sum()
)

n_valid_put_quotes = int(
    (
        (df["BidP"] > 0)
        & (df["AskP"] > 0)
        & (df["AskP"] >= df["BidP"])
    ).sum()
)

summary_rows = [
    ("Total number of rows", n_rows),
    ("Number of expiration dates", n_expirations),
    (
        "First expiration date",
        expiration_min.strftime("%Y-%m-%d")
        if pd.notna(expiration_min) else ""
    ),
    (
        "Last expiration date",
        expiration_max.strftime("%Y-%m-%d")
        if pd.notna(expiration_max) else ""
    ),
    ("Number of distinct strikes", n_unique_strikes),
    ("Minimum strike", strike_min),
    ("Maximum strike", strike_max),
    ("Number of SPX rows", n_spx),
    ("Number of SPXW rows", n_spxw),
    ("Number of rows without identified Root", n_unknown_root),
    ("Number of Call contracts provided", n_calls_named),
    ("Number of Put contracts provided", n_puts_named),
    ("Valid Call Bid/Ask quotes", n_valid_call_quotes),
    ("Valid Put Bid/Ask quotes", n_valid_put_quotes),
]

optional_columns = {
    "Open InterestC": "Positive Call Open Interest",
    "Open InterestP": "Positive Put Open Interest",
    "IVC": "Call IV provided",
    "IVP": "Put IV provided",
}

for column, label in optional_columns.items():
    if column not in df.columns:
        continue

    numeric_col = pd.to_numeric(df[column], errors="coerce")

    if column.startswith("Open Interest"):
        value = int(numeric_col.gt(0).sum())
    else:
        value = int(numeric_col.notna().sum())

    summary_rows.append((label, value))

summary_df = pd.DataFrame(
    summary_rows,
    columns=["Indicator", "Value"]
)

expiration_summary = (
    df.dropna(subset=["Expiration Date"])
    .groupby("Expiration Date", as_index=False)
    .agg(
        T_days=("T_days", "first"),
        Nombre_lignes=("Strike", "size"),
        Nombre_strikes=("Strike", "nunique"),
        Strike_min=("Strike", "min"),
        Strike_max=("Strike", "max"),
        Nombre_SPX=("Root", lambda x: int((x == "SPX").sum())),
        Nombre_SPXW=("Root", lambda x: int((x == "SPXW").sum())),
    )
    .sort_values(["T_days", "Expiration Date"])
    .reset_index(drop=True)
)

expiration_summary["Expiration Date"] = (
    expiration_summary["Expiration Date"]
    .dt.strftime("%Y-%m-%d")
)

print()
print("========================")
print("RAW SPX DATASET OVERVIEW")
print("========================")
print()
print(f"File: {INPUT_PATH}")
print()

for _, row in summary_df.iterrows():
    print(f"{row['Indicator']:<40} : {row['Value']}")

print()
print("==========================")
print("SUMMARY BY EXPIRATION DATE")
print("==========================")
print()
print(expiration_summary.to_string(index=False))

plt.figure(figsize=(11, 6))
plt.bar(
    expiration_summary["T_days"],
    expiration_summary["Nombre_lignes"],
    width=8
)
plt.xlabel("Calendar horizon to expiration (days)")
plt.ylabel("Number of quotes")
plt.title("Distribution of quotes in the raw SPX dataset")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.show()

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)

expiration_summary.to_csv(
    EXPIRATIONS_PATH,
    index=False,
    encoding="utf-8-sig"
)

print()
print("===========")
print("SAVED FILES")
print("===========")
print()
print("Global summary:")
print(SUMMARY_PATH)
print()
print("Summary by expiration:")
print(EXPIRATIONS_PATH)
print()
print("Figure:")
print(FIGURE_PATH)
print()
