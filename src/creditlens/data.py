from __future__ import annotations

import importlib.resources
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "default"

RENAME = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "past_due_90_plus",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}
FEATURES = [c for c in RENAME.values() if c != TARGET]
PAST_DUE = ["past_due_30_59", "past_due_60_89", "past_due_90_plus"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CSV = PROJECT_ROOT / "data" / "cs-training.csv"


def load_raw():
    # use kaggle's cs-training.csv if it's in data/, otherwise the copy inside the empulse package
    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
    else:
        try:
            path = importlib.resources.files("empulse.datasets") / "data" / "creditscoring1.csv.gz"
            df = pd.read_csv(path)
        except (ModuleNotFoundError, FileNotFoundError) as e:
            raise FileNotFoundError(
                f"no data found - put cs-training.csv in {LOCAL_CSV.parent} or pip install empulse==0.13.0"
            ) from e
    df = df.drop(columns=[c for c in df.columns if c in ("id", "Unnamed: 0")])
    return df.rename(columns=RENAME)


def clean(df):
    df = df.drop_duplicates().copy()
    df = df[df["age"] >= 18]  # there's one row with age = 0
    # 96 and 98 show up in the past due columns, they're codes not actual counts
    for c in PAST_DUE:
        df.loc[df[c] >= 96, c] = float("nan")
    # not capping outliers, the binning and lightgbm both deal with them fine
    return df.reset_index(drop=True)


def load_splits(seed=42):
    df = clean(load_raw())
    # 60/20/20 stratified
    train, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=seed)
    valid, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=seed)
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
