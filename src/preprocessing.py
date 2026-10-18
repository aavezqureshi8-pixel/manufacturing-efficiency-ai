"""
preprocessing.py - cleaning, chronological splitting and the reusable sklearn preprocessor.

* clean_data()            validates values, normalises labels, removes only justified duplicates
* chronological_split()   train / validation / test in TIME order (no shuffling)
* build_preprocessor()    imputer (+ scaler for linear models) + one-hot for Operation_Mode.
                          It is FITTED ON TRAINING DATA ONLY and saved with joblib, so the dashboard
                          applies exactly the same transformation as training.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import utils
from .data_audit import add_datetime
from .utils import CLASS_ORDER, NUMERIC_RAW, OPERATION_MODES, PLAUSIBLE_LIMITS, TARGET


def clean_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Return (clean dataframe sorted in file/time order, list of cleaning decisions)."""
    log: List[Dict] = []
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)

    # ---- duplicates: only exact repeats of the WHOLE row (including date/time) are removed
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log.append({"step": "duplicates", "detail": "Removed exact duplicate rows (all 14 columns equal)", "rows_removed": n_dup})

    # ---- text normalisation
    df["Operation_Mode"] = df["Operation_Mode"].astype(str).str.strip().str.title()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
    bad_mode = int((~df["Operation_Mode"].isin(OPERATION_MODES)).sum())
    bad_target = int((~df[TARGET].isin(CLASS_ORDER)).sum())
    df.loc[~df["Operation_Mode"].isin(OPERATION_MODES), "Operation_Mode"] = np.nan   # unexpected category -> missing
    log.append({"step": "categories", "detail": "Operation_Mode / Efficiency_Status stripped + title-cased; unexpected categories -> missing",
                "unexpected_operation_modes": bad_mode, "unexpected_target_values": bad_target})

    # ---- numeric validation
    df["Machine_ID"] = pd.to_numeric(df["Machine_ID"], errors="coerce")
    invalid_total = 0
    for c in NUMERIC_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        lo, hi = PLAUSIBLE_LIMITS[c]
        bad = (df[c] < lo) | (df[c] > hi)
        invalid_total += int(bad.sum())
        df.loc[bad, c] = np.nan                      # invalid -> missing; imputed later INSIDE the pipeline (train medians)
    log.append({"step": "numeric validation", "detail": "Non-numeric or physically impossible values set to missing (imputed with TRAIN medians inside the pipeline)",
                "invalid_values": invalid_total})

    # ---- rows without a usable label cannot be used for supervised learning
    n_before = len(df)
    df = df[df[TARGET].isin(CLASS_ORDER)]
    log.append({"step": "label check", "detail": "Rows without a valid Efficiency_Status removed", "rows_removed": n_before - len(df)})

    df = add_datetime(df.reset_index(drop=True))     # obs_seq = file order (verified chronological in the audit)
    df = df.sort_values("obs_seq").reset_index(drop=True)
    log.append({"step": "ordering", "detail": "Sorted by file order (verified to be one-minute steps); the recorded Date label is kept for display. "
                                              "Two-day block labelled 01-03-2025 is NOT altered.", "rows_final": len(df), "rows_initial": n0})
    df["y"] = df[TARGET].map(utils.CLASS_TO_INT).astype(int)
    return df, log


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> pd.Series:
    """Label every row 'train' / 'val' / 'test' by TIME order (earliest 70% -> train, next 15% -> val, last 15% -> test)."""
    n = len(df)
    n_tr, n_va = int(n * train_frac), int(n * (train_frac + val_frac))
    split = np.where(df["obs_seq"].rank(method="first") <= n_tr, "train",
                     np.where(df["obs_seq"].rank(method="first") <= n_va, "val", "test"))
    return pd.Series(split, index=df.index, name="split")


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scale: bool) -> ColumnTransformer:
    """
    numeric     -> median imputer (+ StandardScaler when scale=True, needed by Logistic Regression)
    categorical -> most-frequent imputer + OneHotEncoder with FIXED categories; unknown category -> all zeros
    Tree models do not need scaling, so scale=False for them.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    parts = []
    if numeric_cols:
        parts.append(("num", Pipeline(num_steps), numeric_cols))
    if categorical_cols:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(categories=[OPERATION_MODES] * len(categorical_cols),
                                                                handle_unknown="ignore", sparse_output=False))]),
                      categorical_cols))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
