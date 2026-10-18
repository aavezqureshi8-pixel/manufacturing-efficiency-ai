"""
feature_engineering.py - derived features and the named feature sets used in the experiments.

Two kinds of features:
 1. INSTANT features - computed from ONE observation (ratios, network stress index). They can be
    computed for a single row typed into the dashboard, so training and inference are identical.
 2. HISTORY features - rolling statistics of the SAME machine's PREVIOUS records (shift(1) first, so the
    current row and future rows are never used). They are only evaluated in the experiments.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import NETWORK_COLS, NUMERIC_RAW, RULE_INPUTS

HISTORY_WINDOW = 5      # previous records of the same machine (a record-based window, because records per machine are ~50 minutes apart)
HISTORY_BASE = ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%",
                "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr", "Error_Rate_%"]


# ------------------------------------------------------------------ network reference (fit on TRAIN only)
def fit_network_reference(train: pd.DataFrame) -> Dict:
    """Mean/std of latency and packet loss and the tertile cut-points of the stress index, from TRAINING rows only."""
    ref = {"lat_mean": float(train["Network_Latency_ms"].mean()), "lat_std": float(train["Network_Latency_ms"].std()),
           "loss_mean": float(train["Packet_Loss_%"].mean()), "loss_std": float(train["Packet_Loss_%"].std())}
    idx = _stress(train, ref)
    ref["good_max"], ref["fair_max"] = float(idx.quantile(1 / 3)), float(idx.quantile(2 / 3))
    return ref


def _stress(df: pd.DataFrame, ref: Dict) -> pd.Series:
    return ((df["Network_Latency_ms"] - ref["lat_mean"]) / ref["lat_std"]
            + (df["Packet_Loss_%"] - ref["loss_mean"]) / ref["loss_std"])


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b, but NaN (not inf / error) when the denominator is zero or missing."""
    b = b.where(b > 0)
    return a / b


def add_instant_features(df: pd.DataFrame, net_ref: Dict) -> pd.DataFrame:
    """Single-observation derived features (the SAME function is used for training and the dashboard)."""
    out = df.copy()
    out["energy_per_unit"] = safe_divide(out["Power_Consumption_kW"], out["Production_Speed_units_per_hr"])   # kWh per unit produced
    out["error_to_speed_ratio"] = safe_divide(out["Error_Rate_%"], out["Production_Speed_units_per_hr"])
    out["network_stress_index"] = _stress(out, net_ref)                       # higher = worse network (latency + packet loss, standardised)
    out["network_quality"] = np.where(out["network_stress_index"] <= net_ref["good_max"], "Good",
                                      np.where(out["network_stress_index"] <= net_ref["fair_max"], "Fair", "Poor"))
    return out


def add_history_features(df: pd.DataFrame, window: int = HISTORY_WINDOW) -> pd.DataFrame:
    """
    Past-only rolling features per machine. df must contain obs_seq and Machine_ID.
    For every row, only PREVIOUS records of the same machine are used (shift(1) before rolling). The CURRENT row's
    measurements are never used (tested in tests/test_pipeline.py: changing the current row leaves its history features unchanged).
    """
    out = df.sort_values("obs_seq").copy()
    g = out.groupby("Machine_ID", sort=False)
    new = {}
    for c in HISTORY_BASE:
        short = c.replace("_%", "pct").replace("_", "")[:14]
        new[f"{c}__prev{window}_mean"] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
        new[f"{c}__prev{window}_std"] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
    # change in speed between the TWO PREVIOUS records of the machine (uses no information from the current row)
    new["speed_change_prev"] = g["Production_Speed_units_per_hr"].shift(1) - g["Production_Speed_units_per_hr"].shift(2)
    new["minutes_since_prev_record"] = out["obs_seq"] - g["obs_seq"].shift(1)       # file order = 1 record per minute
    new["hour"] = out["datetime"].dt.hour
    new["day_of_week"] = out["datetime"].dt.dayofweek
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1).sort_values("obs_seq")


def history_feature_names(window: int = HISTORY_WINDOW) -> List[str]:
    names = []
    for c in HISTORY_BASE:
        names += [f"{c}__prev{window}_mean", f"{c}__prev{window}_std"]
    return names + ["speed_change_prev", "minutes_since_prev_record", "hour", "day_of_week"]


# ------------------------------------------------------------------ named feature sets
CONTEXT_NUMERIC = [c for c in NUMERIC_RAW if c not in RULE_INPUTS]          # 7 measurements without Error_Rate_% and Speed
FEATURE_SETS: Dict[str, Dict] = {
    "F1_raw_all": {
        "numeric": NUMERIC_RAW, "categorical": ["Operation_Mode"],
        "description": "All 9 raw measurements + Operation_Mode (includes the two label-defining columns)."},
    "F2_raw_plus_derived": {
        "numeric": NUMERIC_RAW + ["energy_per_unit", "error_to_speed_ratio", "network_stress_index"], "categorical": ["Operation_Mode"],
        "description": "F1 + single-observation derived features (energy per unit, error-to-speed ratio, network stress index). DEPLOYED set."},
    "F3_context_only": {
        "numeric": CONTEXT_NUMERIC + ["network_stress_index"], "categorical": ["Operation_Mode"],
        "description": "Sensors + network + defect rate + maintenance score + mode; WITHOUT Error_Rate_% and Production_Speed (tests the sensor/6G premise)."},
    "F4_context_plus_history": {
        "numeric": CONTEXT_NUMERIC + ["network_stress_index"] + history_feature_names(), "categorical": ["Operation_Mode"],
        "description": "F3 + past-only rolling statistics of the same machine (incl. previous Error_Rate_% / Speed) + time-of-day features."},
    "F5_rule_inputs_only": {
        "numeric": RULE_INPUTS, "categorical": [],
        "description": "ONLY Error_Rate_% and Production_Speed (shows these two columns are sufficient)."},
}
