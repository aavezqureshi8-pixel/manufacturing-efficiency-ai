"""
dashboard_logic.py - the data logic behind the Streamlit dashboard, written WITHOUT Streamlit
so that it can be unit-tested: filtering, machine ranking with confidence intervals, effect sizes, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .utils import CLASS_ORDER, NUMERIC_RAW, TARGET


@dataclass
class Filters:
    """All sidebar selections. Empty list / None means 'no restriction'."""
    network_quality: List[str] = field(default_factory=list)
    machines: List[int] = field(default_factory=list)
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    operation_mode: str = "All"
    classes: List[str] = field(default_factory=list)
    error_range: Optional[Tuple[float, float]] = None
    speed_range: Optional[Tuple[float, float]] = None


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Return the rows matching every active filter."""
    m = pd.Series(True, index=df.index)
    if f.network_quality:
        m &= df["network_quality"].isin(f.network_quality)
    if f.machines:
        m &= df["Machine_ID"].isin(f.machines)
    if f.date_range is not None:
        start, end = f.date_range
        m &= (df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    if f.operation_mode and f.operation_mode != "All":
        m &= df["Operation_Mode"] == f.operation_mode
    if f.classes:
        m &= df[TARGET].isin(f.classes)
    if f.error_range is not None:
        m &= df["Error_Rate_%"].between(*f.error_range)
    if f.speed_range is not None:
        m &= df["Production_Speed_units_per_hr"].between(*f.speed_range)
    return df[m]


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    n_safe = np.where(n > 0, n, np.nan)
    p = k / n_safe
    den = 1 + z ** 2 / n_safe
    centre = (p + z ** 2 / (2 * n_safe)) / den
    half = z * np.sqrt(p * (1 - p) / n_safe + z ** 2 / (4 * n_safe ** 2)) / den
    return centre - half, centre + half


def machine_ranking(df: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """
    Observed share of Low-efficiency records per machine with a 95% Wilson interval and the number of observations.
    `distinguishable` is True only when the whole interval lies above/below the fleet-wide share, i.e. the machine
    differs from the fleet by more than chance. It is NOT a fault diagnosis.
    """
    if df.empty:
        return pd.DataFrame(columns=["Machine_ID", "records", "low_records", "low_share", "ci_low", "ci_high", "vs_fleet", "enough_data"])
    g = df.groupby("Machine_ID").agg(records=(TARGET, "size"), low_records=(TARGET, lambda x: int((x == "Low").sum()))).reset_index()
    g["low_share"] = g["low_records"] / g["records"]
    g["ci_low"], g["ci_high"] = wilson_interval(g["low_records"].values, g["records"].values)
    fleet = float((df[TARGET] == "Low").mean())
    g["vs_fleet"] = np.where(g["ci_low"] > fleet, "above fleet", np.where(g["ci_high"] < fleet, "below fleet", "not distinguishable"))
    g["enough_data"] = g["records"] >= min_obs
    g["fleet_low_share"] = fleet
    return g.sort_values("low_share", ascending=False).reset_index(drop=True)


def machine_difference_test(df: pd.DataFrame) -> Optional[Dict]:
    """Chi-square test: do machines differ in their class mix more than chance would produce? None if not computable."""
    if df["Machine_ID"].nunique() < 2 or df[TARGET].nunique() < 2:
        return None
    ct = pd.crosstab(df["Machine_ID"], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def eta_squared_by_feature(df: pd.DataFrame, cols: List[str] = NUMERIC_RAW) -> pd.DataFrame:
    """Share of each measurement's variance explained by the efficiency class (0 = unrelated, 1 = fully determined)."""
    rows = []
    for c in cols:
        groups = [g[c].values for _, g in df.groupby(TARGET) if len(g) > 1]
        if len(groups) < 2:
            rows.append({"feature": c, "eta_squared": np.nan})
            continue
        grand = df[c].mean()
        ss_tot = ((df[c] - grand) ** 2).sum()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        rows.append({"feature": c, "eta_squared": float(ss_b / ss_tot) if ss_tot > 0 else np.nan})
    return pd.DataFrame(rows)


FEATURE_GROUP = {"Temperature_C": "Sensor", "Vibration_Hz": "Sensor", "Power_Consumption_kW": "Sensor",
                 "Network_Latency_ms": "Network", "Packet_Loss_%": "Network",
                 "Quality_Control_Defect_Rate_%": "Quality", "Production_Speed_units_per_hr": "Production",
                 "Error_Rate_%": "Production", "Predictive_Maintenance_Score": "Maintenance"}


def class_share_by_period(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Share (0-1) of each recorded class per day ('D') or week ('W'), with the record count."""
    if df.empty:
        return pd.DataFrame(columns=["period", "n"] + CLASS_ORDER)
    per = df["datetime"].dt.to_period(freq).dt.start_time
    ct = pd.crosstab(per, df[TARGET]).reindex(columns=CLASS_ORDER, fill_value=0)
    n = ct.sum(axis=1)
    out = (ct.div(n, axis=0)).reset_index()
    out.columns = ["period"] + CLASS_ORDER
    out["n"] = n.values
    return out


def daily_means(df: pd.DataFrame, cols: List[str], freq: str = "D") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period"] + cols)
    per = df["datetime"].dt.to_period(freq).dt.start_time
    out = df.groupby(per)[cols].mean().reset_index()
    out.columns = ["period"] + cols
    return out


def top_low_efficiency_factors(df: pd.DataFrame) -> List[str]:
    """Plain-English facts comparing Low and High records (association only)."""
    from .utils import SHORT
    lines = []
    low, high = df[df[TARGET] == "Low"], df[df[TARGET] == "High"]
    if low.empty or high.empty:
        return ["Not enough Low and High records in the current selection to compare."]
    sd = df[NUMERIC_RAW].std()
    effects = ((low[NUMERIC_RAW].mean() - high[NUMERIC_RAW].mean()) / sd).abs().sort_values(ascending=False)
    for c in effects.index[:4]:
        lines.append(f"{SHORT[c]}: mean {low[c].mean():.2f} in Low vs {high[c].mean():.2f} in High records "
                     f"(difference = {effects[c]:.2f} standard deviations).")
    return lines
