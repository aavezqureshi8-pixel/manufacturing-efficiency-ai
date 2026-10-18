"""
utils.py - paths, column names, constants and small helpers shared by every module.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
METRICS_DIR = OUT_DIR / "metrics"
REPORTS_DIR = OUT_DIR / "reports"

RAW_CSV = DATA_DIR / "Thales_Group_Manufacturing.csv"
CLEAN_CSV = PROCESSED_DIR / "clean_data.csv.gz"       # (gzip-compressed CSV, pandas reads/writes it transparently) cleaned data + engineered features + predictions

BEST_MODEL = MODELS_DIR / "best_model.joblib"
PREPROCESSOR = MODELS_DIR / "preprocessor.joblib"
CONTEXT_MODEL = MODELS_DIR / "context_model.joblib"            # sensor/network/quality-only model (no Error_Rate, no Speed)
CONTEXT_PREPROCESSOR = MODELS_DIR / "context_preprocessor.joblib"
METADATA = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42

# ---------------------------------------------------------------- columns
EXPECTED_COLUMNS = [
    "Date", "Timestamp", "Machine_ID", "Operation_Mode", "Temperature_C", "Vibration_Hz",
    "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%", "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr", "Predictive_Maintenance_Score", "Error_Rate_%", "Efficiency_Status",
]
TARGET = "Efficiency_Status"
NUMERIC_RAW = [
    "Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr", "Predictive_Maintenance_Score", "Error_Rate_%",
]
CATEGORICAL_RAW = ["Operation_Mode"]
RAW_INPUTS = NUMERIC_RAW + CATEGORICAL_RAW                       # everything a user can type in the dashboard
SENSOR_COLS = ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW"]
NETWORK_COLS = ["Network_Latency_ms", "Packet_Loss_%"]
PRODUCTION_COLS = ["Production_Speed_units_per_hr", "Quality_Control_Defect_Rate_%", "Error_Rate_%",
                   "Predictive_Maintenance_Score"]
# the two inputs from which the label turns out to be derived (found in the audit, see data_audit.py)
RULE_INPUTS = ["Error_Rate_%", "Production_Speed_units_per_hr"]

CLASS_ORDER = ["Low", "Medium", "High"]                          # ordinal order, also the probability column order
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}
OPERATION_MODES = ["Active", "Idle", "Maintenance"]

# Generous physical-plausibility limits (NOT the observed range). Values outside are treated as invalid.
PLAUSIBLE_LIMITS = {
    "Temperature_C": (-50, 250), "Vibration_Hz": (0, 100), "Power_Consumption_kW": (0, 1000),
    "Network_Latency_ms": (0, 5000), "Packet_Loss_%": (0, 100), "Quality_Control_Defect_Rate_%": (0, 100),
    "Production_Speed_units_per_hr": (0, 10000), "Predictive_Maintenance_Score": (0, 1), "Error_Rate_%": (0, 100),
}

SHORT = {   # short display names
    "Temperature_C": "Temperature (°C)", "Vibration_Hz": "Vibration (Hz)", "Power_Consumption_kW": "Power (kW)",
    "Network_Latency_ms": "Latency (ms)", "Packet_Loss_%": "Packet loss (%)",
    "Quality_Control_Defect_Rate_%": "Defect rate (%)", "Production_Speed_units_per_hr": "Speed (units/hr)",
    "Predictive_Maintenance_Score": "Maintenance score", "Error_Rate_%": "Error rate (%)",
    "Operation_Mode": "Operation mode",
}


def get_logger(name: str = "mfg") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, FIG_DIR, METRICS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, pd.Period)):
        return str(o)
    raise TypeError(f"Cannot serialise {type(o)}")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_convert, ensure_ascii=False)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
