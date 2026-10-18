"""
inference.py - ONE place where raw inputs become predictions, used by training, tests and the dashboard.

raw inputs  ->  derived features (feature_engineering.add_instant_features)
            ->  saved preprocessor (fitted on training data only)
            ->  saved model  ->  class probabilities
"""
from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from . import utils
from .feature_engineering import add_instant_features
from .utils import CLASS_ORDER, NUMERIC_RAW, OPERATION_MODES, RAW_INPUTS, SHORT


class EfficiencyPredictor:
    """Wraps (preprocessor, model, metadata) so that every caller uses identical feature engineering."""

    def __init__(self, preprocessor, model, metadata: Dict, feature_set: str):
        self.pre, self.model, self.meta, self.feature_set = preprocessor, model, metadata, feature_set
        self.net_ref = metadata["network_reference"]
        fs = metadata["feature_sets"][feature_set]
        self.columns = fs["numeric"] + fs["categorical"]

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, which: str = "best") -> "EfficiencyPredictor":
        meta = utils.load_json(utils.METADATA)
        if which == "best":
            return cls(joblib.load(utils.PREPROCESSOR), joblib.load(utils.BEST_MODEL), meta, meta["final_model"]["feature_set"])
        return cls(joblib.load(utils.CONTEXT_PREPROCESSOR), joblib.load(utils.CONTEXT_MODEL), meta, meta["context_model"]["feature_set"])

    # ------------------------------------------------------------------ prediction
    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in RAW_INPUTS if c not in raw.columns]
        if missing:
            raise KeyError(f"Missing input columns: {missing}")
        df = raw[RAW_INPUTS].copy()
        for c in NUMERIC_RAW:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Operation_Mode"] = df["Operation_Mode"].astype(str).str.strip().str.title()
        df.loc[~df["Operation_Mode"].isin(OPERATION_MODES), "Operation_Mode"] = np.nan
        return add_instant_features(df, self.net_ref)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """n x 3 matrix, columns = [Low, Medium, High]."""
        feats = self.prepare(raw)
        X = self.pre.transform(feats[self.columns])
        proba = self.model.predict_proba(X)
        out = np.zeros((len(raw), len(CLASS_ORDER)))
        for j, cls_ in enumerate(self.model.classes_):          # classes_ are the integers 0,1,2
            out[:, int(cls_)] = proba[:, j]
        return out

    def predict_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_proba(raw)
        res = pd.DataFrame(p, columns=[f"p_{c}" for c in CLASS_ORDER], index=raw.index)
        res["pred_class"] = [CLASS_ORDER[i] for i in p.argmax(axis=1)]
        res["max_prob"] = p.max(axis=1)
        return res

    # ------------------------------------------------------------------ input checking
    def input_warnings(self, row: Dict) -> List[str]:
        """Human-readable warnings for one input record (invalid, missing, unusual or unknown values)."""
        warns = []
        ranges = self.meta["feature_ranges"]
        for c in NUMERIC_RAW:
            v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            if pd.isna(v):
                warns.append(f"{SHORT[c]} is missing or not a number; the training median will be used.")
                continue
            lo_lim, hi_lim = utils.PLAUSIBLE_LIMITS[c]
            if v < lo_lim or v > hi_lim:
                warns.append(f"{SHORT[c]} = {v:g} is physically implausible (limit {lo_lim} to {hi_lim}); treated as missing.")
            elif v < ranges[c]["min"] or v > ranges[c]["max"]:
                warns.append(f"{SHORT[c]} = {v:g} lies OUTSIDE the range seen in training ({ranges[c]['min']:g} to {ranges[c]['max']:g}); "
                             "the prediction is an extrapolation and should be treated with caution.")
        if str(row.get("Operation_Mode")).strip().title() not in OPERATION_MODES:
            warns.append(f"Operation mode '{row.get('Operation_Mode')}' is unknown; it is encoded as 'no mode'.")
        return warns

    def confidence_note(self, proba_row: np.ndarray) -> str:
        top = float(np.max(proba_row))
        srt = np.sort(proba_row)[::-1]
        if top < 0.6 or (srt[0] - srt[1]) < 0.2:
            return "LOW separation between the top two classes - treat this prediction as uncertain."
        return ("The model is confident, but remember that the highest class probability is a model score, "
                "not a guaranteed probability of being correct.")
