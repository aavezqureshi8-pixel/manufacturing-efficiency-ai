"""
explainability.py - model-agnostic explanations that need only scikit-learn / numpy
(SHAP is not required).

* permutation_importance_raw()  GLOBAL: how much a score drops when one raw input is shuffled (validation data)
* monte_carlo_shapley()         LOCAL: Shapley-value attribution of one prediction, estimated by sampling feature orderings
                                (interventional version with a background sample; same idea as SHAP, estimated by simulation)
* class_profile()               how each measurement differs between High / Medium / Low

Important: these are statistical ASSOCIATIONS learned by the model. They do not prove that a feature CAUSES
efficiency to change.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from .utils import CLASS_ORDER, RAW_INPUTS, TARGET


def permutation_importance_raw(predictor, X_raw: pd.DataFrame, y: np.ndarray, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """Shuffle ONE raw input at a time (derived features are recomputed from it) and measure the drop in macro-F1 and balanced accuracy."""
    rng = np.random.default_rng(seed)
    base = predictor.predict_proba(X_raw).argmax(axis=1)
    base_f1, base_ba = f1_score(y, base, average="macro"), balanced_accuracy_score(y, base)
    rows = []
    for col in RAW_INPUTS:
        d_f1, d_ba = [], []
        for _ in range(n_repeats):
            Xp = X_raw.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            pred = predictor.predict_proba(Xp).argmax(axis=1)
            d_f1.append(base_f1 - f1_score(y, pred, average="macro"))
            d_ba.append(base_ba - balanced_accuracy_score(y, pred))
        rows.append({"feature": col, "f1_drop_mean": float(np.mean(d_f1)), "f1_drop_std": float(np.std(d_f1)),
                     "balanced_acc_drop_mean": float(np.mean(d_ba))})
    return pd.DataFrame(rows).sort_values("f1_drop_mean", ascending=False).reset_index(drop=True)


def monte_carlo_shapley(predictor, x: Dict, background: pd.DataFrame, class_idx: int, n_perm: int = 30,
                        seed: int = 42) -> pd.DataFrame:
    """
    Shapley attribution of P(class) for ONE record over the raw inputs.

    For each random ordering of the inputs, inputs are switched one by one from a random background record's values
    to the explained record's values; the change in P(class) at each switch is credited to that input.
    Averaged over orderings this estimates the Shapley value. Properties (checked in tests):
        sum(attributions) = P(class | record) - mean P(class | background)      (up to sampling noise from the unused permutations)
    """
    rng = np.random.default_rng(seed)
    feats = list(RAW_INPUTS)
    d = len(feats)
    bg = background[feats].reset_index(drop=True)
    x_row = pd.Series({f: x[f] for f in feats})

    rows = []
    meta = []
    for _ in range(n_perm):
        order = rng.permutation(d)
        b = bg.iloc[int(rng.integers(len(bg)))]
        cur = b.copy()
        rows.append(cur.copy())
        meta.append((None, 0))
        for k, j in enumerate(order, start=1):
            cur = cur.copy()
            cur[feats[j]] = x_row[feats[j]]
            rows.append(cur)
            meta.append((feats[j], k))
    batch = pd.DataFrame(rows).reset_index(drop=True)
    for c in feats:
        if c != "Operation_Mode":
            batch[c] = pd.to_numeric(batch[c])
    p = predictor.predict_proba(batch)[:, class_idx]
    contrib = {f: 0.0 for f in feats}
    i = 0
    for _ in range(n_perm):
        prev = p[i]
        i += 1
        for _k in range(d):
            f = meta[i][0]
            contrib[f] += p[i] - prev
            prev = p[i]
            i += 1
    out = pd.DataFrame({"feature": feats, "attribution": [contrib[f] / n_perm for f in feats],
                        "value": [x_row[f] for f in feats]})
    base_value = float(predictor.predict_proba(bg)[:, class_idx].mean())
    out.attrs["base_value"] = base_value
    out.attrs["prediction"] = float(predictor.predict_proba(pd.DataFrame([x_row]).astype({c: float for c in feats if c != "Operation_Mode"}))[0, class_idx])
    return out.sort_values("attribution", key=lambda s: -s.abs()).reset_index(drop=True)


def plain_english(attr: pd.DataFrame, class_name: str, top: int = 3) -> str:
    """One short explanation sentence from the attribution table."""
    from .utils import SHORT
    pred, base = attr.attrs["prediction"], attr.attrs["base_value"]
    parts = []
    for _, r in attr.head(top).iterrows():
        if abs(r["attribution"]) < 0.005:
            continue
        val = r["value"] if isinstance(r["value"], str) else f"{r['value']:.3g}"
        parts.append(f"{SHORT[r['feature']]} = {val} {'raised' if r['attribution'] > 0 else 'lowered'} P({class_name}) by about {abs(r['attribution']):.2f}")
    if not parts:
        return (f"P({class_name}) is {pred:.2f}, close to the average background value {base:.2f}; no single input moved it noticeably. "
                "(Attributions describe the model's behaviour, not physical causes.)")
    return (f"P({class_name}) is {pred:.2f} versus an average of {base:.2f} for typical records. " + "; ".join(parts) +
            ". (These are associations learned by the model, not proven physical causes.)")


def class_profile(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Mean of every measurement per efficiency class and the standardised difference High-Low."""
    g = df.groupby(TARGET)[cols].mean().reindex(CLASS_ORDER).T
    g["std_diff_High_minus_Low"] = (g["High"] - g["Low"]) / df[cols].std()
    return g
