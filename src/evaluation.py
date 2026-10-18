"""
evaluation.py - every metric requested in the brief, calculated from real predictions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support, precision_score, recall_score)

from .utils import CLASS_ORDER


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE: bins predictions by their highest probability and compares it with the observed accuracy."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_multiclass(proba: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability_table(proba: np.ndarray, y: np.ndarray, class_idx: int, n_bins: int = 10) -> pd.DataFrame:
    """One-vs-rest reliability data for one class: mean predicted probability vs observed frequency per bin."""
    p = proba[:, class_idx]
    t = (y == class_idx).astype(float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append({"class": CLASS_ORDER[class_idx], "bin": b, "n": int(m.sum()),
                         "mean_predicted": float(p[m].mean()), "observed_frequency": float(t[m].mean())})
    return pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict:
    """All requested metrics for one model on one dataset."""
    y_true = np.asarray(y_true)
    y_pred = proba.argmax(axis=1)
    labels = list(range(len(CLASS_ORDER)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    m = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-15, 1), labels=labels)),
        "brier": brier_multiclass(proba, y_true),
        "ece_top_label": expected_calibration_error(proba, y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=CLASS_ORDER, zero_division=0, digits=4),
    }
    for i, c in enumerate(CLASS_ORDER):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"], m[f"support_{c}"] = float(p[i]), float(r[i]), float(f[i]), int(s[i])
    return m


FLAT_KEYS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
             "precision_Low", "recall_Low", "f1_Low", "precision_Medium", "recall_Medium", "f1_Medium",
             "precision_High", "recall_High", "f1_High", "log_loss", "brier", "ece_top_label"]


def flat(m: Dict) -> Dict:
    return {k: m[k] for k in FLAT_KEYS}


def majority_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> Dict:
    """What a model that always predicts the most common training class would score."""
    k = int(np.bincount(y_train).argmax())
    proba = np.zeros((len(y_eval), len(CLASS_ORDER)))
    proba[:, k] = 1.0
    return compute_metrics(y_eval, np.clip(proba, 1e-9, 1))
