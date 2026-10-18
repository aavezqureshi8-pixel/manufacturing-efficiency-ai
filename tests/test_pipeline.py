"""
Automated checks. Run after training:   python tests/test_pipeline.py     (or:  pytest tests -q)

Covers: dataset loading, cleaning, chronological split, no-leakage of preprocessing and history features,
label-rule check, metrics, saved-model loading, probabilities, local explanations, handling of missing /
unexpected inputs, dashboard filter logic and CSV export.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src import data_audit, evaluation, utils
from src.dashboard_logic import Filters, apply_filters, eta_squared_by_feature, machine_difference_test, machine_ranking, wilson_interval
from src.explainability import monte_carlo_shapley
from src.feature_engineering import (add_history_features, add_instant_features, fit_network_reference, history_feature_names, safe_divide)
from src.inference import EfficiencyPredictor
from src.preprocessing import build_preprocessor, chronological_split, clean_data

_C = {}


def _data():
    if "d" not in _C:
        raw = data_audit.load_raw()
        df, log = clean_data(raw)
        sp = chronological_split(df)
        ref = fit_network_reference(df[sp == "train"])
        _C.update(raw=raw, df=df, sp=sp, ref=ref, d=add_instant_features(df, ref))
    return _C


# ------------------------------------------------------------------ data
def test_dataset_loading_and_schema():
    raw = _data()["raw"]
    assert list(raw.columns) == utils.EXPECTED_COLUMNS
    assert raw.shape == (100000, 14)
    assert set(raw[utils.TARGET].unique()) == set(utils.CLASS_ORDER)


def test_cleaning_keeps_rows_and_orders_by_time():
    df = _data()["df"]
    assert len(df) == 100000 and df["obs_seq"].is_monotonic_increasing
    assert df["datetime"].notna().all() and df[utils.NUMERIC_RAW].notna().all().all()


def test_cleaning_handles_bad_values():
    raw = _data()["raw"].head(50).copy()
    raw.loc[0, "Temperature_C"] = 9999          # impossible
    raw.loc[1, "Operation_Mode"] = "Turbo"      # unexpected category
    raw.loc[2, "Efficiency_Status"] = "Unknown"  # unusable label
    raw = pd.concat([raw, raw.iloc[[5]]])       # exact duplicate
    df, log = clean_data(raw)
    assert len(df) == 49                                  # 51 rows - 1 duplicate - 1 unusable label
    assert df["Temperature_C"].isna().sum() == 1
    assert df["Operation_Mode"].isna().sum() == 1


def test_chronological_split_has_no_overlap():
    df, sp = _data()["df"], _data()["sp"]
    tr, va, te = (df.loc[sp == s, "obs_seq"] for s in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert (len(tr), len(va), len(te)) == (70000, 15000, 15000)


def test_label_rule_matches_data():
    df = _data()["df"]
    agree = (data_audit.apply_label_rule(df) == df[utils.TARGET]).mean()
    assert agree > 0.9999


# ------------------------------------------------------------------ preprocessing / leakage
def test_preprocessor_is_fitted_on_training_rows_only():
    d, sp = _data()["d"], _data()["sp"]
    cols = ["Temperature_C", "Error_Rate_%"]
    pre = build_preprocessor(cols, [], scale=True).fit(d.loc[sp == "train", cols])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_, d.loc[sp == "train", cols].mean().values)
    assert not np.allclose(scaler.mean_, d[cols].mean().values, atol=1e-12)     # differs from full-data mean


def test_network_reference_uses_training_rows_only():
    d, sp, ref = _data()["d"], _data()["sp"], _data()["ref"]
    assert abs(ref["lat_mean"] - d.loc[sp == "train", "Network_Latency_ms"].mean()) < 1e-9


def test_safe_divide_handles_zero_denominator():
    out = safe_divide(pd.Series([1.0, 2.0, 3.0]), pd.Series([0.0, 2.0, np.nan]))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 1.0 and np.isnan(out.iloc[2]) and not np.isinf(out).any()


def test_history_features_never_use_current_or_future_rows():
    d = _data()["d"].iloc[:30000].copy()
    base = d.copy()
    h1 = add_history_features(base)
    targets = list(range(3000, 30000, 3000))
    alt = base.copy()
    idx = alt["obs_seq"].isin(targets)
    for c in utils.NUMERIC_RAW:
        alt.loc[idx, c] = alt.loc[idx, c] * 7 + 3
    h2 = add_history_features(alt)
    cols = history_feature_names()
    a = h1.loc[h1.obs_seq.isin(targets), cols].fillna(-9).values
    b = h2.loc[h2.obs_seq.isin(targets), cols].fillna(-9).values
    assert np.allclose(a, b), "history features of a row changed when only that row was altered (leak)"
    # power of the test: the next record of the same machine DOES see the altered row
    m = base.loc[base.obs_seq == targets[0], "Machine_ID"].iloc[0]
    later = h1[(h1.Machine_ID == m) & (h1.obs_seq > targets[0])].obs_seq.iloc[0]
    assert h1.loc[h1.obs_seq == later, "Temperature_C__prev5_mean"].iloc[0] != h2.loc[h2.obs_seq == later, "Temperature_C__prev5_mean"].iloc[0]


# ------------------------------------------------------------------ metrics
def test_metrics_on_known_cases():
    y = np.array([0, 0, 1, 2, 2, 2])
    perfect = np.eye(3)[y] * 0.98 + 0.01
    m = evaluation.compute_metrics(y, perfect)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["balanced_accuracy"] == 1.0
    always0 = np.tile([0.98, 0.01, 0.01], (len(y), 1))
    m0 = evaluation.compute_metrics(y, always0)
    assert abs(m0["balanced_accuracy"] - 1 / 3) < 1e-9 and m0["recall_High"] == 0.0
    assert 0 <= evaluation.expected_calibration_error(perfect, y) <= 1


# ------------------------------------------------------------------ saved artifacts
def test_saved_artifacts_exist_and_predict_valid_probabilities():
    for p in (utils.BEST_MODEL, utils.PREPROCESSOR, utils.CONTEXT_MODEL, utils.CONTEXT_PREPROCESSOR, utils.METADATA):
        assert p.exists(), f"missing {p.name}: run python -m src.train_models"
    pred = EfficiencyPredictor.load("best")
    d = _data()["d"].iloc[:500]
    p = pred.predict_proba(d)
    assert p.shape == (500, 3) and np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()


def test_saved_predictions_match_metadata_test_metrics():
    meta = utils.load_json(utils.METADATA)
    out = pd.read_csv(utils.CLEAN_CSV)
    te = out[out["split"] == "test"]
    y = te[utils.TARGET].map(utils.CLASS_TO_INT).values
    proba = te[["p_Low", "p_Medium", "p_High"]].values
    m = evaluation.compute_metrics(y, proba)
    assert abs(m["macro_f1"] - meta["final_model"]["test_metrics"]["macro_f1"]) < 1e-3


def test_inference_handles_missing_and_unexpected_inputs():
    pred = EfficiencyPredictor.load("best")
    row = {c: pred.meta["feature_ranges"][c]["median"] for c in utils.NUMERIC_RAW}
    row["Operation_Mode"] = "Active"
    bad = dict(row)
    bad["Temperature_C"] = np.nan
    bad["Operation_Mode"] = "Turbo"
    bad["Vibration_Hz"] = -5
    bad["Network_Latency_ms"] = 99999
    p = pred.predict_proba(pd.DataFrame([bad]))
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1.0)
    w = " ".join(pred.input_warnings(bad))
    assert "Temperature" in w and "Turbo" in w and "implausible" in w
    outside = dict(row)
    outside["Production_Speed_units_per_hr"] = 2000
    assert any("OUTSIDE" in x for x in pred.input_warnings(outside))
    try:
        pred.predict_proba(pd.DataFrame([{"Temperature_C": 1}]))
        raise AssertionError("expected KeyError for missing columns")
    except KeyError:
        pass


def test_known_rule_cases_are_classified_correctly():
    pred = EfficiencyPredictor.load("best")
    base = {c: pred.meta["feature_ranges"][c]["median"] for c in utils.NUMERIC_RAW}
    base["Operation_Mode"] = "Active"
    high = {**base, "Error_Rate_%": 1.0, "Production_Speed_units_per_hr": 450.0}
    med = {**base, "Error_Rate_%": 3.0, "Production_Speed_units_per_hr": 300.0}
    low = {**base, "Error_Rate_%": 9.0, "Production_Speed_units_per_hr": 450.0}
    res = pred.predict_frame(pd.DataFrame([high, med, low]))
    assert list(res["pred_class"]) == ["High", "Medium", "Low"]


def test_local_shapley_attributions_are_additive():
    pred = EfficiencyPredictor.load("best")
    d = _data()["d"]
    x = d.iloc[10][utils.RAW_INPUTS].to_dict()
    bg = d.iloc[[20000]][utils.RAW_INPUTS]               # single background row -> telescoping sum is exact
    attr = monte_carlo_shapley(pred, x, bg, class_idx=0, n_perm=20, seed=1)
    px = pred.predict_proba(pd.DataFrame([x]))[0, 0]
    pb = pred.predict_proba(bg)[0, 0]
    assert abs(attr["attribution"].sum() - (px - pb)) < 1e-9


# ------------------------------------------------------------------ dashboard logic
def test_filters_return_expected_rows():
    out = pd.read_csv(utils.CLEAN_CSV, parse_dates=["datetime"])
    f = Filters(network_quality=["Poor"], machines=[5, 6], operation_mode="Active", classes=["Low", "High"])
    r = apply_filters(out, f)
    manual = out[(out.network_quality == "Poor") & out.Machine_ID.isin([5, 6]) & (out.Operation_Mode == "Active") & out[utils.TARGET].isin(["Low", "High"])]
    assert len(r) == len(manual) and len(r) > 0
    d0 = apply_filters(out, Filters(date_range=(pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-06"))))
    assert d0["datetime"].min() >= pd.Timestamp("2025-01-05") and d0["datetime"].max() < pd.Timestamp("2025-01-07")
    assert len(apply_filters(out, Filters())) == len(out)
    assert apply_filters(out, Filters(machines=[9999])).empty


def test_machine_ranking_and_statistics():
    out = pd.read_csv(utils.CLEAN_CSV, parse_dates=["datetime"])
    rk = machine_ranking(out)
    assert rk["records"].sum() == len(out) and rk["low_records"].sum() == (out[utils.TARGET] == "Low").sum()
    assert ((rk["ci_low"] >= 0) & (rk["ci_high"] <= 1) & (rk["ci_low"] <= rk["low_share"]) & (rk["low_share"] <= rk["ci_high"])).all()
    lo, hi = wilson_interval(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert lo[0] < 0.01 and hi[2] > 0.98 and 0.2 < lo[1] < 0.5 < hi[1] < 0.8
    assert machine_difference_test(out)["p_value"] > 0 and machine_difference_test(out[out.Machine_ID == 1]) is None
    assert eta_squared_by_feature(out).set_index("feature").loc["Error_Rate_%", "eta_squared"] > 0.3


def test_csv_export_round_trip():
    out = pd.read_csv(utils.CLEAN_CSV, nrows=100)
    again = pd.read_csv(pd.io.common.BytesIO(out.to_csv(index=False).encode("utf-8")))
    assert again.shape == out.shape and list(again.columns) == list(out.columns)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\nAll {len(tests)} tests passed.")
