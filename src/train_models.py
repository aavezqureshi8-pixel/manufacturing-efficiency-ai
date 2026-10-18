"""
train_models.py - the whole modelling pipeline in one command:

    python -m src.train_models

1  audit + clean + chronological split (train 70% / validation 15% / test 15%, in TIME order)
2  feature engineering (network reference fitted on TRAIN only; history features use the past only)
3  experiment grid: 3 models x 4 feature sets x 2 class-weight settings (+ a 2-feature reference tree)
4  model selection on VALIDATION results only; test results are reported afterwards
5  stability: expanding-window time folds and machines-never-seen-in-training folds
6  calibration, permutation importance, Monte-Carlo Shapley attributions
7  save artifacts (joblib + JSON) and the predictions table used by the dashboard
"""
from __future__ import annotations

import platform
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from . import data_audit, evaluation, explainability, figures, utils
from .feature_engineering import (FEATURE_SETS, add_history_features, add_instant_features, fit_network_reference)
from .inference import EfficiencyPredictor
from .preprocessing import build_preprocessor, chronological_split, clean_data
from .utils import CLASS_ORDER, RAW_INPUTS, RULE_INPUTS, TARGET, get_logger, save_json

log = get_logger()

MODEL_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
SELECTION_TIE_MARGIN = 0.001          # validation macro-F1 differences below this are treated as ties


# ------------------------------------------------------------------------------------------
def make_model(name: str, class_weight: str | None):
    """The three required models. class_weight='balanced' gives rare classes a larger weight in the loss."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000, class_weight="balanced" if class_weight else None, random_state=utils.RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=utils.RANDOM_STATE,
                                      class_weight="balanced_subsample" if class_weight else None)
    if name == "Gradient Boosting":     # scikit-learn HistGradientBoosting (XGBoost is optional and was not required)
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=utils.RANDOM_STATE,
                                              class_weight="balanced" if class_weight else None)
    raise ValueError(name)


def needs_scaling(name: str) -> bool:
    return name == "Logistic Regression"


def build_dataset() -> Dict:
    """Clean data, split in time order, fit the network reference on TRAIN, add derived + history features."""
    raw = data_audit.load_raw()
    df, clean_log = clean_data(raw)
    split = chronological_split(df)
    net_ref = fit_network_reference(df[split == "train"])
    d = add_instant_features(df, net_ref)
    d = add_history_features(d)                         # past-only, safe to compute on the whole ordered series
    d["split"] = split.values
    d = d.sort_values("obs_seq").reset_index(drop=True)
    return {"df": d, "net_ref": net_ref, "clean_log": clean_log}


def fit_score(model_name: str, cw: str | None, fs_name: str, tr: pd.DataFrame, evals: Dict[str, pd.DataFrame]):
    """Fit preprocessor + model on `tr` only, score on each frame in `evals`. Returns (record list, pre, model, probas)."""
    fs = FEATURE_SETS[fs_name]
    cols = fs["numeric"] + fs["categorical"]
    pre = build_preprocessor(fs["numeric"], fs["categorical"], scale=needs_scaling(model_name))
    Xtr = pre.fit_transform(tr[cols])
    if fs_name == "F5_rule_inputs_only":
        model = DecisionTreeClassifier(max_depth=4, random_state=utils.RANDOM_STATE, class_weight="balanced" if cw else None)
    else:
        model = make_model(model_name, cw)
    t0 = time.perf_counter()
    model.fit(Xtr, tr["y"])
    train_s = time.perf_counter() - t0
    recs, probas = [], {}
    for split_name, frame in evals.items():
        Xe = pre.transform(frame[cols])
        t0 = time.perf_counter()
        p_raw = model.predict_proba(Xe)
        pred_ms = 1000 * (time.perf_counter() - t0) / max(len(frame), 1) * 1000
        p = np.zeros((len(frame), 3))
        for j, c in enumerate(model.classes_):
            p[:, int(c)] = p_raw[:, j]
        probas[split_name] = p
        m = evaluation.compute_metrics(frame["y"].values, p)
        recs.append({"feature_set": fs_name, "model": model_name, "class_weight": cw or "none", "split": split_name,
                     "train_seconds": train_s, "predict_ms_per_1k": pred_ms, **evaluation.flat(m)})
    return recs, pre, model, probas


# ------------------------------------------------------------------------------------------
def run_experiments(d: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    rows, store = [], {}
    log.info("Experiment grid (3 models x 4 feature sets x 2 class-weight settings) ...")
    for fs_name in ["F1_raw_all", "F2_raw_plus_derived", "F3_context_only", "F4_context_plus_history"]:
        for model_name in MODEL_NAMES:
            for cw in ("balanced", None):
                recs, pre, model, probas = fit_score(model_name, cw, fs_name, tr, {"val": va, "test": te})
                rows += recs
                store[(fs_name, model_name, cw or "none")] = (pre, model, probas)
                v = [r for r in recs if r["split"] == "val"][0]
                log.info("  %-24s %-20s cw=%-8s val macroF1=%.4f balAcc=%.4f", fs_name, model_name, cw or "none", v["macro_f1"], v["balanced_accuracy"])
    # reference: depth-4 tree on the two label-defining columns only
    recs, pre, model, probas = fit_score("Decision Tree (depth 4)", None, "F5_rule_inputs_only", tr, {"val": va, "test": te})
    rows += recs
    store[("F5_rule_inputs_only", "Decision Tree (depth 4)", "none")] = (pre, model, probas)
    # reference: majority-class baseline
    for split_name, frame in (("val", va), ("test", te)):
        m = evaluation.majority_baseline(tr["y"].values, frame["y"].values)
        rows.append({"feature_set": "baseline", "model": "Majority class (always Low)", "class_weight": "n/a", "split": split_name,
                     "train_seconds": 0.0, "predict_ms_per_1k": 0.0, **evaluation.flat(m)})
    return pd.DataFrame(rows), store


def select_final(results: pd.DataFrame, feature_set: str, candidates_cw=("balanced", "none")) -> Dict:
    """
    Final-model rule (validation data only):
      1. highest validation macro-F1 among the three models (both class-weight settings) on `feature_set`;
      2. candidates within SELECTION_TIE_MARGIN of the best are treated as tied and ranked by
         High-class F1 (minority class), then by training time (simpler/faster).
    """
    v = results[(results["feature_set"] == feature_set) & (results["split"] == "val") & (results["class_weight"].isin(candidates_cw))
                & (results["model"].isin(MODEL_NAMES))].copy()
    best = v["macro_f1"].max()
    tied = v[v["macro_f1"] >= best - SELECTION_TIE_MARGIN].sort_values(["f1_High", "train_seconds"], ascending=[False, True])
    w = tied.iloc[0]
    return {"model": w["model"], "class_weight": w["class_weight"], "feature_set": feature_set,
            "val_macro_f1": float(w["macro_f1"]), "tied_candidates": tied[["model", "class_weight", "macro_f1", "f1_High"]].to_dict("records"),
            "rule": (f"highest validation macro-F1 on {feature_set}; models within {SELECTION_TIE_MARGIN} of the best are tied and "
                     "ranked by High-class F1, then training time")}


# ------------------------------------------------------------------------------------------
def time_fold_stability(d: pd.DataFrame, fs_name: str, cw: str = "balanced", n_splits: int = 5) -> pd.DataFrame:
    """Expanding-window validation over the first 85% of the data (train+val). Each fold trains on the past, tests on the next block."""
    dev = d[d["split"].isin(["train", "val"])].reset_index(drop=True)
    rows = []
    for k, (a, b) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(dev), start=1):
        tr, ev = dev.iloc[a], dev.iloc[b]
        for model_name in MODEL_NAMES:
            recs, *_ = fit_score(model_name, cw, fs_name, tr, {"fold": ev})
            r = recs[0]
            rows.append({"feature_set": fs_name, "fold": k, "train_rows": len(tr), "eval_rows": len(ev), "model": model_name,
                         "macro_f1": r["macro_f1"], "balanced_accuracy": r["balanced_accuracy"], "f1_High": r["f1_High"]})
    return pd.DataFrame(rows)


def machine_holdout(d: pd.DataFrame, fs_name: str, cw: str = "balanced", n_splits: int = 5) -> pd.DataFrame:
    """Train on the training period using 40 machines; evaluate on the TEST period using 10 machines never seen in training."""
    tr_all, te_all = d[d["split"] == "train"], d[d["split"] == "test"]
    machines = np.array(sorted(d["Machine_ID"].unique()))
    rows = []
    for k, (_, held_idx) in enumerate(GroupKFold(n_splits=n_splits).split(machines, groups=machines), start=1):
        held = set(machines[held_idx])
        tr = tr_all[~tr_all["Machine_ID"].isin(held)]
        ev = te_all[te_all["Machine_ID"].isin(held)]
        for model_name in MODEL_NAMES:
            recs, *_ = fit_score(model_name, cw, fs_name, tr, {"unseen": ev})
            r = recs[0]
            rows.append({"feature_set": fs_name, "fold": k, "held_out_machines": len(held), "eval_rows": len(ev), "model": model_name,
                         "macro_f1": r["macro_f1"], "balanced_accuracy": r["balanced_accuracy"], "f1_High": r["f1_High"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
def main() -> Dict:
    t_start = time.time()
    utils.ensure_dirs()
    audit = data_audit.main()
    data = build_dataset()
    d, net_ref = data["df"], data["net_ref"]
    tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
    log.info("Rows: train %d | validation %d | test %d", len(tr), len(va), len(te))

    # ---- 3. experiments
    results, store = run_experiments(d)
    results.to_csv(utils.METRICS_DIR / "experiment_results.csv", index=False)

    # ---- 4. selection (validation only)
    final_sel = select_final(results, "F2_raw_plus_derived")
    ctx_sel = select_final(results, "F3_context_only")
    log.info("FINAL (deployed) model: %s | class_weight=%s | validation macro-F1 %.4f", final_sel["model"], final_sel["class_weight"], final_sel["val_macro_f1"])
    log.info("CONTEXT-ONLY model (no Error_Rate/Speed): %s | validation macro-F1 %.4f", ctx_sel["model"], ctx_sel["val_macro_f1"])
    pre_f, mdl_f, proba_f = store[(final_sel["feature_set"], final_sel["model"], final_sel["class_weight"])]
    pre_c, mdl_c, proba_c = store[(ctx_sel["feature_set"], ctx_sel["model"], ctx_sel["class_weight"])]

    # ---- 5. stability
    log.info("Stability checks (time folds, unseen machines) ...")
    time_folds = pd.concat([time_fold_stability(d, "F2_raw_plus_derived"), time_fold_stability(d, "F3_context_only")])
    mach = pd.concat([machine_holdout(d, "F2_raw_plus_derived"), machine_holdout(d, "F3_context_only")])
    time_folds.to_csv(utils.METRICS_DIR / "stability_time_folds.csv", index=False)
    mach.to_csv(utils.METRICS_DIR / "stability_unseen_machines.csv", index=False)

    # ---- artifacts needed by the explainers
    joblib.dump(pre_f, utils.PREPROCESSOR)
    joblib.dump(mdl_f, utils.BEST_MODEL)
    joblib.dump(pre_c, utils.CONTEXT_PREPROCESSOR)
    joblib.dump(mdl_c, utils.CONTEXT_MODEL)

    ranges = {c: {"min": float(tr[c].min()), "max": float(tr[c].max()), "median": float(tr[c].median()),
                  "p01": float(tr[c].quantile(.01)), "p99": float(tr[c].quantile(.99))} for c in utils.NUMERIC_RAW}
    test_m = evaluation.compute_metrics(te["y"].values, proba_f["test"])
    val_m = evaluation.compute_metrics(va["y"].values, proba_f["val"])
    ctx_test_m = evaluation.compute_metrics(te["y"].values, proba_c["test"])
    ctx_val_m = evaluation.compute_metrics(va["y"].values, proba_c["val"])
    base_test = evaluation.majority_baseline(tr["y"].values, te["y"].values)

    meta = {
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "class_order": CLASS_ORDER, "raw_inputs": RAW_INPUTS, "operation_modes": utils.OPERATION_MODES,
        "network_reference": net_ref, "feature_ranges": ranges,
        "feature_sets": {k: {"numeric": v["numeric"], "categorical": v["categorical"], "description": v["description"]} for k, v in FEATURE_SETS.items()},
        "split": {"train_rows": len(tr), "val_rows": len(va), "test_rows": len(te),
                  "train_period": [str(tr["datetime"].min()), str(tr["datetime"].max())],
                  "val_period": [str(va["datetime"].min()), str(va["datetime"].max())],
                  "test_period": [str(te["datetime"].min()), str(te["datetime"].max())], "method": "chronological by file order, no shuffling"},
        "final_model": {**final_sel, "hyperparameters": {k: str(v) for k, v in mdl_f.get_params().items()},
                        "trained_on": "training split only", "validation_metrics": val_m, "test_metrics": test_m,
                        "uses_label_defining_inputs": RULE_INPUTS,
                        "warning": ("This model receives Error_Rate_% and Production_Speed, the two columns from which Efficiency_Status is derived. "
                                    "Its high score shows that it recovers the labelling rule; it is NOT evidence that sensors or the network predict efficiency.")},
        "context_model": {**ctx_sel, "validation_metrics": ctx_val_m, "test_metrics": ctx_test_m,
                          "note": "Same task WITHOUT Error_Rate_% and Production_Speed. Tests whether sensor/network/quality data carry any signal."},
        "baseline_majority_test": base_test,
        "label_rule": audit["label_derivation"]["rule"], "label_rule_agreement": audit["label_derivation"]["rule_agreement"],
        "simulated_live_monitoring_note": "The CSV is historical; 'live monitoring' in the dashboard replays recorded rows.",
    }
    save_json(meta, utils.METADATA)

    # ---- 6. predictions for every row (for the dashboard) + explainability
    best = EfficiencyPredictor.load("best")
    ctx = EfficiencyPredictor.load("context")
    pf = best.predict_frame(d)
    d = d.join(pf)
    pc = ctx.predict_frame(d)
    d["ctx_pred_class"], d["ctx_max_prob"] = pc["pred_class"], pc["max_prob"]
    for c in CLASS_ORDER:
        d[f"ctx_p_{c}"] = pc[f"p_{c}"]
    # verification: saved-artifact predictions equal the in-memory ones
    assert np.allclose(best.predict_proba(te), proba_f["test"], atol=1e-9), "saved artifacts disagree with training predictions"

    log.info("Permutation importance + Shapley attributions ...")
    sub_idx = va.sample(8000, random_state=utils.RANDOM_STATE).index
    pi_best = explainability.permutation_importance_raw(best, va.loc[sub_idx], va.loc[sub_idx, "y"].values)
    pi_ctx = explainability.permutation_importance_raw(ctx, va.loc[sub_idx], va.loc[sub_idx, "y"].values)
    pi_best.to_csv(utils.METRICS_DIR / "permutation_importance_final.csv", index=False)
    pi_ctx.to_csv(utils.METRICS_DIR / "permutation_importance_context.csv", index=False)

    rng = np.random.default_rng(utils.RANDOM_STATE)
    bg = tr.sample(300, random_state=utils.RANDOM_STATE)
    # global Shapley: 60 validation records of each class
    glob_rows = []
    for ci, cname in enumerate(CLASS_ORDER):
        pool = va[va["y"] == ci]
        for _, r in pool.sample(min(60, len(pool)), random_state=utils.RANDOM_STATE).iterrows():
            for target_class in (0, 2):                      # explain P(Low) and P(High)
                a = explainability.monte_carlo_shapley(best, r.to_dict(), bg, target_class, n_perm=25, seed=int(rng.integers(1e9)))
                for _, ar in a.iterrows():
                    glob_rows.append({"actual_class": cname, "explained_class": CLASS_ORDER[target_class], "feature": ar["feature"],
                                      "abs_attribution": abs(ar["attribution"]), "attribution": ar["attribution"]})
    glob = pd.DataFrame(glob_rows)
    glob.groupby(["explained_class", "feature"])["abs_attribution"].mean().reset_index().to_csv(utils.METRICS_DIR / "shapley_global_final.csv", index=False)
    class_prof = explainability.class_profile(d, utils.NUMERIC_RAW)
    class_prof.to_csv(utils.METRICS_DIR / "class_feature_means.csv")

    # ---- calibration tables
    rel = pd.concat([evaluation.reliability_table(proba_f["test"], te["y"].values, i) for i in range(3)])
    rel.to_csv(utils.METRICS_DIR / "reliability_final_test.csv", index=False)
    rel_c = pd.concat([evaluation.reliability_table(proba_c["test"], te["y"].values, i) for i in range(3)])
    rel_c.to_csv(utils.METRICS_DIR / "reliability_context_test.csv", index=False)

    # ---- dashboard table
    keep = (["obs_seq", "datetime", "Date", "Timestamp", "Machine_ID", "Operation_Mode"] + utils.NUMERIC_RAW +
            ["energy_per_unit", "error_to_speed_ratio", "network_stress_index", "network_quality", TARGET, "split",
             "p_Low", "p_Medium", "p_High", "pred_class", "max_prob", "ctx_pred_class"])
    out = d[keep].copy()
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(4)
    out.to_csv(utils.CLEAN_CSV, index=False)

    # ---- figures
    log.info("Figures ...")
    figures.make_all(d, results, time_folds, mach, pi_best, pi_ctx, glob, rel, rel_c, test_m, ctx_test_m, proba_f["test"], proba_c["test"],
                     te["y"].values, audit)
    save_json({"cleaning_log": data["clean_log"], "elapsed_seconds": time.time() - t_start}, utils.METRICS_DIR / "cleaning_and_run_log.json")
    log.info("Done in %.0f s", time.time() - t_start)
    return {"meta": meta, "results": results, "d": d}


if __name__ == "__main__":
    main()
