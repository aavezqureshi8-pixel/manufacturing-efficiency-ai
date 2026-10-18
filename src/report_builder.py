"""
report_builder.py - writes research_paper.md and executive_summary.md from the REAL results files.

Run after training:  python -m src.report_builder

Every number is read from outputs/metrics/*, models/model_metadata.json or data/processed/clean_data.csv.gz,
so the documents cannot disagree with the analysis. Interpretive sentences were written for THIS dataset;
if you retrain on different data, re-read them.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from . import utils
from .dashboard_logic import machine_ranking
from .utils import CLASS_ORDER, NUMERIC_RAW, SHORT, TARGET


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.astype(object).itertuples(index=False):
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def fp(p: float) -> str:
    """p-value formatting."""
    return "< 0.0001" if p < 1e-4 else f"{p:.3f}"


def fe(v: float) -> str:
    """effect-size formatting (scientific for tiny values)."""
    return f"{v:.4f}" if v >= 1e-3 else f"{v:.1e}"


def P(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def build() -> None:
    meta = utils.load_json(utils.METADATA)
    audit = utils.load_json(utils.METRICS_DIR / "data_audit.json")
    res = pd.read_csv(utils.METRICS_DIR / "experiment_results.csv")
    tf = pd.read_csv(utils.METRICS_DIR / "stability_time_folds.csv")
    mh = pd.read_csv(utils.METRICS_DIR / "stability_unseen_machines.csv")
    pi_b = pd.read_csv(utils.METRICS_DIR / "permutation_importance_final.csv")
    pi_c = pd.read_csv(utils.METRICS_DIR / "permutation_importance_context.csv")
    sh = pd.read_csv(utils.METRICS_DIR / "shapley_global_final.csv")
    cmeans = pd.read_csv(utils.METRICS_DIR / "class_feature_means.csv", index_col=0)
    d = pd.read_csv(utils.CLEAN_CSV, parse_dates=["datetime"])

    n = len(d)
    ld = audit["label_derivation"]
    fm, cm = meta["final_model"], meta["context_model"]
    tm, ctm = fm["test_metrics"], cm["test_metrics"]
    base = meta["baseline_majority_test"]
    td = audit["target_distribution"]
    tp = audit["target_distribution_pct"]
    sp = meta["split"]
    models = ["Logistic Regression", "Random Forest", "Gradient Boosting"]

    # ---------------- derived statistics
    fvl = audit["feature_vs_label"]
    mode_ct = pd.crosstab(d["Operation_Mode"], d[TARGET], normalize="index")[CLASS_ORDER] * 100
    net_ct = pd.crosstab(d["network_quality"], d[TARGET], normalize="index").reindex(["Good", "Fair", "Poor"])[CLASS_ORDER] * 100
    quint = pd.qcut(d["Network_Latency_ms"], 5)
    q_ct = pd.crosstab(quint, d[TARGET], normalize="index")[CLASS_ORDER] * 100
    week = d["datetime"].dt.to_period("W")
    wk_low = d.groupby(week)[TARGET].apply(lambda x: (x == "Low").mean())
    rk = machine_ranking(d)
    n_above, n_below = int((rk["vs_fleet"] == "above fleet").sum()), int((rk["vs_fleet"] == "below fleet").sum())
    corr_ps = d["Power_Consumption_kW"].corr(d["Production_Speed_units_per_hr"])
    ind = audit["independence_tests"]

    def t(fs, model, cw, split):
        r = res[(res.feature_set == fs) & (res.model == model) & (res.class_weight == cw) & (res.split == split)]
        return r.iloc[0]

    # ---------------- tables
    cls_tbl = pd.DataFrame({"Class": CLASS_ORDER, "Records": [td[c] for c in CLASS_ORDER], "Share": [f"{tp[c]:.2f}%" for c in CLASS_ORDER]})
    means_tbl = cmeans.reset_index().rename(columns={"index": "Measurement"})
    means_tbl["Measurement"] = means_tbl["Measurement"].map(lambda c: SHORT.get(c, c))
    means_tbl = means_tbl[["Measurement", "Low", "Medium", "High", "std_diff_High_minus_Low"]].rename(columns={"std_diff_High_minus_Low": "(High-Low)/SD"})
    assoc = pd.DataFrame([{"Measurement": SHORT[c], "Kruskal-Wallis p": fp(v['kruskal_p']), "Eta-squared": fe(v['eta_squared']),
                           "Mutual information": f"{v['mutual_information']:.4f}"} for c, v in fvl.items()])

    comp_rows = []
    for m in models:
        for cw in ("balanced", "none"):
            v, te = t("F2_raw_plus_derived", m, cw, "val"), t("F2_raw_plus_derived", m, cw, "test")
            comp_rows.append({"Model": m, "Class weight": cw, "Val macro-F1": v.macro_f1, "Test accuracy": te.accuracy, "Test bal. acc.": te.balanced_accuracy,
                              "Test macro-P": te.macro_precision, "Test macro-R": te.macro_recall, "Test macro-F1": te.macro_f1, "Test weighted-F1": te.weighted_f1,
                              "Train s": f"{te.train_seconds:.1f}", "Predict ms/1k rows": f"{te.predict_ms_per_1k:.1f}"})
    comp_tbl = pd.DataFrame(comp_rows)
    pcls = []
    for m in models:
        te = t("F2_raw_plus_derived", m, "balanced", "test")
        pcls.append({"Model (balanced)": m, **{f"P {c}": getattr(te, f"precision_{c}") for c in CLASS_ORDER},
                     **{f"R {c}": getattr(te, f"recall_{c}") for c in CLASS_ORDER}, **{f"F1 {c}": getattr(te, f"f1_{c}") for c in CLASS_ORDER}})
    pcls_tbl = pd.DataFrame(pcls)

    abl = []
    for fs, label in [("F1_raw_all", "F1 raw measurements"), ("F2_raw_plus_derived", "F2 raw + derived (deployed)"),
                      ("F3_context_only", "F3 context only (no Error rate, no Speed)"), ("F4_context_plus_history", "F4 context + past-only history"),
                      ("F5_rule_inputs_only", "F5 Error rate + Speed only")]:
        for m in (models if fs != "F5_rule_inputs_only" else ["Decision Tree (depth 4)"]):
            cwk = "balanced" if fs != "F5_rule_inputs_only" else "none"
            te = t(fs, m, cwk, "test")
            abl.append({"Feature set": label, "Model": m, "Test bal. acc.": te.balanced_accuracy, "Test macro-F1": te.macro_f1, "Recall High": te.recall_High, "F1 High": te.f1_High})
    abl.append({"Feature set": "Baseline", "Model": "Always predict Low", "Test bal. acc.": base["balanced_accuracy"], "Test macro-F1": base["macro_f1"],
                "Recall High": base["recall_High"], "F1 High": base["f1_High"]})
    abl_tbl = pd.DataFrame(abl)

    cw_rows = []
    for fs, lab in [("F2_raw_plus_derived", "F2"), ("F3_context_only", "F3")]:
        for m in models:
            for cw in ("balanced", "none"):
                te = t(fs, m, cw, "test")
                cw_rows.append({"Feature set": lab, "Model": m, "Class weight": cw, "Recall Low": te.recall_Low, "Recall Medium": te.recall_Medium,
                                "Recall High": te.recall_High, "Precision High": te.precision_High, "Bal. acc.": te.balanced_accuracy})
    cw_tbl = pd.DataFrame(cw_rows)

    tfs = tf.groupby(["feature_set", "model"])["macro_f1"].agg(["mean", "std", "min", "max"]).reset_index()
    tfs["feature_set"] = tfs["feature_set"].map({"F2_raw_plus_derived": "F2 (with Error/Speed)", "F3_context_only": "F3 (without)"})
    tfs.columns = ["Feature set", "Model", "Mean macro-F1", "SD", "Min", "Max"]
    mhs = mh.groupby(["feature_set", "model"])[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"]).reset_index()
    mhs.columns = ["Feature set", "Model", "Macro-F1 mean", "Macro-F1 SD", "Bal. acc. mean", "Bal. acc. SD"]
    mhs["Feature set"] = mhs["Feature set"].map({"F2_raw_plus_derived": "F2 (with Error/Speed)", "F3_context_only": "F3 (without)"})

    pi_tbl = pi_b.assign(Measurement=pi_b["feature"].map(SHORT))[["Measurement", "f1_drop_mean", "f1_drop_std"]].rename(
        columns={"f1_drop_mean": "Macro-F1 drop (deployed)", "f1_drop_std": "SD"})
    pi_ctx_tbl = pi_c.assign(Measurement=pi_c["feature"].map(SHORT))[["Measurement", "f1_drop_mean", "f1_drop_std"]].rename(
        columns={"f1_drop_mean": "Macro-F1 drop (context-only)", "f1_drop_std": "SD"})
    sh_low = sh[sh.explained_class == "Low"].sort_values("abs_attribution", ascending=False).head(3)
    sh_high = sh[sh.explained_class == "High"].sort_values("abs_attribution", ascending=False).head(3)

    cmx = np.array(tm["confusion_matrix"])
    cmx_c = np.array(ctm["confusion_matrix"])
    cm_tbl = pd.DataFrame(cmx, index=[f"Actual {c}" for c in CLASS_ORDER], columns=[f"Pred {c}" for c in CLASS_ORDER]).reset_index().rename(columns={"index": ""})
    cm_c_tbl = pd.DataFrame(cmx_c, index=[f"Actual {c}" for c in CLASS_ORDER], columns=[f"Pred {c}" for c in CLASS_ORDER]).reset_index().rename(columns={"index": ""})

    fs_tbl = pd.DataFrame([{"Set": k, "Numeric features": len(v["numeric"]), "Categorical": len(v["categorical"]), "Description": v["description"]}
                           for k, v in meta["feature_sets"].items()])
    clean_log_rows = pd.DataFrame(utils.load_json(utils.METRICS_DIR / "cleaning_and_run_log.json")["cleaning_log"])
    top_low_lines = []
    for c in ["Error_Rate_%", "Production_Speed_units_per_hr"]:
        top_low_lines.append(f"{SHORT[c]}: mean {cmeans.loc[c, 'Low']:.2f} (Low), {cmeans.loc[c, 'Medium']:.2f} (Medium), {cmeans.loc[c, 'High']:.2f} (High)")
    others = [c for c in NUMERIC_RAW if c not in utils.RULE_INPUTS]
    max_other_diff = float(cmeans.loc[others, "std_diff_High_minus_Low"].abs().max())
    max_other_eta = max(fvl[c]["eta_squared"] for c in others)
    min_other_p = min(fvl[c]["kruskal_p"] for c in others)
    best_cls = tm

    # =====================================================================================
    paper = f"""# AI-Based Manufacturing Efficiency Classification Using Sensor, Production, and 6G Network Data

*Domain: Artificial Intelligence · Machine Learning · Smart Manufacturing · Industrial IoT · Network Analytics*

> **Scope statement.** The project brief names Unified Mentor and Thales Group. This work does **not** claim approval, endorsement or deployment by either organisation.
> All numbers in this paper are generated from the supplied file `Thales_Group_Manufacturing.csv` by the code in this repository (`python -m src.train_models`, then `python -m src.report_builder`).

---

## 1. Title
AI-Based Manufacturing Efficiency Classification Using Sensor, Production, and 6G Network Data

## 2. Abstract
We analysed {n:,} one-minute records from {audit['unique']['machines']} machines (temperature, vibration, power, network latency, packet loss, defect rate, production speed, maintenance score, error rate, operation mode)
and trained models to classify the observed efficiency state as High, Medium or Low. The classes are strongly imbalanced (Low {tp['Low']:.1f}%, Medium {tp['Medium']:.1f}%, High {tp['High']:.1f}%).
**The audit shows that the label is a deterministic rule of two columns:** High = error rate ≤ 2 % and speed > 400 units/hr; Medium = error rate ≤ 5 % and speed > 200; otherwise Low. The rule matches {100*ld['rule_agreement']:.3f}% of records.
Consequently, models that receive those two columns reach a test macro-F1 of {tm['macro_f1']:.4f} (Random Forest, chronological test set of {sp['test_rows']:,} records),
but this demonstrates *recovery of a labelling rule*, not prediction of efficiency from sensors or network conditions.
When the two columns are withheld, all three algorithms (Logistic Regression, Random Forest, Gradient Boosting) fall to chance level (best test balanced accuracy {ctm['balanced_accuracy']:.3f} versus {1/3:.3f} expected by chance),
with and without past-only history features, and also for machines never seen in training. Network latency, packet loss, temperature, vibration, power, defect rate, maintenance score and operation mode
show no association with efficiency (numeric measurements: all Kruskal-Wallis p ≥ {min_other_p:.2f}, eta-squared ≤ {fe(max_other_eta)}; operation mode: chi-square p = {ind['operation_mode_vs_efficiency']['p']:.2f}). We provide a Streamlit dashboard with prediction, probabilities, explanations,
machine-level views and a clearly labelled *simulated* live replay. The results do not show that the system can predict future breakdowns.

## 3. Introduction
Smart factories combine Industrial IoT sensors with low-latency wireless links (Lee et al., 2015; Saad et al., 2020). Dashboards typically describe what has already happened.
Machine learning can classify the current state automatically, but its value depends on whether the label reflects an independently measured outcome and whether the input data really carry information about it.
This paper builds a classifier for High/Medium/Low efficiency and, equally importantly, tests what that classifier has actually learned.

## 4. Background and context
Manufacturing efficiency can fluctuate with sensor deviations, network instability, quality and error changes, speed changes and operating conditions. Network metrics (latency, packet loss) are of interest because
unreliable connectivity may delay control signals or data. Whether such effects exist in a given plant is an empirical question; in this dataset it is tested directly (sections 9 and 15).
Evaluation of time-ordered data requires chronological splitting (Bergmeir & Benítez, 2012), and label leakage is a well-documented cause of over-optimistic results (Kaufman et al., 2012; Kapoor & Narayanan, 2023).

## 5. Problem statement
Manufacturing teams face delayed detection of efficiency degradation, manual interpretation of many metrics, a lack of automated and interpretable assessment, and possible production losses from delayed action.
The goal is a machine-learning solution that analyses measurements, predicts the efficiency class, shows confidence, and explains influential factors.

## 6. Research objectives
1. Audit the dataset (structure, quality, timeline, relationships, outliers).
2. Build a reproducible preprocessing and feature-engineering pipeline without leakage.
3. Compare Logistic Regression, Random Forest and Gradient Boosting under identical chronological evaluation, with class-imbalance handling.
4. Quantify the contribution of sensor, network, production and quality measurements, including whether the label can be predicted *without* the label-defining columns.
5. Provide global and local explanations, calibrated-confidence discussion, and an interactive dashboard.
6. State clearly what the results do and do not show.

## 7. Dataset description
| Property | Value |
|---|---|
| File | `Thales_Group_Manufacturing.csv` (verified from the file; the brief's schema was not assumed) |
| Rows × columns | {audit['shape']['rows']:,} × {audit['shape']['columns']} (matches the 14 expected columns: {audit['columns_match_expected']}) |
| Period | {audit['datetime']['first']} to {audit['datetime']['last']} (1 record per minute) |
| Machines | {audit['unique']['machines']} (IDs {audit['unique']['machine_id_range'][0]}-{audit['unique']['machine_id_range'][1]}); {audit['machine_coverage']['rows_per_machine_min']:,}-{audit['machine_coverage']['rows_per_machine_max']:,} records each |
| Operation modes | {', '.join(f'{k} {v:,}' for k, v in audit['unique']['operation_modes'].items())} |
| Missing values / duplicate rows | {sum(audit['missing_values'].values())} / {audit['duplicate_rows']} |
| Target | `Efficiency_Status` |

{md_table(cls_tbl)}

**Data-quality findings.**
* *Uniform, independent measurements.* Every numeric column is distributed almost exactly uniformly between fixed limits (largest Kolmogorov-Smirnov distance to a uniform distribution {max(v['ks_distance_to_uniform'] for v in audit['outliers_and_shape'].values()):.4f}), and the largest absolute correlation between any two measurements is {audit['feature_correlation_max_abs_offdiagonal']:.3f}. Real sensor streams are rarely like this; the data appear simulated.
* *Outliers.* There are no IQR outliers and no values outside plausible physical limits ({sum(v['iqr_outliers'] for v in audit['outliers_and_shape'].values())} and {sum(v['outside_plausible_limits'] for v in audit['outliers_and_shape'].values())}).
* *Timeline anomaly.* File order follows one-minute steps except at row {list(audit['datetime']['file_order_is_time_order_except'].keys())[0]}, where the clock jumps back {abs(list(audit['datetime']['file_order_is_time_order_except'].values())[0]):.0f} minutes. As a result the date label `01-03-2025` carries {audit['datetime']['rows_per_date_not_1440']['01-03-2025']:,} records (two days) and {audit['datetime']['duplicated_timestamps']:,} timestamps occur twice. {audit['datetime']['likely_explanation'].split('. ', 1)[1]}
* *Sparse machine histories.* Each minute holds one record from a randomly different machine, so the median gap between two records of the same machine is {audit['machine_coverage']['median_gap_between_records_of_same_machine_minutes']:.0f} minutes (mean {audit['machine_coverage']['mean_gap_minutes']:.0f}).

## 8. Data cleaning and preprocessing
{md_table(clean_log_rows[['step', 'detail']])}

* No missing values, duplicates or invalid values were found, but the code handles them: duplicates are removed only when all 14 columns are identical; impossible or non-numeric values become missing and are imputed with **training** medians inside the pipeline; unexpected `Operation_Mode` categories are encoded as "no mode" (all-zero one-hot) and flagged in the dashboard.
* `Date` + `Timestamp` were combined into a datetime (format dd-mm-YYYY HH:MM:SS). Because of the timeline anomaly, ordering, splitting and history features use **file order**, which the audit verified to be chronological; the recorded date is retained for display.
* `Machine_ID` is preserved for filtering but is **not** a model input (an identifier would not generalise to new machines).
* `Operation_Mode` is one-hot encoded with fixed categories; numeric features are median-imputed, and standardised **only** for Logistic Regression (tree ensembles do not need scaling).
* The preprocessor is fitted on the training period only and saved (`models/preprocessor.joblib`); the dashboard applies the same object. A test confirms that the scaler's mean equals the training mean and not the full-data mean.

## 9. Exploratory data analysis
### 9.1 Class balance
Figure `outputs/figures/01_class_distribution.png`. High records are rare ({td['High']:,} of {n:,}); accuracy alone would be misleading (always predicting Low gives {P(base['accuracy'])} accuracy on the test period but a balanced accuracy of {base['balanced_accuracy']:.3f}).

### 9.2 Sensors (temperature, vibration, power)
![sensors](outputs/figures/02_sensors_by_class.png)

Mean temperature, vibration and power are practically identical across classes (see table below; standardised High−Low differences of {cmeans.loc[['Temperature_C','Vibration_Hz','Power_Consumption_kW'],'std_diff_High_minus_Low'].abs().max():.3f} SD at most).

{md_table(means_tbl, "{:,.3f}")}

### 9.3 Network
![network](outputs/figures/03_network_vs_efficiency.png)

Class shares by latency quintile range only from {q_ct['Low'].min():.1f}% to {q_ct['Low'].max():.1f}% Low and {q_ct['High'].min():.2f}% to {q_ct['High'].max():.2f}% High: no gradient. By network-quality band (tertiles of a latency + packet-loss index):

{md_table(net_ct.reset_index().rename(columns={'network_quality': 'Network quality'}), "{:,.2f}")}

(percentage of records in each class; `outputs/figures/11_efficiency_by_network_quality.png`).

### 9.4 Production, quality and maintenance
![production](outputs/figures/04_production_by_class.png)

Only production speed and error rate differ between classes ({'; '.join(top_low_lines)}). Defect rate and maintenance score do not.
Power versus production speed has a correlation of {corr_ps:.3f} (`05_power_vs_speed.png`): in this data power does not rise with output.

![rule](outputs/figures/06_speed_vs_error_rule.png)

The scatter plot of speed against error rate shows the classes as rectangular regions bounded by speed 200/400 and error rate 2 %/5 %.

### 9.5 Statistical association of every measurement with the class
{md_table(assoc)}

Production speed and error rate are the only measurements with a non-negligible relationship (eta-squared {fvl['Production_Speed_units_per_hr']['eta_squared']:.3f} and {fvl['Error_Rate_%']['eta_squared']:.3f}).
For the other seven measurements eta-squared is at most {fe(max_other_eta)} (essentially zero) and the smallest Kruskal-Wallis p-value is {min_other_p:.3f}. With 7 tests, a smallest p of 0.158 is entirely unremarkable.

### 9.6 Machines, operation modes and time
* **Operation mode:** Low share {', '.join(f'{m} {mode_ct.loc[m, "Low"]:.1f}%' for m in mode_ct.index)}; chi-square p = {ind['operation_mode_vs_efficiency']['p']:.3f} (no difference).
* **Machines:** Low share per machine ranges from {P(rk['low_share'].min())} to {P(rk['low_share'].max())} (fleet {P(rk['fleet_low_share'].iloc[0])}). Using 95% Wilson intervals, {n_above} machines are distinguishable above and {n_below} below the fleet average (with {audit['unique']['machines']} machines, about {0.05 * audit['unique']['machines']:.1f} intervals are expected to exclude the fleet value by chance alone); the chi-square test of machine versus class gives p = {ind['machine_vs_efficiency']['p']:.3f}, i.e. no evidence of machine-level differences beyond chance (`08_machine_low_share.png`).
* **Time:** weekly Low share ranges from {P(wk_low.min())} to {P(wk_low.max())}; chi-square p = {ind['week_vs_efficiency']['p']:.3f}. No degradation period is visible (`09_efficiency_over_time.png`).

## 10. Feature engineering
{md_table(fs_tbl)}

Implemented and kept: *energy per unit* (power ÷ speed), *error-to-speed ratio*, and a *network stress index* (standardised latency + standardised packet loss, reference mean/SD fitted on training data; tertiles define Good/Fair/Poor).
Zero denominators return missing values rather than infinity.
Evaluated and **excluded from the deployed model** (set F4): past-only rolling means and standard deviations (window of the previous {5} records of the same machine, a record-based window because records of one machine are ~{audit['machine_coverage']['median_gap_between_records_of_same_machine_minutes']:.0f} minutes apart) of eight measurements, the speed change between the two previous records, minutes since the previous record, hour of day and day of week.
They gave no benefit (section 15), they cannot be computed for a single manually entered observation, and the audit found no time structure, so they are not justified by the data.

## 11. Methodology
* **Chronological split** (no shuffling): training {sp['train_rows']:,} records ({sp['train_period'][0][:10]} to {sp['train_period'][1][:10]}), validation {sp['val_rows']:,} ({sp['val_period'][0][:10]} to {sp['val_period'][1][:10]}), test {sp['test_rows']:,} ({sp['test_period'][0][:10]} to {sp['test_period'][1][:10]}). Because of the repeated date label, "periods" are defined by file order.
* All models and preprocessors are fitted on training data only; the final model is chosen from **validation** results; test results are reported afterwards.
* **Three evaluation set-ups, each measuring something different:** (i) the chronological test set measures performance on *later* records of the same machines; (ii) five expanding-window time folds on the first 85% measure stability over time; (iii) five folds in which 10 machines are withheld from training and evaluation uses the *test period* records of those machines measure transfer to unseen machines. None of them measures prediction of future *events* such as breakdowns, because the label describes the *current* observed state.
* **Experiment grid:** 3 algorithms × 4 feature sets (F1-F4) × 2 class-weight settings, plus a depth-4 decision tree on the two label-defining columns (F5) and an always-Low baseline.
* Selection rule (recorded in the model metadata): {fm['rule']}.

## 12. Machine-learning models
| Model | Configuration |
|---|---|
| Logistic Regression | multinomial, L2 (C = 1), standardised inputs, max 3,000 iterations |
| Random Forest | 100 trees, max depth 12, min leaf 5, `balanced_subsample` weights when weighted |
| Gradient Boosting | scikit-learn `HistGradientBoostingClassifier`, 150 iterations, learning rate 0.1 (XGBoost was not used; it is optional in the brief and was not installed in the build environment) |

## 13. Data-leakage prevention
**Finding.** `Efficiency_Status` is derived from `Error_Rate_%` and `Production_Speed_units_per_hr`: a depth-4 tree on only these two columns scores {100*ld['tree_depth4_test_accuracy_by_inputs']['both (Error_Rate_% + Production_Speed)']:.1f}% on the later 30% of the data, while a tree on the other seven measurements plus mode scores {100*ld['tree_depth4_test_accuracy_by_inputs']['all other 7 measurements + mode (no Error_Rate_%, no Speed)']:.2f}% (majority class: {100*ld['majority_class_accuracy_test']:.2f}%).
The explicit rule agrees with {100*ld['rule_agreement']:.3f}% of records; the {ld['rule_mismatches']} exceptions have an error rate displayed as exactly 5.000, consistent with rounding of stored values.

Because the label is a function of two *input* columns, supplying them to a model is not "using the target" in the strict sense, but it makes the task trivial. We therefore (a) report results both with and without those columns,
(b) label the deployed model accordingly, and (c) never claim that its score demonstrates predictive power from sensors or the network.

**Safeguards.** The target is never an input; the split is chronological; imputer, scaler, encoder and network-reference statistics are fitted on training rows only; no resampling is used before splitting (none was used at all);
history features use `shift(1)` before rolling windows and never the current or future rows, which is **tested** by changing a single row and confirming that its own history features are unchanged while the same machine's next record does change;
`Machine_ID` is not a feature; derived features use only the same record.
*Development note:* an earlier draft of a "speed change" feature subtracted the previous speed from the **current** speed. That leaked a label-defining column: in the no-signal feature set one model's validation balanced accuracy rose above chance (about 0.49). It was detected from that anomaly, replaced by the change between the two *previous* records, and the stricter single-row test above was added. All reported results use the corrected feature.

## 14. Class-imbalance handling
High is {tp['High']:.1f}% of the data. We compared (1) no weighting and (2) class-weighted training for every algorithm (class weights inversely proportional to class frequency). Training-only resampling (e.g. SMOTE) was **not** used: the imbalanced-learn library was not installed, and for a deterministic threshold rule synthetic examples are not justified (class weights already restore the minority class's influence).
Per-class results on the test set:

{md_table(cw_tbl)}

With informative features, weighting helps Logistic Regression's recall of High ({t('F2_raw_plus_derived','Logistic Regression','balanced','test').recall_High:.3f} vs {t('F2_raw_plus_derived','Logistic Regression','none','test').recall_High:.3f}) at the cost of High precision ({t('F2_raw_plus_derived','Logistic Regression','balanced','test').precision_High:.3f} vs {t('F2_raw_plus_derived','Logistic Regression','none','test').precision_High:.3f}); tree ensembles are nearly insensitive.
With uninformative features, weighting merely redistributes errors: without weights every model predicts Low for everything (recall High 0.000, balanced accuracy 0.333); with weights some High records are "found" ({t('F3_context_only','Random Forest','balanced','test').recall_High:.3f} for Random Forest) but with precision {t('F3_context_only','Random Forest','balanced','test').precision_High:.3f}, i.e. at random.
Macro-F1 therefore *looks* higher with weights ({t('F3_context_only','Random Forest','balanced','test').macro_f1:.3f} vs {t('F3_context_only','Random Forest','none','test').macro_f1:.3f}) although balanced accuracy shows no gain; balanced accuracy and per-class precision must be read together with macro-F1.

## 15. Model evaluation
### 15.1 Model comparison (deployed feature set F2, chronological split)
{md_table(comp_tbl)}

Per-class precision, recall and F1 on the test set (class-weighted models):

{md_table(pcls_tbl)}

Classification report of the selected model on the test set:
```
{tm['classification_report']}```
Confusion matrix (selected model, test set):

{md_table(cm_tbl)}

### 15.2 What each feature set can and cannot do
{md_table(abl_tbl)}

![ablation](outputs/figures/21_feature_set_ablation.png)

### 15.3 Stability
Expanding-window time folds (macro-F1):

{md_table(tfs)}

Machines never seen in training (macro-F1 and balanced accuracy over 5 folds):

{md_table(mhs)}

![stability](outputs/figures/26_stability.png)

### 15.4 Calibration and confidence
For the selected model on the test set the top-label expected calibration error is {tm['ece_top_label']:.4f}, the multiclass Brier score {tm['brier']:.4f} and the log-loss {tm['log_loss']:.4f}.
For the context-only model the ECE is {ctm['ece_top_label']:.4f} and the Brier score {ctm['brier']:.4f}: its probabilities are spread out although the observed class frequencies hardly depend on them.
The dashboard shows P(High), P(Medium), P(Low) and the highest class probability for every prediction. **The highest class probability is a model score, not a guaranteed probability of being correct**; even a small ECE on one test period does not guarantee calibration elsewhere or near the rule boundaries.
The dashboard warns when the top two probabilities are close, when inputs are outside the training range, physically impossible, missing or of an unknown category.

![calibration](outputs/figures/25_calibration.png)

## 16. Results and discussion
1. **With Error rate and Speed available**, Random Forest ({fm['class_weight']} weights) was selected on validation macro-F1 ({fm['val_macro_f1']:.4f}) and scored macro-F1 {tm['macro_f1']:.4f}, balanced accuracy {tm['balanced_accuracy']:.4f} and High-class F1 {tm['f1_High']:.4f} on the test set ({int(cmx.sum()):,} records, {int(cmx[2].sum())} High). Gradient Boosting was within {abs(t('F2_raw_plus_derived','Random Forest','balanced','test').macro_f1 - t('F2_raw_plus_derived','Gradient Boosting','balanced','test').macro_f1):.4f} macro-F1 and trained {t('F2_raw_plus_derived','Random Forest','balanced','test').train_seconds / max(t('F2_raw_plus_derived','Gradient Boosting','balanced','test').train_seconds, 1e-9):.0f}× faster; Logistic Regression reached {t('F2_raw_plus_derived','Logistic Regression','balanced','test').macro_f1:.3f} because the rule is a set of axis-aligned thresholds that a linear boundary can only approximate. A single depth-4 tree on the two columns is also perfect on the test set ({t('F5_rule_inputs_only','Decision Tree (depth 4)','none','test').macro_f1:.4f}), confirming that the ensembles add nothing beyond the rule.
2. **Without Error rate and Speed** the best test balanced accuracy is {max(t(fs, m, cw, 'test').balanced_accuracy for fs in ('F3_context_only', 'F4_context_plus_history') for m in models for cw in ('balanced', 'none')):.3f}, i.e. chance (0.333), for every algorithm, for sensor/network/quality features, for past-only history features, and for machines unseen in training.
3. **Network and sensor conditions show no association with efficiency in this dataset**: statistical tests (section 9.5), class-share tables (9.3) and model results (15.2) agree. This is a statement about this file. It does not show that networks cannot affect real manufacturing efficiency.
4. **No machine, mode or period stands out** beyond chance (section 9.6).
5. **Selected-model quality is stable** over time folds and unseen machines (deployed feature set: macro-F1 ≥ {tf[(tf.feature_set=='F2_raw_plus_derived') & (tf.model=='Random Forest')].macro_f1.min():.4f} for Random Forest in every time fold).
6. **Meaning of the high score.** A model that recovers a deterministic rule is correct whenever the rule holds, including on unseen machines and later periods, which is why the scores are near perfect and stable; it is not evidence about forecasting.

## 17. Feature importance and explainability
**Global.** Permutation importance (macro-F1 drop when one input is shuffled, 8,000 validation records, 5 repeats):

{md_table(pi_tbl)}

For the context-only model every drop is below {pi_c['f1_drop_mean'].max() + 0.0005:.3f} macro-F1; for a model that itself performs at chance this is negligible, and the signs vary (several are negative), so no measurement has a meaningful importance:

{md_table(pi_ctx_tbl)}

**Local and global attributions.** SHAP was not installed in the build environment. Instead, `src/explainability.py` implements a Monte-Carlo Shapley estimator (Štrumbelj & Kononenko, 2014) over the ten raw inputs with a 300-record background sample; along every sampled ordering the attributions sum exactly to the change in probability (unit-tested with a single background record).
Mean absolute attribution to P(Low): {', '.join(f"{SHORT[r.feature]} {r.abs_attribution:.3f}" for r in sh_low.itertuples())}. To P(High): {', '.join(f"{SHORT[r.feature]} {r.abs_attribution:.3f}" for r in sh_high.itertuples())}.
The dashboard explains individual predictions with these attributions and a plain-English sentence, and offers a one-measurement sensitivity curve.

**Factors associated with Low efficiency.** Low records have a mean error rate of {cmeans.loc['Error_Rate_%','Low']:.2f} % versus {cmeans.loc['Error_Rate_%','High']:.2f} % in High records and a mean speed of {cmeans.loc['Production_Speed_units_per_hr','Low']:.0f} versus {cmeans.loc['Production_Speed_units_per_hr','High']:.0f} units/hr; every other measurement differs by at most {max_other_diff:.3f} standard deviations.
**Association is not causation:** the importances show what the model uses to reproduce the label; because the label is *defined* by error rate and speed, they say nothing about physical causes of inefficiency.

![permutation](outputs/figures/23_permutation_importance.png)
![shapley](outputs/figures/24_shapley_global.png)

## 18. Manufacturing insights
* Under this label definition, efficiency is governed by error rate and production speed alone; improving "efficiency" means lowering error rate below 5 % (2 % for High) and raising speed above 200 (400 for High).
* No evidence was found that network latency or packet loss relate to efficiency; the 6G/IIoT premise of the brief cannot be supported or refuted by this file.
* Machine-level and time-level monitoring shows no distinguishable outliers; with {audit['machine_coverage']['rows_per_machine_mean']:.0f} records per machine the Wilson intervals are about ±1.8 percentage points, so only differences larger than that could be detected.
* A better target would be an independently measured outcome (e.g. overall equipment effectiveness from throughput, downtime and quality, or recorded failures) so that sensor and network data can be tested as predictors.

## 19. Recommendations
1. Treat the classifier as a **monitoring tool that reproduces the current efficiency definition**, not as an early-warning system.
2. Obtain data with an independently defined efficiency or failure outcome, real timestamps (resolving the date issue) and, if network effects matter, simultaneous network and process measurements per machine.
3. Log timestamps and machine IDs consistently so each machine has a regular history; then re-test history features and forecasting horizons.
4. Keep the dashboard's input checks, uncertainty warnings and "simulation" labelling in any deployment.
5. Re-validate against real outcomes before using any prediction to trigger maintenance or production decisions.

## 20. Limitations
* The data appear simulated (uniform, uncorrelated measurements, rule-defined label, one random machine per minute); conclusions do not transfer automatically to a real plant.
* Perfect scores reflect recovery of a labelling rule; the absence of sensor/network signal is specific to this file.
* The label describes the **current** state; nothing here predicts future breakdowns or degradation.
* The date label `01-03-2025` covers two days; the "29 February" explanation is a hypothesis.
* Gradient Boosting is scikit-learn's histogram implementation, not XGBoost; explanations use Monte-Carlo Shapley, not the SHAP library.
* Confidence is assessed on one test period; a single ECE value does not guarantee calibration under other conditions.
* The network-quality bands are tertiles of this dataset, not engineering standards. The dashboard's live mode is a replay of recorded rows.

## 21. Future scope
Independent outcome labels (downtime, defects, failures); streaming ingestion with real time stamps; per-machine models and drift monitoring; survival or time-to-failure modelling; cost-sensitive thresholds for maintenance decisions; network-side telemetry from an actual 6G/5G testbed; SHAP/XGBoost comparisons; formal calibration (isotonic or Platt) once real labels exist.

## 22. Conclusion
The pipeline, dashboard and tests are complete and reproducible. The data audit shows that the High/Medium/Low label is a deterministic function of error rate and production speed; models given those columns achieve a test macro-F1 of {tm['macro_f1']:.4f}, stable over time and across machines, while sensor, network, quality and history information — tested under identical conditions — provides no predictive signal (balanced accuracy at chance level).
The value of the project is therefore a working, explainable and honest monitoring tool and a clear specification of what data would be needed to make efficiency *prediction* from sensor and network data possible.

## 23. References
1. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
3. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.
4. Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research*, 20(177), 1-81.
5. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning*, 1321-1330.
7. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
8. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
9. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), Article 15.
10. Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
11. Lee, J., Bagheri, B., & Kao, H.-A. (2015). A cyber-physical systems architecture for Industry 4.0-based manufacturing systems. *Manufacturing Letters*, 3, 18-23.
12. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
13. Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
14. Saad, W., Bennis, M., & Chen, M. (2020). A vision of 6G wireless systems: applications, trends, technologies, and open research problems. *IEEE Network*, 34(3), 134-142.
15. Shapley, L. S. (1953). A value for n-person games. In *Contributions to the Theory of Games, Vol. II* (pp. 307-317). Princeton University Press.
16. Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. *Knowledge and Information Systems*, 41(3), 647-665.
17. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209-212.
"""
    (utils.ROOT / "research_paper.md").write_text(paper, encoding="utf-8")

    # =====================================================================================
    ctx_best_ba = max(t(fs, m, cw, 'test').balanced_accuracy for fs in ('F3_context_only', 'F4_context_plus_history') for m in models for cw in ('balanced', 'none'))
    summary = f"""# Executive Summary
## AI-Based Manufacturing Efficiency Classification Using Sensor, Production and Network Data

*Prepared for non-technical government stakeholders. Based on the supplied dataset of {n:,} records; all figures below were measured in this project unless marked as "possible".*
*No approval, endorsement or deployment by Unified Mentor or Thales Group is claimed.*

### 1. Project overview
The project built and tested an artificial-intelligence tool that reads the measurements of factory machines (temperature, vibration, power use, network delay and data loss, production speed, quality and error rates) and classifies each record as **High, Medium or Low efficiency**. It includes an interactive dashboard that shows the classification, how confident the tool is, and which inputs mattered.

### 2. The manufacturing problem
Factories can lose output when efficiency drops and the drop is noticed late. Staff must read many numbers at once, and traditional dashboards mostly show what has already happened.

### 3. Objectives
Classify efficiency automatically; show confidence and explanations; test whether machine sensors and network conditions help to identify efficiency; and be transparent about limits.

### 4. Proposed AI solution
Three standard machine-learning methods were compared. The chosen method (Random Forest) is deployed in a dashboard with prediction and confidence, machine-by-machine views, explanations, a "what-if" simulation and a replay mode that plays back recorded data. The replay mode is **not** a live connection.

### 5. Key findings (measured)
* **The efficiency label in this dataset is defined by two measurements only:** the error rate and the production speed. Records are "High" when the error rate is at most 2 % and the speed is above 400 units per hour, "Medium" when the error rate is at most 5 % and the speed above 200, and "Low" otherwise. This rule matches {100*ld['rule_agreement']:.3f}% of the {n:,} records.
* **Network and sensor conditions showed no relationship with efficiency in this dataset.** Network delay and data loss, temperature, vibration, power use, defect rate, maintenance score and operating mode were all statistically unrelated to the efficiency class.
* **No machine, operating mode or week stood out** beyond what chance would produce ({audit['unique']['machines']} machines were compared, with about {audit['machine_coverage']['rows_per_machine_mean']:.0f} records each).
* The data are very regular (every measurement is evenly spread between fixed limits), which suggests they are simulated rather than recorded in a real factory. One calendar date label (1 March 2025) contains two days of records, a data-quality problem that should be corrected at its source.
* Only {tp['High']:.1f}% of records are "High" efficiency, {tp['Medium']:.1f}% are "Medium" and {tp['Low']:.1f}% are "Low", so results were judged class by class, not only by overall accuracy.

### 6. Model performance (measured on later, unseen records)
| Question | Result |
|---|---|
| How well does the deployed tool classify when it receives the error rate and speed? | Near perfect: {P(tm['accuracy'], 2)} of {int(cmx.sum()):,} test records correct; every High record found (balanced accuracy {tm['balanced_accuracy']:.3f}). |
| Why is it so accurate? | Because it has effectively learned the rule above. This shows the tool can apply the current efficiency definition, **not** that it can foresee problems. |
| How well can the sensors and network alone identify efficiency? | No better than chance (balanced accuracy at most {ctx_best_ba:.3f}; chance is 0.333), for three different methods, also when using each machine's recent history, and for machines the tool had never seen. |
| How reliable are the confidence figures? | Close to the observed outcomes on the test period (calibration error {tm['ece_top_label']:.3f}), but the highest probability shown is a model score, not a guarantee. |

### 7. Potential operational benefits (possible, not measured)
* Automatic, consistent classification of each record under an agreed efficiency definition, with explanations and uncertainty warnings.
* Faster review of machine-level patterns through the dashboard.
* A tested template for evaluating future data sources, including whether network quality matters.
These benefits are **possible**; no savings, productivity gains or economic effects were measured or are claimed.

### 8. Relevance to smart manufacturing and Industrial IoT
The project demonstrates the full pipeline needed for data-driven monitoring: data checking, leak-free evaluation over time, explainable output and input-validation. It also shows an important lesson for Industrial IoT programmes: **a model can look perfect when its target is defined by its own inputs**, so independent outcome measures are essential.

### 9. Network reliability considerations
Network delay and packet loss were analysed explicitly. In this dataset they had no measurable relationship with efficiency. This does not show that networks cannot affect real production; it shows that this file cannot answer that question. Answering it would need simultaneous, real network and process measurements per machine.

### 10. Deployment requirements
* A real data feed (sensors or controllers through a secure gateway) using the same fields and units, with reliable clocks and machine identifiers.
* A scoring service that uses the same preprocessing and saved model, with logging and input checks.
* An independently defined outcome (for example downtime, defects or recorded failures) for training and for ongoing validation.
* Operating processes: who acts on a classification, review of uncertain cases, scheduled re-validation and model updates.

### 11. Data privacy and cybersecurity
The dataset contains machine measurements and no personal data. In a real deployment: authenticate and encrypt all connections; separate factory and analytics networks; restrict and log access to data, models and dashboards; protect the model and data against tampering (a manipulated input could change a classification); keep records only as long as needed; and review any link between machine data and individual operators for privacy compliance.

### 12. Limitations
* The data appear simulated; results may not carry over to a real factory.
* The tool classifies the **current** state. It was **not** shown to predict future breakdowns or degradation.
* Importance and explanation figures describe the model's behaviour; they do not prove causes.
* The live-monitoring page replays recorded data; no live sensor, 6G or factory connection exists.

### 13. Recommendations
1. Use the tool as a transparent monitor of the agreed efficiency definition, not as a breakdown predictor.
2. Collect real data with an independently measured efficiency or failure outcome and correct timestamps.
3. Test network effects with simultaneous network and machine data before investing in network-based efficiency claims.
4. Pilot on one line with human review of every alert and compare results with actual outcomes.

### 14. Future implementation roadmap
| Stage | Activity |
|---|---|
| 1. Data foundation | Fix timestamps; define an independent efficiency/failure outcome; collect real sensor and network data. |
| 2. Re-evaluation | Repeat this study on the new data; test forecasting of future events. |
| 3. Pilot | Connect one line through a secure gateway; run in parallel with existing monitoring; human review. |
| 4. Governance | Security review, privacy assessment, operating procedures, performance and drift monitoring. |
| 5. Scale-up | Extend only if the pilot shows measured, independently verified value. |
"""
    (utils.ROOT / "executive_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    build()
    print("research_paper.md and executive_summary.md written.")
