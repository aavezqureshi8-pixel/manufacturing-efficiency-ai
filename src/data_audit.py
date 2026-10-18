"""
data_audit.py - complete audit of the raw CSV BEFORE any modelling.

Run on its own:   python -m src.data_audit

It checks: shape, columns, dtypes, missing values, duplicates, categories, class balance,
statistics, outliers, distribution shape, date/timestamp consistency, machine coverage,
correlations, feature-vs-label association and - most importantly - whether the label
`Efficiency_Status` is a direct function of the input measurements (leakage / label-derivation check).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, export_text

from . import utils
from .utils import (CLASS_TO_INT, EXPECTED_COLUMNS, NUMERIC_RAW, OPERATION_MODES, PLAUSIBLE_LIMITS, RULE_INPUTS,
                    TARGET, get_logger, save_json)

log = get_logger()

# --------------------------------------------------------------------------------------------
# Label rule discovered in this audit (see discover_label_rule). It is kept here so that the
# audit, the report and the tests all use the same numbers. Agreement with the data is MEASURED
# every time the audit runs, it is not assumed.
# --------------------------------------------------------------------------------------------
LABEL_RULE = {
    "High":   "Error_Rate_% <= 2  AND  Production_Speed_units_per_hr > 400",
    "Medium": "Error_Rate_% <= 5  AND  Production_Speed_units_per_hr > 200  (and not High)",
    "Low":    "everything else",
}


def apply_label_rule(df: pd.DataFrame) -> pd.Series:
    """Apply the discovered rule. Used ONLY to verify the audit finding, never as a model input."""
    e, s = df["Error_Rate_%"], df["Production_Speed_units_per_hr"]
    out = np.where((e <= 2) & (s > 400), "High", np.where((e <= 5) & (s > 200), "Medium", "Low"))
    return pd.Series(out, index=df.index, name="rule_label")


# --------------------------------------------------------------------------------------------
# loading and date handling
# --------------------------------------------------------------------------------------------
def load_raw(path=utils.RAW_CSV) -> pd.DataFrame:
    """Read the CSV and check that the expected columns exist."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Dataset not found: {path}. Put Thales_Group_Manufacturing.csv in the data/ folder.") from err
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing expected columns: {missing}")
    return df


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Date + Timestamp into `datetime` (format dd-mm-YYYY HH:MM:SS, verified in the audit)
    and add `obs_seq` = row position in the file. The audit shows the file is in time order
    (one-minute steps), so obs_seq - not the label date - is used to sort, split and build history.
    """
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["Date"].astype(str).str.strip() + " " + out["Timestamp"].astype(str).str.strip(),
                                     format="%d-%m-%Y %H:%M:%S", errors="coerce")
    if out["datetime"].isna().any():          # try day/month ambiguity fallback
        alt = pd.to_datetime(out["Date"].astype(str) + " " + out["Timestamp"].astype(str), dayfirst=True, errors="coerce")
        out["datetime"] = out["datetime"].fillna(alt)
    out["obs_seq"] = np.arange(len(out))
    return out


# --------------------------------------------------------------------------------------------
# label derivation analysis
# --------------------------------------------------------------------------------------------
def discover_label_rule(df: pd.DataFrame) -> Dict:
    """
    Is Efficiency_Status a function of the measurements? Fit very shallow decision trees
    (chronological 70/30 split) and see how much of the label a few thresholds explain.
    """
    d = df.copy()
    d["mode_code"] = d["Operation_Mode"].map({m: i for i, m in enumerate(OPERATION_MODES)}).fillna(-1)
    feats = NUMERIC_RAW + ["mode_code"]
    y = d[TARGET].map(CLASS_TO_INT)
    n = len(d)
    cut = int(0.7 * n)
    res: Dict = {"tree_accuracy_by_depth_test": {}}
    for depth in (1, 2, 3, 4, 6):
        t = DecisionTreeClassifier(max_depth=depth, random_state=utils.RANDOM_STATE).fit(d[feats].iloc[:cut], y.iloc[:cut])
        res["tree_accuracy_by_depth_test"][str(depth)] = float(t.score(d[feats].iloc[cut:], y.iloc[cut:]))
    t4 = DecisionTreeClassifier(max_depth=4, random_state=utils.RANDOM_STATE).fit(d[feats].iloc[:cut], y.iloc[:cut])
    imp = pd.Series(t4.feature_importances_, index=feats).sort_values(ascending=False)
    res["depth4_tree_feature_importance"] = imp.round(4).to_dict()
    res["depth4_tree_rules"] = export_text(t4, feature_names=feats)

    # which of the two rule inputs are needed?
    def acc(cols):
        t = DecisionTreeClassifier(max_depth=4, random_state=utils.RANDOM_STATE).fit(d[cols].iloc[:cut], y.iloc[:cut])
        return float(t.score(d[cols].iloc[cut:], y.iloc[cut:]))
    res["tree_depth4_test_accuracy_by_inputs"] = {
        "Error_Rate_% only": acc(["Error_Rate_%"]),
        "Production_Speed only": acc(["Production_Speed_units_per_hr"]),
        "both (Error_Rate_% + Production_Speed)": acc(RULE_INPUTS),
        "all other 7 measurements + mode (no Error_Rate_%, no Speed)": acc(
            [c for c in NUMERIC_RAW if c not in RULE_INPUTS] + ["mode_code"]),
    }

    rule = apply_label_rule(d)
    agree = rule == d[TARGET]
    res["rule"] = LABEL_RULE
    res["rule_agreement"] = float(agree.mean())
    res["rule_mismatches"] = int((~agree).sum())
    mm = d.loc[~agree, ["obs_seq", "Error_Rate_%", "Production_Speed_units_per_hr", TARGET]].copy()
    mm["rule_says"] = rule[~agree]
    res["mismatch_rows"] = mm.round(4).to_dict("records")
    res["majority_class_accuracy_test"] = float((d[TARGET].iloc[cut:] == d[TARGET].mode().iloc[0]).mean())
    res["conclusion"] = (
        "Efficiency_Status is (to within the listed mismatches) a deterministic function of Error_Rate_% and "
        "Production_Speed_units_per_hr. Any model that receives these two columns can reproduce the label, which "
        "demonstrates recovery of the labelling rule - not independent prediction of efficiency from sensors or the network."
    )
    return res


# --------------------------------------------------------------------------------------------
# main audit
# --------------------------------------------------------------------------------------------
def run_audit(df_raw: pd.DataFrame) -> Dict:
    df = add_datetime(df_raw)
    a: Dict = {}
    a["shape"] = {"rows": int(df.shape[0]), "columns": int(df_raw.shape[1])}
    a["columns"] = list(df_raw.columns)
    a["columns_match_expected"] = list(df_raw.columns) == EXPECTED_COLUMNS
    a["dtypes"] = {c: str(t) for c, t in df_raw.dtypes.items()}
    a["missing_values"] = {c: int(v) for c, v in df_raw.isna().sum().items()}
    a["duplicate_rows"] = int(df_raw.duplicated().sum())
    a["duplicate_rows_ignoring_date_time"] = int(df_raw.drop(columns=["Date", "Timestamp"]).duplicated().sum())
    a["unique"] = {"machines": int(df["Machine_ID"].nunique()),
                   "machine_id_range": [int(df["Machine_ID"].min()), int(df["Machine_ID"].max())],
                   "operation_modes": df["Operation_Mode"].value_counts().to_dict(),
                   "dates": int(df["Date"].nunique()), "timestamps_of_day": int(df["Timestamp"].nunique())}
    vc = df[TARGET].value_counts()
    a["target_distribution"] = {k: int(vc.get(k, 0)) for k in utils.CLASS_ORDER}
    a["target_distribution_pct"] = {k: float(100 * vc.get(k, 0) / len(df)) for k in utils.CLASS_ORDER}
    a["unexpected_target_values"] = [v for v in vc.index if v not in utils.CLASS_ORDER]
    a["unexpected_operation_modes"] = [v for v in df["Operation_Mode"].unique() if v not in OPERATION_MODES]
    a["numeric_statistics"] = df[NUMERIC_RAW].describe().round(4).to_dict()

    # ---- outliers / suspicious values
    out = {}
    for c in NUMERIC_RAW:
        x = df[c]
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = utils.PLAUSIBLE_LIMITS[c]
        ks = stats.kstest((x - x.min()) / (x.max() - x.min()), "uniform")
        out[c] = {"iqr_outliers": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
                  "outside_plausible_limits": int(((x < lo) | (x > hi)).sum()),
                  "min": float(x.min()), "max": float(x.max()),
                  "ks_distance_to_uniform": float(ks.statistic), "skewness": float(stats.skew(x))}
    a["outliers_and_shape"] = out
    a["all_features_look_uniform"] = bool(all(v["ks_distance_to_uniform"] < 0.02 for v in out.values()))

    # ---- date / timestamp consistency
    step = df["datetime"].diff().dt.total_seconds() / 60
    odd = step[(step != 1) & step.notna()]
    date_counts = df.groupby("Date", sort=False).size()
    unexpected_counts = date_counts[(date_counts != 1440)]
    a["datetime"] = {
        "format_used": "%d-%m-%Y %H:%M:%S",
        "unparseable": int(df["datetime"].isna().sum()),
        "first": str(df["datetime"].min()), "last": str(df["datetime"].max()),
        "file_order_is_time_order_except": {str(int(i)): float(v) for i, v in odd.items()},
        "one_minute_steps": int((step == 1).sum()),
        "duplicated_timestamps": int(df["datetime"].duplicated().sum()),
        "rows_per_date_not_1440": {k: int(v) for k, v in unexpected_counts.items()},
        "likely_explanation": ("The date label 01-03-2025 holds 2 days of records and the clock resets inside it. "
                               "A likely explanation is that a non-existent date (29-02-2025; 2025 is not a leap year) "
                               "was rolled over to 01-03-2025 by the data generator. This is a hypothesis, not a proven fact."),
        "handling": "Sort/split/rolling use file order (obs_seq); the recorded Date is kept for display and filtering."}

    # ---- machine coverage
    per_m = df.groupby("Machine_ID").size()
    gaps = df.sort_values("obs_seq").groupby("Machine_ID")["obs_seq"].diff().dropna()
    a["machine_coverage"] = {"rows_per_machine_min": int(per_m.min()), "rows_per_machine_max": int(per_m.max()),
                             "rows_per_machine_mean": float(per_m.mean()),
                             "median_gap_between_records_of_same_machine_minutes": float(gaps.median()),
                             "mean_gap_minutes": float(gaps.mean()),
                             "note": "Each minute holds ONE record from a randomly different machine, so a machine's own history is sparse and irregular."}

    # ---- correlations
    corr = df[NUMERIC_RAW].corr()
    off = corr.values[~np.eye(len(corr), dtype=bool)]
    a["feature_correlation_max_abs_offdiagonal"] = float(np.abs(off).max())
    ordinal = df[TARGET].map(CLASS_TO_INT)
    a["spearman_corr_with_efficiency_ordinal"] = {c: float(stats.spearmanr(df[c], ordinal)[0]) for c in NUMERIC_RAW}

    # ---- features vs label
    sub = df.sample(20000, random_state=utils.RANDOM_STATE)
    mi = mutual_info_classif(sub[NUMERIC_RAW], sub[TARGET].map(CLASS_TO_INT), random_state=utils.RANDOM_STATE)
    rows = {}
    for c, m in zip(NUMERIC_RAW, mi):
        groups = [g[c].values for _, g in df.groupby(TARGET)]
        h, p = stats.kruskal(*groups)
        grand = df[c].mean()
        eta2 = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / ((df[c] - grand) ** 2).sum()
        rows[c] = {"kruskal_H": float(h), "kruskal_p": float(p), "eta_squared": float(eta2), "mutual_information": float(m),
                   **{f"mean_{k}": float(df.loc[df[TARGET] == k, c].mean()) for k in utils.CLASS_ORDER}}
    a["feature_vs_label"] = rows
    chi_mode = stats.chi2_contingency(pd.crosstab(df["Operation_Mode"], df[TARGET]))
    chi_mach = stats.chi2_contingency(pd.crosstab(df["Machine_ID"], df[TARGET]))
    a["independence_tests"] = {"operation_mode_vs_efficiency": {"chi2": float(chi_mode[0]), "p": float(chi_mode[1])},
                               "machine_vs_efficiency": {"chi2": float(chi_mach[0]), "p": float(chi_mach[1])}}
    weeks = df["datetime"].dt.to_period("W").astype(str)
    chi_w = stats.chi2_contingency(pd.crosstab(weeks, df[TARGET]))
    a["independence_tests"]["week_vs_efficiency"] = {"chi2": float(chi_w[0]), "p": float(chi_w[1])}

    # ---- label derivation (leakage check)
    a["label_derivation"] = discover_label_rule(df)
    return a


def audit_to_markdown(a: Dict) -> str:
    ld = a["label_derivation"]
    lines = ["# Data audit summary", "",
             f"* Rows x columns: **{a['shape']['rows']:,} x {a['shape']['columns']}**; columns match the expected schema: **{a['columns_match_expected']}**",
             f"* Missing values: **{sum(a['missing_values'].values())}**; duplicate rows: **{a['duplicate_rows']}**",
             f"* Machines: **{a['unique']['machines']}**; operation modes: {a['unique']['operation_modes']}",
             f"* Target distribution: {a['target_distribution']} ({', '.join(f'{k} {v:.1f}%' for k, v in a['target_distribution_pct'].items())})",
             f"* All features distributed like a uniform distribution (KS distance < 0.02): **{a['all_features_look_uniform']}**",
             f"* Max |correlation| between features: **{a['feature_correlation_max_abs_offdiagonal']:.3f}**",
             f"* Time anomalies: {a['datetime']['file_order_is_time_order_except']}; dates with row count != 1440: {a['datetime']['rows_per_date_not_1440']}",
             "", "## Label derivation (leakage check)",
             f"* Rule: High = `{ld['rule']['High']}`; Medium = `{ld['rule']['Medium']}`; Low = else",
             f"* Agreement of rule with the label: **{100*ld['rule_agreement']:.3f}%** ({ld['rule_mismatches']} mismatches)",
             f"* Depth-4 decision-tree test accuracy by inputs: {ld['tree_depth4_test_accuracy_by_inputs']}",
             f"* Majority-class accuracy: {100*ld['majority_class_accuracy_test']:.2f}%", "", f"**Conclusion.** {ld['conclusion']}"]
    return "\n".join(lines)


def main() -> Dict:
    utils.ensure_dirs()
    df = load_raw()
    a = run_audit(df)
    save_json(a, utils.METRICS_DIR / "data_audit.json")
    (utils.REPORTS_DIR / "data_audit.md").write_text(audit_to_markdown(a), encoding="utf-8")
    log.info("Audit complete: rule agreement %.4f%%, mismatches %d", 100 * a["label_derivation"]["rule_agreement"],
             a["label_derivation"]["rule_mismatches"])
    return a


if __name__ == "__main__":
    main()
