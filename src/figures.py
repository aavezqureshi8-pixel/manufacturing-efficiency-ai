"""
figures.py - static Matplotlib / Seaborn figures for the research paper (PNG in outputs/figures).
The interactive Plotly versions are in app.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import utils
from .utils import CLASS_ORDER, NUMERIC_RAW, SHORT, TARGET

sns.set_theme(style="whitegrid")
COLORS = {"Low": "#c0444f", "Medium": "#e0a030", "High": "#3a9e6f"}


def _save(fig, name: str) -> Path:
    utils.FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = utils.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def _box(ax, d, col):
    sns.boxplot(data=d, x=TARGET, y=col, order=CLASS_ORDER, palette=COLORS, ax=ax, showfliers=False, hue=TARGET, legend=False)
    ax.set_title(SHORT[col], fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("")


def make_eda(d: pd.DataFrame, audit: Dict) -> None:
    # 01 class distribution
    fig, ax = plt.subplots(figsize=(5.5, 4))
    vc = d[TARGET].value_counts().reindex(CLASS_ORDER)
    bars = ax.bar(vc.index, vc.values, color=[COLORS[c] for c in vc.index])
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100*v/len(d):.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Efficiency status distribution (class imbalance)", fontweight="bold"); ax.set_ylabel("Records"); ax.set_ylim(0, vc.max() * 1.15)
    _save(fig, "01_class_distribution.png")

    # 02 sensors by class
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, c in zip(axes, utils.SENSOR_COLS):
        _box(ax, d, c)
    fig.suptitle("Sensor measurements by efficiency class", fontweight="bold"); _save(fig, "02_sensors_by_class.png")

    # 03 network by class + binned relation
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, c in zip(axes[:2], utils.NETWORK_COLS):
        _box(ax, d, c)
    q = pd.qcut(d["Network_Latency_ms"], 5)
    share = pd.crosstab(q, d[TARGET], normalize="index")[CLASS_ORDER] * 100
    for c in CLASS_ORDER:
        axes[2].plot(range(5), share[c].values, marker="o", color=COLORS[c], label=c)
    axes[2].set_xticks(range(5)); axes[2].set_xticklabels([f"Q{i+1}" for i in range(5)])
    axes[2].set_title("Class share by latency quintile (Q1 = lowest)", fontweight="bold"); axes[2].set_ylabel("% of records"); axes[2].legend()
    fig.suptitle("Network metrics versus efficiency", fontweight="bold"); _save(fig, "03_network_vs_efficiency.png")

    # 04 production & quality by class
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, c in zip(axes, utils.PRODUCTION_COLS):
        _box(ax, d, c)
    fig.suptitle("Production, quality and maintenance measurements by efficiency class", fontweight="bold"); _save(fig, "04_production_by_class.png")

    # 05 power vs speed
    s = d.sample(6000, random_state=1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in CLASS_ORDER:
        m = s[TARGET] == c
        ax.scatter(s.loc[m, "Production_Speed_units_per_hr"], s.loc[m, "Power_Consumption_kW"], s=6, alpha=.5, color=COLORS[c], label=c)
    ax.set_xlabel("Production speed (units/hr)"); ax.set_ylabel("Power (kW)")
    ax.set_title(f"Power vs speed (corr = {d['Power_Consumption_kW'].corr(d['Production_Speed_units_per_hr']):.3f})", fontweight="bold"); ax.legend(markerscale=3)
    _save(fig, "05_power_vs_speed.png")

    # 06 THE KEY CHART: speed vs error with thresholds
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c in CLASS_ORDER[::-1]:
        m = s[TARGET] == c
        ax.scatter(s.loc[m, "Production_Speed_units_per_hr"], s.loc[m, "Error_Rate_%"], s=7, alpha=.6, color=COLORS[c], label=c)
    for x in (200, 400):
        ax.axvline(x, color="k", ls="--", lw=.8)
    for y in (2, 5):
        ax.axhline(y, color="k", ls=":", lw=.8)
    ax.set_xlabel("Production speed (units/hr)"); ax.set_ylabel("Error rate (%)")
    ax.legend(markerscale=3, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Efficiency class is determined by two columns\n(dashed: speed 200 / 400, dotted: error 2 % / 5 %)", fontweight="bold")
    _save(fig, "06_speed_vs_error_rule.png")

    # 07 operation mode
    ct = pd.crosstab(d["Operation_Mode"], d[TARGET], normalize="index")[CLASS_ORDER] * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ct.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in CLASS_ORDER], rot=0)
    ax.set_ylabel("% of records"); ax.set_title("Efficiency by operation mode", fontweight="bold"); ax.legend(title="")
    _save(fig, "07_efficiency_by_mode.png")

    # 08 machine low-share with CI
    g = d.groupby("Machine_ID").agg(n=(TARGET, "size"), low=(TARGET, lambda x: (x == "Low").sum()))
    lo, hi = wilson(g["low"].values, g["n"].values)
    p = g["low"] / g["n"]
    order = p.sort_values().index
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.errorbar(range(len(order)), 100 * p[order], yerr=[100 * (p[order] - lo[np.searchsorted(g.index, order)]), 100 * (hi[np.searchsorted(g.index, order)] - p[order])],
                fmt="o", color="#1d5a8a", ecolor="#9db7cf", capsize=2)
    ax.axhline(100 * (d[TARGET] == "Low").mean(), color="r", ls="--", label="overall Low share")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7, rotation=90)
    ax.set_xlabel("Machine ID (sorted by Low share)"); ax.set_ylabel("% Low-efficiency records")
    ax.set_title("Low-efficiency share per machine with 95% confidence intervals (n ≈ 2,000 each)", fontweight="bold"); ax.legend()
    _save(fig, "08_machine_low_share.png")

    # 09 efficiency over time (daily)
    daily = pd.crosstab(d["datetime"].dt.date, d[TARGET], normalize="index")[CLASS_ORDER] * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in CLASS_ORDER:
        ax.plot(pd.to_datetime(daily.index), daily[c], color=COLORS[c], label=c, lw=1.2)
    ax.set_title("Daily share of each recorded class over time (log scale; no trend or degradation period visible)", fontweight="bold", fontsize=11)
    ax.set_yscale("log"); ax.set_ylabel("% of the day's records (log scale)")
    ax.set_yticks([1, 2, 3, 5, 10, 20, 50, 80]); ax.set_yticklabels(["1", "2", "3", "5", "10", "20", "50", "80"]); ax.minorticks_off(); ax.legend(loc="center right")
    _save(fig, "09_efficiency_over_time.png")

    # 10 correlation
    fig, ax = plt.subplots(figsize=(8, 6.5))
    dd = d[NUMERIC_RAW + ["y"]].rename(columns={**SHORT, "y": "Efficiency (0=Low,1=Med,2=High)"})
    sns.heatmap(dd.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation matrix (Pearson)", fontweight="bold"); _save(fig, "10_correlation.png")

    # 11 network quality band
    ct = pd.crosstab(d["network_quality"], d[TARGET], normalize="index").reindex(["Good", "Fair", "Poor"])[CLASS_ORDER] * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ct.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in CLASS_ORDER], rot=0)
    ax.set_ylabel("% of records"); ax.set_xlabel("Network-quality band (tertiles of latency + packet-loss index)")
    ax.set_title("Efficiency by network quality", fontweight="bold"); ax.legend(title="")
    _save(fig, "11_efficiency_by_network_quality.png")

    # 12 distributions
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, c in zip(axes.ravel(), NUMERIC_RAW):
        sns.histplot(d[c], bins=40, ax=ax, color="#1d5a8a"); ax.set_title(SHORT[c], fontsize=10); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Feature distributions: all roughly uniform between fixed limits", fontweight="bold"); _save(fig, "12_distributions.png")


def make_model_figs(results, time_folds, mach, pi_best, pi_ctx, glob, rel, rel_c, test_m, ctx_test_m, p_best, p_ctx, y_test) -> None:
    models = ["Logistic Regression", "Random Forest", "Gradient Boosting"]

    # 20 model comparison on deployed feature set
    r = results[(results["feature_set"] == "F2_raw_plus_derived") & results["model"].isin(models)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        s = r[r["split"] == split]
        sns.barplot(data=s, x="model", y="macro_f1", hue="class_weight", ax=ax, order=models)
        ax.set_title(f"Macro-F1 on {'validation' if split == 'val' else 'test'} data (full features, F2)", fontweight="bold")
        ax.set_xlabel(""); ax.set_ylim(0.6, 1.02); ax.tick_params(axis="x", labelsize=8)
    _save(fig, "20_model_comparison.png")

    # 21 ablation by feature set
    r = results[results["model"].isin(models) & (results["class_weight"] == "balanced") & (results["split"] == "test")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=r, x="feature_set", y="balanced_accuracy", hue="model", ax=ax,
                order=["F1_raw_all", "F2_raw_plus_derived", "F3_context_only", "F4_context_plus_history"])
    ax.axhline(1 / 3, color="k", ls="--", label="chance for 3 balanced classes (0.333)")
    ax.set_title("Balanced accuracy on test data by feature set\n(F3/F4 exclude Error_Rate_% and Speed)", fontweight="bold"); ax.set_xlabel(""); ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8); _save(fig, "21_feature_set_ablation.png")

    # 22 confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, m, title in zip(axes, (test_m, ctx_test_m), ("Deployed model (with Error_Rate_% + Speed)", "Context-only model (without them)")):
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax, xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10, fontweight="bold")
    fig.suptitle("Confusion matrices on the chronological test set", fontweight="bold"); _save(fig, "22_confusion_matrices.png")

    # 23 permutation importance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, pi, title in zip(axes, (pi_best, pi_ctx), ("Deployed model", "Context-only model")):
        p = pi.sort_values("f1_drop_mean")
        ax.barh([SHORT[f] for f in p["feature"]], p["f1_drop_mean"], xerr=p["f1_drop_std"], color="#1d5a8a")
        ax.set_xlabel("Drop in macro-F1 when the input is shuffled"); ax.set_title(title, fontweight="bold")
    fig.suptitle("Permutation importance (validation data)", fontweight="bold"); _save(fig, "23_permutation_importance.png")

    # 24 global Shapley
    g = glob.groupby(["explained_class", "feature"])["abs_attribution"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True)
    for ax, cls_ in zip(axes, ("Low", "High")):
        s = g[g["explained_class"] == cls_].sort_values("abs_attribution")
        ax.barh([SHORT[f] for f in s["feature"]], s["abs_attribution"], color=COLORS[cls_])
        ax.set_title(f"Mean |attribution| to P({cls_})", fontweight="bold"); ax.set_xlabel("probability points")
    fig.suptitle("Monte-Carlo Shapley attributions of the deployed model (180 validation records)", fontweight="bold"); _save(fig, "24_shapley_global.png")

    # 25 calibration
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, rt, title in zip(axes, (rel, rel_c), ("Deployed model", "Context-only model")):
        for c in CLASS_ORDER:
            s = rt[rt["class"] == c]
            ax.plot(s["mean_predicted"], s["observed_frequency"], marker="o", color=COLORS[c], label=c)
        ax.plot([0, 1], [0, 1], "k--", lw=.8)
        ax.set_xlabel("Mean predicted probability"); ax.set_title(title, fontweight="bold")
    axes[0].set_ylabel("Observed frequency"); axes[0].legend()
    fig.suptitle("Reliability (calibration) curves, one-vs-rest, test data", fontweight="bold"); _save(fig, "25_calibration.png")

    # 26 stability
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    t = time_folds[time_folds["feature_set"] == "F2_raw_plus_derived"]
    sns.lineplot(data=t, x="fold", y="macro_f1", hue="model", marker="o", ax=axes[0])
    axes[0].set_title("Expanding-window time folds (deployed feature set)", fontweight="bold"); axes[0].set_ylim(0.6, 1.02)
    mm = mach.copy()
    mm["set"] = mm["feature_set"].map({"F2_raw_plus_derived": "With Error/Speed (F2)", "F3_context_only": "Without (F3)"})
    sns.boxplot(data=mm, x="model", y="balanced_accuracy", hue="set", ax=axes[1])
    axes[1].set_title("Machines never seen in training (5 folds)", fontweight="bold"); axes[1].set_xlabel(""); axes[1].tick_params(axis="x", labelsize=8)
    _save(fig, "26_stability.png")

    # 27 class-weight effect on context-only features
    r = results[(results["feature_set"] == "F3_context_only") & (results["split"] == "test") & results["model"].isin(models)]
    long = r.melt(id_vars=["model", "class_weight"], value_vars=["recall_Low", "recall_Medium", "recall_High"], var_name="class", value_name="recall")
    long["class"] = long["class"].str.replace("recall_", "")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, mname in zip(axes, models):
        s = long[long["model"] == mname]
        sns.barplot(data=s, x="class", y="recall", hue="class_weight", ax=ax, order=CLASS_ORDER)
        ax.set_title(mname, fontsize=10, fontweight="bold"); ax.set_xlabel("")
    fig.suptitle("Effect of class weighting on per-class recall (context-only features, test data)", fontweight="bold"); _save(fig, "27_class_weight_effect.png")


def make_all(d, results, time_folds, mach, pi_best, pi_ctx, glob, rel, rel_c, test_m, ctx_test_m, p_best, p_ctx, y_test, audit) -> None:
    make_eda(d, audit)
    make_model_figs(results, time_folds, mach, pi_best, pi_ctx, glob, rel, rel_c, test_m, ctx_test_m, p_best, p_ctx, y_test)
