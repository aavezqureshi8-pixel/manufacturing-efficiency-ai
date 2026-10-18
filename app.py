"""
app.py - Manufacturing Efficiency Classification dashboard (Streamlit + Plotly).

Run:  streamlit run app.py

The app only LOADS the artifacts written by `python -m src.train_models`
(models/*.joblib, models/model_metadata.json, data/processed/clean_data.csv.gz, outputs/metrics/*.csv). It never retrains.

IMPORTANT HONESTY NOTES (also shown in the app):
* The dataset is a historical CSV. "Live monitoring" here REPLAYS recorded rows; no live sensor or 6G link exists.
* The label Efficiency_Status is derived from Error_Rate_% and Production_Speed. The deployed model recovers that rule.
* Feature importances / attributions are statistical associations learned by the model, not proven causes.
"""
from __future__ import annotations

import inspect
import time
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import utils
from src.dashboard_logic import (FEATURE_GROUP, Filters, apply_filters, class_share_by_period, daily_means,
                                 eta_squared_by_feature, machine_difference_test, machine_ranking, top_low_efficiency_factors)
from src.explainability import monte_carlo_shapley, plain_english
from src.inference import EfficiencyPredictor
from src.utils import CLASS_ORDER, NUMERIC_RAW, OPERATION_MODES, SHORT, TARGET

st.set_page_config(page_title="Manufacturing Efficiency AI", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

COLORS = {"Low": "#c0444f", "Medium": "#e0a030", "High": "#3a9e6f"}
CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg,#0b1f33 0%,#17466e 60%,#2a7ea6 100%); color:#fff; padding:1rem 1.4rem; border-radius:14px; margin-bottom:.9rem;}
.hero h1 {color:#fff; font-size:1.55rem; margin:0; padding:0;} .hero p {color:#cfe3f1; margin:.2rem 0 0 0; font-size:.92rem;}
.kpi {background:#fff; border:1px solid #e3e8ee; border-left:5px solid #17466e; border-radius:12px; padding:.7rem .9rem; box-shadow:0 1px 3px rgba(11,31,51,.08); min-height:84px;}
.kpi .label {font-size:.74rem; color:#5b6b7b; text-transform:uppercase; letter-spacing:.04em;} .kpi .value {font-size:1.45rem; font-weight:700; color:#0b1f33;}
.kpi .sub {font-size:.72rem; color:#7a8794;}
.badge {display:inline-block; padding:.35rem 1rem; border-radius:999px; color:#fff; font-weight:700; font-size:1.3rem;}
.note {background:#f2f7fb; border-left:4px solid #2a7ea6; padding:.65rem .9rem; border-radius:8px; font-size:.88rem;}
.warnbox {background:#fff7e8; border-left:4px solid #e0a030; padding:.65rem .9rem; border-radius:8px; font-size:.88rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ helpers
def _accepts_width(func) -> bool:
    try:
        return "width" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def show_plot(fig, key: str | None = None) -> None:
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=45, b=10), font=dict(size=12))
    if _accepts_width(st.plotly_chart):
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df: pd.DataFrame, height: int | None = None) -> None:
    kw = {"height": height} if height else {}
    if _accepts_width(st.dataframe):
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def csv_button(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key)


def note(text: str, kind: str = "note") -> None:
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ loading
@st.cache_data(show_spinner="Loading data ...")
def load_data() -> pd.DataFrame:
    if not utils.CLEAN_CSV.exists():
        raise FileNotFoundError(utils.CLEAN_CSV.name)
    df = pd.read_csv(utils.CLEAN_CSV, parse_dates=["datetime"])
    return df


@st.cache_data(show_spinner=False)
def load_tables() -> Dict:
    def rd(name):
        p = utils.METRICS_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    audit = utils.load_json(utils.METRICS_DIR / "data_audit.json") if (utils.METRICS_DIR / "data_audit.json").exists() else {}
    return {"perm_best": rd("permutation_importance_final.csv"), "perm_ctx": rd("permutation_importance_context.csv"),
            "shap_global": rd("shapley_global_final.csv"), "experiments": rd("experiment_results.csv"), "audit": audit,
            "class_means": rd("class_feature_means.csv")}


@st.cache_resource(show_spinner="Loading saved model ...")
def load_models():
    try:
        return {"best": EfficiencyPredictor.load("best"), "context": EfficiencyPredictor.load("context"), "meta": utils.load_json(utils.METADATA)}
    except Exception as exc:          # noqa: BLE001
        return {"error": str(exc)}


# ------------------------------------------------------------------------------------------ sidebar
def sidebar_filters(df: pd.DataFrame) -> Filters:
    sb = st.sidebar
    sb.markdown("### 🔎 Filters")
    sb.caption("Empty boxes mean 'everything'. Filters apply to pages 2-5.")
    nq = sb.multiselect("Network quality", ["Good", "Fair", "Poor"], help="Tertiles of a latency + packet-loss index (fitted on training data).")
    machines = sb.multiselect("Machine", sorted(df["Machine_ID"].unique().tolist()))
    dmin, dmax = df["datetime"].min().date(), df["datetime"].max().date()
    dr = sb.date_input("Time window", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    date_range = (pd.Timestamp(dr[0]), pd.Timestamp(dr[1])) if isinstance(dr, (tuple, list)) and len(dr) == 2 else None
    mode = sb.selectbox("Operation mode", ["All"] + OPERATION_MODES)
    classes = sb.multiselect("Recorded efficiency class", CLASS_ORDER)
    with sb.expander("Numeric ranges"):
        e_lo, e_hi = float(df["Error_Rate_%"].min()), float(df["Error_Rate_%"].max())
        s_lo, s_hi = float(df["Production_Speed_units_per_hr"].min()), float(df["Production_Speed_units_per_hr"].max())
        er = st.slider("Error rate (%)", e_lo, e_hi, (e_lo, e_hi))
        sr = st.slider("Production speed (units/hr)", s_lo, s_hi, (s_lo, s_hi))
    f = Filters(network_quality=nq, machines=machines, date_range=date_range, operation_mode=mode, classes=classes,
                error_range=er if er != (e_lo, e_hi) else None, speed_range=sr if sr != (s_lo, s_hi) else None)
    sb.markdown("---")
    return f


# ------------------------------------------------------------------------------------------ page 1
def _record_defaults(df: pd.DataFrame, meta: Dict, seq: int | None) -> Dict:
    if seq is None:
        d = {c: float(meta["feature_ranges"][c]["median"]) for c in NUMERIC_RAW}
        d["Operation_Mode"] = "Active"
        return d
    r = df.iloc[int(seq)]
    d = {c: float(r[c]) for c in NUMERIC_RAW}
    d["Operation_Mode"] = str(r["Operation_Mode"])
    return d


def _random_record(n: int) -> None:
    st.session_state["rec_seq"] = int(np.random.default_rng().integers(n))


def page_prediction(df: pd.DataFrame, f: Filters, ctx: Dict) -> None:
    models = ctx["models"]
    best, context, meta = models["best"], models["context"], models["meta"]
    note("<b>How to read this page.</b> The model classifies the <i>current observed</i> efficiency state of one record. "
         "It receives <b>Error rate</b> and <b>Production speed</b>, the two columns from which the recorded label is derived, so it reproduces "
         "that labelling rule. It does <b>not</b> show that sensors or the network predict efficiency, and it does not forecast breakdowns.", "warnbox")
    st.write("")
    n = len(df)
    mode_sel = st.radio("Starting point", ["Recorded observation", "Manual entry (training medians)"], horizontal=True)
    seq = None
    recorded = None
    if mode_sel == "Recorded observation":
        c1, c2 = st.columns([3, 1])
        st.session_state.setdefault("rec_seq", 0)
        seq = c1.number_input("Observation number (position in the file, 0 to %d)" % (n - 1), 0, n - 1, key="rec_seq", step=1)
        c2.write("")
        c2.button("🎲 Random record", on_click=_random_record, args=(n,))
        recorded = str(df.iloc[int(seq)][TARGET])
    base = _record_defaults(df, meta, None if seq is None else int(seq))
    tag = "m" if seq is None else str(int(seq))

    st.subheader("1 · Measurements")
    obs: Dict = {}
    cols = st.columns(3)
    step = {"Temperature_C": 1.0, "Vibration_Hz": 0.1, "Power_Consumption_kW": 0.1, "Network_Latency_ms": 1.0, "Packet_Loss_%": 0.1,
            "Quality_Control_Defect_Rate_%": 0.1, "Production_Speed_units_per_hr": 5.0, "Predictive_Maintenance_Score": 0.01, "Error_Rate_%": 0.1}
    for i, c in enumerate(NUMERIC_RAW):
        obs[c] = cols[i % 3].number_input(SHORT[c], value=float(round(base[c], 3)), step=step[c], format="%.3f", key=f"in_{c}_{tag}")
    obs["Operation_Mode"] = cols[0].selectbox("Operation mode", OPERATION_MODES, index=OPERATION_MODES.index(base["Operation_Mode"]) if base["Operation_Mode"] in OPERATION_MODES else 0, key=f"in_mode_{tag}")

    st.subheader("2 · Metric sensitivity sliders (scenario simulation)")
    st.caption("Change a measurement by a percentage to see how the prediction responds. This is a **what-if simulation on one record**; "
               "it does not change any real equipment.")
    scen = dict(obs)
    with st.expander("Open sensitivity sliders", expanded=False):
        sc = st.columns(3)
        for i, c in enumerate(NUMERIC_RAW):
            pctv = sc[i % 3].slider(f"{SHORT[c]}: change (%)", -50, 50, 0, key=f"sl_{c}_{tag}")
            scen[c] = obs[c] * (1 + pctv / 100.0)
    changed = any(abs(scen[c] - obs[c]) > 1e-12 for c in NUMERIC_RAW)

    show_obs = scen if changed else obs
    label = "scenario" if changed else "observation"
    res = best.predict_frame(pd.DataFrame([show_obs])).iloc[0]
    base_res = best.predict_frame(pd.DataFrame([obs])).iloc[0]
    probs = np.array([res[f"p_{c}"] for c in CLASS_ORDER])

    st.subheader(f"3 · Prediction for the {label}")
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    c1.markdown(f'<div class="badge" style="background:{COLORS[res["pred_class"]]}">{res["pred_class"]} efficiency</div>', unsafe_allow_html=True)
    c1.caption(f"Highest class probability: {res['max_prob']:.3f} (a model score, not a guaranteed chance of being right)")
    kpi(c2, "P(Low)", f"{probs[0]:.3f}"); kpi(c3, "P(Medium)", f"{probs[1]:.3f}"); kpi(c4, "P(High)", f"{probs[2]:.3f}")
    if recorded is not None:
        st.write(f"Recorded label for observation {int(seq)}: **{recorded}**" +
                 ("" if changed else f" - model {'agrees' if recorded == res['pred_class'] else 'DISAGREES'}."))
    if changed:
        st.write(f"Original observation prediction: **{base_res['pred_class']}** (max probability {base_res['max_prob']:.3f}); "
                 f"scenario prediction: **{res['pred_class']}**.")

    pdf = pd.DataFrame({"class": CLASS_ORDER, "probability": probs})
    show_plot(px.bar(pdf, x="probability", y="class", orientation="h", color="class", color_discrete_map=COLORS, range_x=[0, 1],
                     text="probability", title="Class probabilities").update_traces(texttemplate="%{text:.3f}").update_layout(showlegend=False), "pred_bar")

    for w in best.input_warnings(show_obs):
        st.warning(w)
    st.info(best.confidence_note(probs))
    with st.expander("About prediction confidence and calibration"):
        tm = meta["final_model"]["test_metrics"]
        st.write(f"On the chronological test set the deployed model had a top-label expected calibration error of **{tm['ece_top_label']:.4f}** and a multiclass "
                 f"Brier score of **{tm['brier']:.4f}**. The probabilities are close to 0 or 1 because the label is a sharp rule of two inputs. "
                 "Near the rule boundaries (error rate ≈ 2 % or 5 %, speed ≈ 200 or 400) the model can be wrong even when it looks confident.")

    st.subheader("4 · Which inputs drive this prediction?")
    background = ctx["background"]
    cls_idx = CLASS_ORDER.index(res["pred_class"])
    attr = monte_carlo_shapley(best, show_obs, background, cls_idx, n_perm=40, seed=7)
    attr["label"] = [SHORT[x] for x in attr["feature"]]
    attr["direction"] = np.where(attr["attribution"] >= 0, "raises", "lowers")
    show_plot(px.bar(attr.sort_values("attribution"), x="attribution", y="label", orientation="h", color="direction",
                     color_discrete_map={"raises": "#3a9e6f", "lowers": "#c0444f"},
                     title=f"Contribution of each input to P({res['pred_class']}) (Monte-Carlo Shapley estimate)"), "pred_attr")
    st.write(plain_english(attr, res["pred_class"]))

    st.subheader("5 · Relevant measurements compared with the training range")
    rng = meta["feature_ranges"]
    tbl = pd.DataFrame({"Measurement": [SHORT[c] for c in NUMERIC_RAW], "Value": [round(show_obs[c], 3) for c in NUMERIC_RAW],
                        "Training min": [round(rng[c]["min"], 3) for c in NUMERIC_RAW], "Training median": [round(rng[c]["median"], 3) for c in NUMERIC_RAW],
                        "Training max": [round(rng[c]["max"], 3) for c in NUMERIC_RAW]})
    show_table(tbl)

    with st.expander("Compare: model WITHOUT Error rate and Speed (sensors + network + quality only)"):
        cres = context.predict_frame(pd.DataFrame([show_obs])).iloc[0]
        st.write(f"Context-only prediction: **{cres['pred_class']}** with probabilities "
                 f"Low {cres['p_Low']:.3f} / Medium {cres['p_Medium']:.3f} / High {cres['p_High']:.3f}. "
                 f"On the test set this model's balanced accuracy was **{meta['context_model']['test_metrics']['balanced_accuracy']:.3f}** "
                 "(0.333 = chance), i.e. it has essentially no predictive signal.")

    out = pd.DataFrame([{**show_obs, "predicted_class": res["pred_class"], **{f"p_{c}": probs[i] for i, c in enumerate(CLASS_ORDER)}}])
    csv_button(out, "⬇ Download this prediction (CSV)", "prediction.csv", "dl_pred")

    with st.expander("📂 Batch prediction from a CSV file"):
        st.caption("Columns required: " + ", ".join(utils.RAW_INPUTS))
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                raw = pd.read_csv(up)
                missing = [c for c in utils.RAW_INPUTS if c not in raw.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    pf = best.predict_frame(raw)
                    bad_mode = int((~raw["Operation_Mode"].astype(str).str.strip().str.title().isin(OPERATION_MODES)).sum())
                    n_missing = int(raw[NUMERIC_RAW].apply(pd.to_numeric, errors="coerce").isna().any(axis=1).sum())
                    if bad_mode or n_missing:
                        st.warning(f"{n_missing} rows had missing/non-numeric values (training medians used) and {bad_mode} rows had an unknown operation mode.")
                    outb = pd.concat([raw.reset_index(drop=True), pf.reset_index(drop=True)], axis=1)
                    show_table(outb.head(200), height=300)
                    csv_button(outb, "⬇ Download predictions", "batch_predictions.csv", "dl_batch")
            except Exception as exc:       # noqa: BLE001
                st.error(f"Could not read the file: {exc}")


# ------------------------------------------------------------------------------------------ page 2
def page_machine(df: pd.DataFrame, f: Filters, ctx: Dict) -> None:
    note("Charts show <b>recorded</b> measurements and the <b>recorded</b> efficiency labels. Model predictions are shown separately and only when you ask for them.")
    machines = sorted(df["Machine_ID"].unique().tolist())
    default = f.machines[0] if f.machines else machines[0]
    mid = st.selectbox("Machine", machines, index=machines.index(default))
    f2 = Filters(**{**f.__dict__, "machines": [mid]})
    m = apply_filters(df, f2)
    if m.empty:
        st.warning("No records for this machine with the current filters.")
        return
    freq = st.radio("Aggregation", ["Daily", "Weekly"], horizontal=True)
    fq = "D" if freq == "Daily" else "W"
    c = st.columns(5)
    kpi(c[0], "Records", f"{len(m):,}", f"machine {mid}")
    for i, k in enumerate(CLASS_ORDER):
        kpi(c[i + 1], f"Recorded {k}", f"{100 * (m[TARGET] == k).mean():.1f}%", f"{int((m[TARGET] == k).sum())} records")
    kpi(c[4], "Avg error rate", f"{m['Error_Rate_%'].mean():.2f}%", "recorded")
    st.write("")
    share = class_share_by_period(m, fq)
    long = share.melt(id_vars=["period", "n"], value_vars=CLASS_ORDER, var_name="class", value_name="share")
    show_plot(px.bar(long, x="period", y="share", color="class", color_discrete_map=COLORS, hover_data=["n"], barmode="stack",
                     title=f"Recorded efficiency class share ({freq.lower()}); hover shows the number of records"), "mach_share")
    metrics = ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%",
               "Production_Speed_units_per_hr", "Quality_Control_Defect_Rate_%", "Error_Rate_%"]
    dm = daily_means(m, metrics, fq)
    cols = st.columns(2)
    for i, met in enumerate(metrics):
        with cols[i % 2]:
            show_plot(px.line(dm, x="period", y=met, markers=True, title=f"{SHORT[met]}: {freq.lower()} mean (recorded)"), f"mach_{met}")
    with st.expander("Recorded labels versus model predictions (kept separate on purpose)", expanded=False):
        which = st.radio("Prediction source rows", ["Test period only (unseen data)", "All periods (includes training rows)"], horizontal=True)
        mp = m[m["split"] == "test"] if which.startswith("Test") else m
        if mp.empty:
            st.info("No rows in that period for this machine/filters.")
        else:
            cm = pd.crosstab(mp[TARGET], mp["pred_class"]).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
            show_plot(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Recorded"),
                                title=f"Recorded vs predicted ({len(mp):,} records)"), "mach_cm")
            st.caption("Predictions on training rows are in-sample and look better than they would on new data. The model reproduces a labelling rule, "
                       "so high agreement is expected and says nothing about forecasting.")


# ------------------------------------------------------------------------------------------ page 3
def page_explain(df: pd.DataFrame, f: Filters, ctx: Dict) -> None:
    tables, models = ctx["tables"], ctx["models"]
    best = models["best"]
    note("Importances and attributions describe what the <b>model</b> uses (statistical association). They do <b>not</b> prove that a measurement "
         "<i>causes</i> low efficiency.", "warnbox")
    st.write("")
    st.subheader("Global feature importance")
    c1, c2 = st.columns(2)
    pb, pc = tables["perm_best"], tables["perm_ctx"]
    if not pb.empty:
        pb = pb.assign(label=pb["feature"].map(SHORT)).sort_values("f1_drop_mean")
        with c1:
            show_plot(px.bar(pb, x="f1_drop_mean", y="label", orientation="h", error_x="f1_drop_std",
                             title="Permutation importance, deployed model (drop in macro-F1)"), "perm_best")
        pc = pc.assign(label=pc["feature"].map(SHORT)).sort_values("f1_drop_mean")
        with c2:
            show_plot(px.bar(pc, x="f1_drop_mean", y="label", orientation="h", error_x="f1_drop_std",
                             title="Permutation importance, context-only model (no Error/Speed)"), "perm_ctx")
        st.caption("Only Error rate and Production speed matter for the deployed model. The context-only model has nothing to rely on, so every importance is about zero.")
    sg = tables["shap_global"]
    if not sg.empty:
        sg = sg.assign(label=sg["feature"].map(SHORT))
        show_plot(px.bar(sg.sort_values("abs_attribution"), x="abs_attribution", y="label", color="explained_class", barmode="group",
                         orientation="h", color_discrete_map=COLORS, title="Mean |Shapley attribution| of the deployed model (180 validation records)"), "shap_glob")

    st.subheader("Important factors associated with Low efficiency (current filters)")
    fdf = apply_filters(df, f)
    for line in top_low_efficiency_factors(fdf):
        st.markdown(f"- {line}")
    st.caption("Large differences appear only for Error rate and Production speed. Sensor and network measurements look the same in every class.")

    st.subheader("Feature comparison across High / Medium / Low")
    feat = st.selectbox("Measurement", NUMERIC_RAW, format_func=lambda c: SHORT[c])
    sample = fdf.sample(min(len(fdf), 20000), random_state=1) if len(fdf) else fdf
    if len(sample):
        show_plot(px.box(sample, x=TARGET, y=feat, color=TARGET, color_discrete_map=COLORS, category_orders={TARGET: CLASS_ORDER},
                         title=f"{SHORT[feat]} by recorded class"), "cmp_box")
        eta = eta_squared_by_feature(fdf)
        eta["Measurement"] = eta["feature"].map(SHORT)
        eta["Group"] = eta["feature"].map(FEATURE_GROUP)
        show_plot(px.bar(eta.sort_values("eta_squared"), x="eta_squared", y="Measurement", color="Group", orientation="h",
                         title="Share of each measurement's variance explained by efficiency class (0 = unrelated)"), "cmp_eta")
        sc = sample.sample(min(len(sample), 5000), random_state=2)
        fig = px.scatter(sc, x="Production_Speed_units_per_hr", y="Error_Rate_%", color=TARGET, color_discrete_map=COLORS, opacity=0.6,
                         category_orders={TARGET: CLASS_ORDER}, title="The recorded class follows thresholds of speed (200 / 400) and error rate (2 % / 5 %)")
        for xv in (200, 400):
            fig.add_vline(x=xv, line_dash="dash", line_color="black")
        for yv in (2, 5):
            fig.add_hline(y=yv, line_dash="dot", line_color="black")
        show_plot(fig, "cmp_rule")

    st.subheader("Explain an individual record")
    st.session_state.setdefault("exp_seq", 0)
    seq = st.number_input("Observation number", 0, len(df) - 1, key="exp_seq", step=1)
    row = df.iloc[int(seq)]
    obs = {c: float(row[c]) for c in NUMERIC_RAW}
    obs["Operation_Mode"] = str(row["Operation_Mode"])
    pr = best.predict_frame(pd.DataFrame([obs])).iloc[0]
    st.write(f"Recorded label: **{row[TARGET]}** · model prediction: **{pr['pred_class']}** "
             f"(P(Low) {pr['p_Low']:.3f}, P(Medium) {pr['p_Medium']:.3f}, P(High) {pr['p_High']:.3f})")
    cls_name = st.radio("Explain probability of", CLASS_ORDER, index=CLASS_ORDER.index(pr["pred_class"]), horizontal=True)
    attr = monte_carlo_shapley(best, obs, ctx["background"], CLASS_ORDER.index(cls_name), n_perm=40, seed=11)
    attr["label"] = [SHORT[x] for x in attr["feature"]]
    attr["direction"] = np.where(attr["attribution"] >= 0, "raises", "lowers")
    show_plot(px.bar(attr.sort_values("attribution"), x="attribution", y="label", orientation="h", color="direction",
                     color_discrete_map={"raises": "#3a9e6f", "lowers": "#c0444f"}, title=f"Contribution to P({cls_name})"), "loc_attr")
    st.write(plain_english(attr, cls_name))

    st.subheader("How the model responds when one measurement changes")
    st.caption("Scenario simulation on the record chosen above: one measurement is varied across its training range, all others stay fixed.")
    var = st.selectbox("Measurement to vary", NUMERIC_RAW, index=NUMERIC_RAW.index("Error_Rate_%"), format_func=lambda c: SHORT[c])
    rng = models["meta"]["feature_ranges"][var]
    grid = np.linspace(rng["min"], rng["max"], 60)
    sweep = pd.DataFrame([{**obs, var: g} for g in grid])
    ps = best.predict_proba(sweep)
    long = pd.DataFrame({"value": np.tile(grid, 3), "class": np.repeat(CLASS_ORDER, len(grid)), "probability": ps.T.ravel()})
    show_plot(px.line(long, x="value", y="probability", color="class", color_discrete_map=COLORS, title=f"Predicted probabilities as {SHORT[var]} varies"), "sweep")

    with st.expander("Decision rules discovered in the data audit"):
        a = tables["audit"].get("label_derivation", {})
        if a:
            st.write(f"Rule agreement with the recorded label: **{100 * a['rule_agreement']:.3f}%** ({a['rule_mismatches']} mismatches).")
            st.json(a["rule"])
            st.code(a["depth4_tree_rules"])
            st.write(a["conclusion"])


# ------------------------------------------------------------------------------------------ page 4
def page_monitor(df: pd.DataFrame, f: Filters, ctx: Dict) -> None:
    fdf = apply_filters(df, f)
    if fdf.empty:
        st.warning("No records match the current filters.")
        return
    c = st.columns(5)
    kpi(c[0], "Records", f"{len(fdf):,}", f"{fdf['Machine_ID'].nunique()} machines")
    for i, k in enumerate(CLASS_ORDER):
        kpi(c[i + 1], f"Recorded {k}", f"{100 * (fdf[TARGET] == k).mean():.1f}%", f"{int((fdf[TARGET] == k).sum()):,} records")
    kpi(c[4], "Avg speed", f"{fdf['Production_Speed_units_per_hr'].mean():.0f}", "units/hr")
    st.write("")
    a, b = st.columns(2)
    with a:
        ct = pd.crosstab(fdf["Operation_Mode"], fdf[TARGET], normalize="index").reindex(columns=CLASS_ORDER, fill_value=0).reset_index()
        n_mode = fdf["Operation_Mode"].value_counts()
        ct["records"] = ct["Operation_Mode"].map(n_mode)
        long = ct.melt(id_vars=["Operation_Mode", "records"], value_vars=CLASS_ORDER, var_name="class", value_name="share")
        show_plot(px.bar(long, x="Operation_Mode", y="share", color="class", color_discrete_map=COLORS, hover_data=["records"],
                         title="Efficiency by operation mode (share of records)"), "op_mode")
    with b:
        nq = pd.crosstab(fdf["network_quality"], fdf[TARGET], normalize="index").reindex(index=["Good", "Fair", "Poor"], columns=CLASS_ORDER, fill_value=0).reset_index()
        long = nq.melt(id_vars="network_quality", value_vars=CLASS_ORDER, var_name="class", value_name="share")
        show_plot(px.bar(long, x="network_quality", y="share", color="class", color_discrete_map=COLORS, title="Efficiency by network-quality band"), "op_net")

    st.subheader("Network versus sensor impact")
    eta = eta_squared_by_feature(fdf)
    eta["Measurement"] = eta["feature"].map(SHORT)
    eta["Group"] = eta["feature"].map(FEATURE_GROUP)
    show_plot(px.bar(eta.sort_values("eta_squared", ascending=False), x="Measurement", y="eta_squared", color="Group",
                     title="Association with recorded efficiency class (eta-squared, 0 = none)"), "op_eta")
    grp = eta.groupby("Group")["eta_squared"].max().reset_index().sort_values("eta_squared", ascending=False)
    show_table(grp.rename(columns={"eta_squared": "Strongest association in group"}).round(4))
    st.caption("Association is not causation. In this dataset network and sensor measurements show no relationship with the recorded class; only production speed and error rate do.")

    st.subheader("Quality and production metrics by efficiency class")
    qp = fdf.groupby(TARGET)[["Production_Speed_units_per_hr", "Quality_Control_Defect_Rate_%", "Error_Rate_%", "Power_Consumption_kW", "Predictive_Maintenance_Score"]].mean().reindex(CLASS_ORDER).round(3).reset_index()
    show_table(qp)

    st.subheader("Machine-level efficiency trends")
    wk = fdf.assign(week=fdf["datetime"].dt.to_period("W").dt.start_time.dt.strftime("%m-%d"))
    piv = wk.assign(low=(wk[TARGET] == "Low").astype(int)).pivot_table(index="Machine_ID", columns="week", values="low", aggfunc="mean")
    if piv.shape[0] >= 1 and piv.shape[1] >= 1:
        show_plot(px.imshow(piv, aspect="auto", color_continuous_scale="RdYlGn_r", labels=dict(x="Week starting (MM-DD)", y="Machine", color="Low share"),
                            title="Share of Low-efficiency records by machine and week"), "op_heat")

    st.subheader("Machines with a high observed share of Low-efficiency records")
    rk = machine_ranking(fdf)
    min_obs = 30
    st.caption("Ranking by the observed share of Low records. Intervals are 95% Wilson confidence intervals; **observations per machine are shown**. "
               "A high share is NOT a fault diagnosis.")
    if len(rk):
        fig = px.scatter(rk, x="Machine_ID", y="low_share", error_y=rk["ci_high"] - rk["low_share"], error_y_minus=rk["low_share"] - rk["ci_low"],
                         color="vs_fleet", hover_data=["records", "low_records"], title="Low-efficiency share per machine (95% CI)")
        fig.add_hline(y=float(rk["fleet_low_share"].iloc[0]), line_dash="dash", line_color="black")
        show_plot(fig, "op_rank")
        show_table(rk.assign(low_share=(100 * rk["low_share"]).round(2), ci_low=(100 * rk["ci_low"]).round(2), ci_high=(100 * rk["ci_high"]).round(2))
                   [["Machine_ID", "records", "low_records", "low_share", "ci_low", "ci_high", "vs_fleet", "enough_data"]]
                   .rename(columns={"low_share": "Low %", "ci_low": "CI low %", "ci_high": "CI high %", "records": "Observations"}), height=320)
        if (rk["records"] < min_obs).any():
            st.warning(f"{int((rk['records'] < min_obs).sum())} machines have fewer than {min_obs} observations under these filters; rankings for them are unreliable.")
        t = machine_difference_test(fdf)
        if t:
            verdict = ("machines differ more than chance would explain" if t["p_value"] < 0.05 else "no evidence that machines differ beyond chance")
            st.write(f"Chi-square test of machine vs class mix: χ²={t['chi2']:.1f}, p={t['p_value']:.3f} → {verdict}.")
        csv_button(rk, "⬇ Download machine ranking (CSV)", "machine_ranking.csv", "dl_rank")
    csv_button(fdf.drop(columns=["ctx_p_Low", "ctx_p_Medium", "ctx_p_High"], errors="ignore"), "⬇ Download filtered records (CSV)", "filtered_records.csv", "dl_filtered")


# ------------------------------------------------------------------------------------------ page 5
def _advance(delta: int, n: int) -> None:
    st.session_state["live_pos"] = int(min(max(st.session_state.get("live_pos", 0) + delta, 0), n - 1))


def _render_live(rows: pd.DataFrame, pos: int, key: str) -> None:
    r = rows.iloc[pos]
    probs = np.array([r["p_Low"], r["p_Medium"], r["p_High"]])
    c = st.columns(4)
    kpi(c[0], "Observation", f"#{int(r['obs_seq'])}", f"machine {int(r['Machine_ID'])} · {r['Operation_Mode']}")
    kpi(c[1], "Recorded label", str(r[TARGET]), str(r["datetime"]))
    kpi(c[2], "Model prediction", str(r["pred_class"]), f"max prob {r['max_prob']:.3f} · {'unseen test period' if r['split'] == 'test' else r['split'] + ' period (in-sample)'}")
    kpi(c[3], "Network quality", str(r["network_quality"]), f"latency {r['Network_Latency_ms']:.1f} ms · loss {r['Packet_Loss_%']:.2f}%")
    show_plot(px.bar(pd.DataFrame({"class": CLASS_ORDER, "probability": probs}), x="class", y="probability", color="class", color_discrete_map=COLORS,
                     range_y=[0, 1], title="Class probabilities").update_layout(showlegend=False), f"live_p_{key}")
    meas = pd.DataFrame({"Measurement": [SHORT[m] for m in NUMERIC_RAW], "Value": [round(float(r[m]), 3) for m in NUMERIC_RAW]})
    show_table(meas)
    hist = rows.iloc[max(0, pos - 14): pos + 1]
    h = hist.assign(index=range(len(hist)))
    cols = st.columns(2)
    with cols[0]:
        show_plot(px.line(h, x="datetime", y="Production_Speed_units_per_hr", markers=True, color_discrete_sequence=["#17466e"],
                          title="Recent production speed"), f"live_s_{key}")
    with cols[1]:
        show_plot(px.line(h, x="datetime", y="Error_Rate_%", markers=True, color_discrete_sequence=["#c0444f"], title="Recent error rate"), f"live_e_{key}")
    show_table(hist[["datetime", "Machine_ID", "Operation_Mode", "Error_Rate_%", "Production_Speed_units_per_hr", TARGET, "pred_class", "max_prob"]]
               .rename(columns={TARGET: "recorded", "pred_class": "predicted"}).iloc[::-1], height=260)


def page_live(df: pd.DataFrame, f: Filters, ctx: Dict) -> None:
    note("<b>SIMULATED live monitoring.</b> This replays <b>historical records</b> from the CSV one by one. There is no live sensor feed, "
         "no 6G connection and no factory integration behind this page.", "warnbox")
    st.write("")
    base = apply_filters(df, f)
    machines = ["All machines"] + sorted(base["Machine_ID"].unique().tolist())
    sel = st.selectbox("Replay records of", machines)
    rows = base if sel == "All machines" else base[base["Machine_ID"] == sel]
    rows = rows.sort_values("obs_seq").reset_index(drop=True)
    if rows.empty:
        st.warning("No records match the current filters.")
        return
    n = len(rows)
    st.session_state["live_pos"] = int(min(st.session_state.get("live_pos", 0), n - 1))
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    c1.button("◀ Previous", on_click=_advance, args=(-1, n))
    c2.button("Next ▶", on_click=_advance, args=(1, n))
    autoplay = c3.button("▶ Auto-play 10")
    c4.slider("Position in the replay", 0, max(n - 1, 1), key="live_pos")
    pos = int(st.session_state["live_pos"])
    holder = st.empty()
    if autoplay:
        for i in range(10):
            _advance(1, n)
            with holder.container():
                _render_live(rows, int(st.session_state["live_pos"]), f"auto{i}")
            time.sleep(0.5)
    else:
        with holder.container():
            _render_live(rows, pos, "main")
    with st.expander("What would be needed to connect a real data source?"):
        st.markdown("""
- **A data feed**: sensors / PLCs publishing readings (for example through an industrial protocol gateway such as OPC UA or MQTT) into a message broker or time-series store.
- **The same fields and units** as the training data (9 measurements + operation mode) with synchronised clocks, and a timestamp convention that avoids the date issue found in this file.
- **A scoring service** that applies `src/inference.py` (same feature engineering + saved preprocessor + model) to each new record.
- **Monitoring**: input range checks (already in this app), prediction logs, drift checks, and periodic re-validation against real outcomes.
- **Security and privacy**: authenticated, encrypted connections, network segmentation between factory and analytics systems, access control and an audit trail.
- **Real labels**: the model currently reproduces a labelling rule. A real deployment needs a trusted, independently defined efficiency measure, otherwise the model will just reproduce whatever rule produced the labels.
""")


# ------------------------------------------------------------------------------------------ page 6
def page_about(df: pd.DataFrame, f: Filters, ctx: Dict) -> None:
    meta, a = ctx["models"]["meta"], ctx["tables"]["audit"]
    ld = a.get("label_derivation", {})
    st.subheader("What the data audit found")
    c = st.columns(4)
    kpi(c[0], "Records", f"{a['shape']['rows']:,}", f"{a['shape']['columns']} columns")
    kpi(c[1], "Machines", f"{a['unique']['machines']}", "1 record per minute, random machine")
    kpi(c[2], "Missing / duplicates", f"{sum(a['missing_values'].values())} / {a['duplicate_rows']}", "none found")
    kpi(c[3], "High class", f"{a['target_distribution_pct']['High']:.1f}%", f"Low {a['target_distribution_pct']['Low']:.1f}% · Medium {a['target_distribution_pct']['Medium']:.1f}%")
    st.write("")
    st.markdown(f"""
* **The label is a rule.** High = `{ld['rule']['High']}`; Medium = `{ld['rule']['Medium']}`. It matches **{100 * ld['rule_agreement']:.3f}%** of rows ({ld['rule_mismatches']} mismatches).
* **Sensors and network carry no signal for this label** (temperature, vibration, power, latency, packet loss, defect rate, maintenance score and operation mode are statistically unrelated to it).
* **All measurements are uniformly distributed** between fixed limits and almost uncorrelated, which suggests simulated data.
* **Timeline issue:** {a['datetime']['likely_explanation']}
* **Class imbalance:** High is only {a['target_distribution_pct']['High']:.1f}% of records; class weighting and per-class metrics are used.
""")
    st.subheader("How the models were evaluated")
    sp = meta["split"]
    st.markdown(f"""
* **Chronological split** (no shuffling): train {sp['train_rows']:,} rows · validation {sp['val_rows']:,} · test {sp['test_rows']:,}. Measures performance on later records of the same factory.
* **Expanding-window time folds** and **unseen-machine folds** check stability. Unseen-machine folds measure transfer to other machines.
* Preprocessing and the network-quality reference are fitted on **training data only**. History features use **past records only**.
""")
    fm, cm = meta["final_model"], meta["context_model"]
    st.subheader("Deployed model and the premise test")
    st.markdown(f"""
| | Deployed model | Context-only model (no Error rate, no Speed) |
|---|---|---|
| Algorithm | {fm['model']} (class weight: {fm['class_weight']}) | {cm['model']} (class weight: {cm['class_weight']}) |
| Test macro-F1 | {fm['test_metrics']['macro_f1']:.4f} | {cm['test_metrics']['macro_f1']:.4f} |
| Test balanced accuracy | {fm['test_metrics']['balanced_accuracy']:.4f} | {cm['test_metrics']['balanced_accuracy']:.4f} |
| Test High-class F1 | {fm['test_metrics']['f1_High']:.4f} | {cm['test_metrics']['f1_High']:.4f} |
""")
    st.warning(fm["warning"])
    ex = ctx["tables"]["experiments"]
    if not ex.empty:
        with st.expander("Full experiment table"):
            show_table(ex.round(4), height=400)
            csv_button(ex, "⬇ Download experiment results", "experiment_results.csv", "dl_exp")
    st.subheader("Limitations")
    st.markdown("""
* Results come from one (apparently simulated) dataset; they do not show performance on a real factory.
* Classification of the **current** state does not prove the system can predict future breakdowns.
* Attributions and importances are associations, not causes.
* No claim of approval, endorsement or deployment by Unified Mentor or Thales Group is made.
""")


# ------------------------------------------------------------------------------------------ main
PAGES = {
    "1 · Efficiency Prediction": page_prediction,
    "2 · Machine-Level Insights": page_machine,
    "3 · Explainability Panel": page_explain,
    "4 · Operational Monitoring": page_monitor,
    "5 · Simulated Live Monitoring": page_live,
    "6 · About the Data & Model": page_about,
}


def main() -> None:
    st.markdown('<div class="hero"><h1>🏭 AI-Based Manufacturing Efficiency Classification</h1>'
                '<p>Sensor, production and network data · High / Medium / Low efficiency · explainable, with honest limits</p></div>', unsafe_allow_html=True)
    try:
        df = load_data()
    except FileNotFoundError as err:
        st.error(f"Missing generated file ({err}). Run `python -m src.train_models` first, then reload.")
        st.stop()
    models = load_models()
    if "error" in models:
        st.error("The saved models could not be loaded (often a scikit-learn version difference). "
                 "Run `python -m src.train_models` once to re-create them. Details: " + models["error"])
        st.stop()
    background = df[df["split"] == "train"].sample(300, random_state=0)[utils.RAW_INPUTS].reset_index(drop=True)
    ctx = {"models": models, "tables": load_tables(), "background": background}
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")
    f = sidebar_filters(df)
    PAGES[page](df, f, ctx)


main()
