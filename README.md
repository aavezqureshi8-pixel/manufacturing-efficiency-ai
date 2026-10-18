# AI-Based Manufacturing Efficiency Classification Using Sensor, Production and 6G Network Data

**Domain:** AI · Machine Learning · Smart Manufacturing · Industrial IoT · Network Analytics
The brief mentions Unified Mentor and Thales Group. This project makes **no claim** of approval, endorsement or deployment by either.

---

## ⚠️ The most important finding (read first)
The data audit shows that the label `Efficiency_Status` is **a rule applied to two input columns**:

| Class | Rule (matches 99.998% of the 100,000 records) |
|---|---|
| High | `Error_Rate_% ≤ 2` **and** `Production_Speed_units_per_hr > 400` |
| Medium | `Error_Rate_% ≤ 5` **and** `Production_Speed_units_per_hr > 200` (and not High) |
| Low | everything else |

* Models that receive those two columns score ~100% (this **recovers the rule**; it is not evidence that sensors or the network predict efficiency).
* Without them, all three algorithms drop to **chance level** (best test balanced accuracy 0.338 vs 0.333 for chance) - with sensor, network, quality, maintenance and past-only history features, and also for machines never seen in training.
* The dashboard therefore labels the model honestly, and also offers a "context-only" comparison model.

## 1. Project overview
End-to-end project: data audit → leak-free preprocessing → feature engineering → comparison of Logistic Regression, Random Forest and Gradient Boosting under chronological evaluation → explainability → Streamlit dashboard → research paper and executive summary.

## 2. Problem statement
Factories face delayed detection of efficiency degradation, manual interpretation of many metrics and no automated, interpretable assessment. The project classifies the **current observed** efficiency state (High / Medium / Low), shows prediction confidence and explains the influential inputs. It does **not** forecast future breakdowns.

## 3. Objectives
Audit the data · prevent leakage · compare three models with imbalance handling · test whether sensor / network / quality data carry signal · explain predictions · deliver dashboard, paper and executive summary.

## 4. Dataset (verified from the file, not assumed)
`data/Thales_Group_Manufacturing.csv` - 100,000 rows × 14 columns, 1 record per minute (2025-01-01 → 2025-03-10), 50 machines, modes Active/Idle/Maintenance.
No missing values or duplicates. Classes: Low 77.8%, Medium 19.2%, **High 3.0%**.
Data-quality findings: all measurements are uniformly distributed and almost uncorrelated (looks simulated); the date label `01-03-2025` holds **two days** of records (likely a rolled-over 29-Feb-2025; a hypothesis); each machine has only sparse, irregular records (median gap 35 min).

## 5. Technologies
Python · Pandas · NumPy · SciPy · Scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit · Joblib.
*XGBoost and SHAP were not used*: Gradient Boosting is scikit-learn's `HistGradientBoostingClassifier`, and explanations use permutation importance plus a Monte-Carlo Shapley estimator written in `src/explainability.py`.

## 6. Methodology
Chronological split 70/15/15 (no shuffling) · preprocessing fitted on training rows only · experiment grid 3 models × 4 feature sets × 2 class-weight settings · selection on **validation** results only · test set reported afterwards · stability via expanding-window time folds and unseen-machine folds · calibration (ECE, Brier) · permutation importance and Shapley attributions.

## 7. Model comparison (test set, deployed feature set, class-weighted)
| Model | Macro-F1 | Balanced accuracy |
|---|---|---|
| Logistic Regression | 0.852 | 0.924 |
| Random Forest (**selected**) | 1.000 | 1.000 |
| Gradient Boosting | 0.999 | 0.999 |

Full tables (per-class metrics, confusion matrices, training time, stability, ablations): `research_paper.md`, `outputs/metrics/`.

## 8. Dashboard features (`app.py`)
1. **Efficiency Prediction** - enter or load a record; class, P(High/Medium/Low), highest probability, warnings for invalid/unusual inputs, local explanation, metric **sensitivity sliders (scenario simulation)**, CSV download, batch CSV upload.
2. **Machine-Level Insights** - per-machine trends (temperature, vibration, power, latency, packet loss, speed, error, defects), time-window filter; recorded labels and model predictions kept separate.
3. **Explainability Panel** - permutation importance, Shapley attributions, class comparisons, individual explanations, sensitivity curves.
4. **Operational Monitoring** - efficiency by mode and network quality, network vs sensor impact, machine × week heat-map, machine ranking **with number of observations and 95% confidence intervals** (not a fault diagnosis).
5. **Simulated Live Monitoring** - replays historical rows (no live feed, no 6G link) + notes on connecting a real source.
6. **About the Data & Model** - audit findings, evaluation set-ups, limitations.

Sidebar filters: network quality · machine · time window · operation mode · efficiency class · error-rate and speed ranges.

## 9. Installation
```bash
python -m venv venv
# Windows: venv\Scripts\activate        macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## 10. Train the models (optional - trained files are included)
```bash
python -m src.data_audit        # audit only (seconds)
python -m src.train_models      # audit + all experiments + models + figures (about 6 min on one CPU core)
python -m src.report_builder    # regenerate research_paper.md and executive_summary.md from the results
```
Run training again if you see a scikit-learn version warning when the dashboard loads the models.

## 11. Launch the dashboard
```bash
streamlit run app.py
```

## 12. Tests
```bash
python tests/test_pipeline.py          # 18 checks of data, leakage safeguards, metrics, models, explanations, filters, CSV
python tests/test_app_streamlit.py     # starts every dashboard page with Streamlit's own test runner
```

## 13. Project structure
```
manufacturing_efficiency_ai/
├── data/                 Thales_Group_Manufacturing.csv   (+ processed/clean_data.csv.gz: cleaned data + features + predictions)
├── models/               best_model.joblib, preprocessor.joblib, model_metadata.json (+ context_model / context_preprocessor)
├── outputs/              figures/ (PNG) · metrics/ (CSV/JSON) · reports/data_audit.md
├── src/                  data_audit · preprocessing · feature_engineering · train_models · evaluation · explainability ·
│                         inference · dashboard_logic · figures · report_builder · utils
├── tests/                test_pipeline.py · test_app_streamlit.py
├── app.py · requirements.txt · README.md · research_paper.md · executive_summary.md
```

## 14. Results (summary)
* Deployed model: Random Forest, test macro-F1 1.0000, balanced accuracy 1.0000, High-class F1 1.0000 (15,000 test records, 463 High); calibration error 0.017, Brier 0.0025.
* Sensors / network / quality / history alone: chance level; no machine, mode or week differs beyond chance.
* Stable across five time folds and five unseen-machine folds.

## 15. Limitations
Data appear simulated · scores reflect recovery of a labelling rule · the label is the *current* state (no failure prediction) · attributions are associations, not causes · replay is not live monitoring · confidence assessed on one test period.
**Testing note:** the build environment could not install Streamlit/Plotly, so the dashboard was smoke-tested with stand-in modules (all pages, many filter combinations, edge cases), not in a real browser. `tests/test_app_streamlit.py` is provided to test it for real on your computer.

## 16. Future scope
Independent outcome labels (downtime, defects, failures) · real time stamps and streaming ingestion · per-machine models and drift monitoring · time-to-failure modelling · network telemetry from a real 5G/6G testbed · SHAP / XGBoost comparison.
