# Executive Summary
## AI-Based Manufacturing Efficiency Classification Using Sensor, Production and Network Data

*Prepared for non-technical government stakeholders. Based on the supplied dataset of 100,000 records; all figures below were measured in this project unless marked as "possible".*
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
* **The efficiency label in this dataset is defined by two measurements only:** the error rate and the production speed. Records are "High" when the error rate is at most 2 % and the speed is above 400 units per hour, "Medium" when the error rate is at most 5 % and the speed above 200, and "Low" otherwise. This rule matches 99.998% of the 100,000 records.
* **Network and sensor conditions showed no relationship with efficiency in this dataset.** Network delay and data loss, temperature, vibration, power use, defect rate, maintenance score and operating mode were all statistically unrelated to the efficiency class.
* **No machine, operating mode or week stood out** beyond what chance would produce (50 machines were compared, with about 2000 records each).
* The data are very regular (every measurement is evenly spread between fixed limits), which suggests they are simulated rather than recorded in a real factory. One calendar date label (1 March 2025) contains two days of records, a data-quality problem that should be corrected at its source.
* Only 3.0% of records are "High" efficiency, 19.2% are "Medium" and 77.8% are "Low", so results were judged class by class, not only by overall accuracy.

### 6. Model performance (measured on later, unseen records)
| Question | Result |
|---|---|
| How well does the deployed tool classify when it receives the error rate and speed? | Near perfect: 100.00% of 15,000 test records correct; every High record found (balanced accuracy 1.000). |
| Why is it so accurate? | Because it has effectively learned the rule above. This shows the tool can apply the current efficiency definition, **not** that it can foresee problems. |
| How well can the sensors and network alone identify efficiency? | No better than chance (balanced accuracy at most 0.338; chance is 0.333), for three different methods, also when using each machine's recent history, and for machines the tool had never seen. |
| How reliable are the confidence figures? | Close to the observed outcomes on the test period (calibration error 0.017), but the highest probability shown is a model score, not a guarantee. |

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
