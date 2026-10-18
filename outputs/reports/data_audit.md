# Data audit summary

* Rows x columns: **100,000 x 14**; columns match the expected schema: **True**
* Missing values: **0**; duplicate rows: **0**
* Machines: **50**; operation modes: {'Active': 70054, 'Idle': 20057, 'Maintenance': 9889}
* Target distribution: {'Low': 77825, 'Medium': 19189, 'High': 2986} (Low 77.8%, Medium 19.2%, High 3.0%)
* All features distributed like a uniform distribution (KS distance < 0.02): **True**
* Max |correlation| between features: **0.008**
* Time anomalies: {'86400': -1439.0}; dates with row count != 1440: {'01-03-2025': 2880, '10-03-2025': 640}

## Label derivation (leakage check)
* Rule: High = `Error_Rate_% <= 2  AND  Production_Speed_units_per_hr > 400`; Medium = `Error_Rate_% <= 5  AND  Production_Speed_units_per_hr > 200  (and not High)`; Low = else
* Agreement of rule with the label: **99.998%** (2 mismatches)
* Depth-4 decision-tree test accuracy by inputs: {'Error_Rate_% only': 0.8576666666666667, 'Production_Speed only': 0.7777333333333334, 'both (Error_Rate_% + Production_Speed)': 1.0, 'all other 7 measurements + mode (no Error_Rate_%, no Speed)': 0.7778}
* Majority-class accuracy: 77.79%

**Conclusion.** Efficiency_Status is (to within the listed mismatches) a deterministic function of Error_Rate_% and Production_Speed_units_per_hr. Any model that receives these two columns can reproduce the label, which demonstrates recovery of the labelling rule - not independent prediction of efficiency from sensors or the network.