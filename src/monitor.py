import pandas as pd
import json
import os
from evidently import Report
from evidently.presets import DataDriftPreset

# 1. Load reference data
reference = pd.read_csv("data/reference/churn_reference_clean.csv")
reference_features = reference.drop(columns=["Churn"])

os.makedirs("data/monitoring", exist_ok=True)

results = []

incoming_dir = "data/incoming"
batch_files = sorted(os.listdir(incoming_dir))

for batch_file in batch_files:
    batch_path = os.path.join(incoming_dir, batch_file)
    current = pd.read_csv(batch_path)
    current_features = current.drop(columns=["Churn"])

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_features, current_data=current_features)
    result_dict = snapshot.dict()

    metrics = result_dict["metrics"]

    # First metric is always the overall DriftedColumnsCount summary
    overall = metrics[0]
    drifted_count = float(overall["value"]["count"])
    drifted_share = float(overall["value"]["share"])

    # Remaining metrics are per-column ValueDrift results
    column_drift = []
    for m in metrics[1:]:
        col_name = m["config"]["column"]
        drift_score = float(m["value"])
        threshold = float(m["config"]["threshold"])
        drift_detected = bool(drift_score > threshold)
        column_drift.append({
            "column": col_name,
            "drift_score": drift_score,
            "threshold": threshold,
            "drift_detected": drift_detected
        })

    results.append({
        "batch": batch_file,
        "drifted_columns_count": drifted_count,
        "drifted_columns_share": drifted_share,
        "column_drift": column_drift
    })

    print(f"{batch_file}: drifted_count={drifted_count}, drifted_share={drifted_share:.2f}")

# Save summary as JSON for the dashboard to read
with open("data/monitoring/drift_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved drift summary to data/monitoring/drift_summary.json")