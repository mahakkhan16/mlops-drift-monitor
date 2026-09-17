import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="ML Drift Monitor", layout="wide")

st.title("📊 ML Model Drift Monitoring Dashboard")
st.caption("Tracking data drift across simulated production batches for the churn model")

# Load drift summary
json_path = os.path.join(os.path.dirname(__file__), "..", "data", "monitoring", "drift_summary.json")
with open(json_path, "r") as f:
    results = json.load(f)

# Build a summary dataframe: one row per batch
summary_rows = []
for r in results:
    summary_rows.append({
        "Batch": r["batch"],
        "Drifted Columns": int(r["drifted_columns_count"]),
        "Drifted Share": r["drifted_columns_share"]
    })
summary_df = pd.DataFrame(summary_rows)

# --- Top-level status indicator ---
latest = summary_df.iloc[-1]
latest_share = latest["Drifted Share"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Latest Batch", latest["Batch"])
with col2:
    st.metric("Drifted Columns (latest)", int(latest["Drifted Columns"]))
with col3:
    if latest_share >= 0.3:
        st.error(f"🔴 High Drift: {latest_share:.0%}")
    elif latest_share >= 0.15:
        st.warning(f"🟡 Moderate Drift: {latest_share:.0%}")
    else:
        st.success(f"🟢 Low Drift: {latest_share:.0%}")

st.divider()

# --- Drift over time chart ---
st.subheader("Drift Share Over Time")
st.line_chart(summary_df.set_index("Batch")["Drifted Share"])

st.subheader("Drifted Column Count Over Time")
st.bar_chart(summary_df.set_index("Batch")["Drifted Columns"])

st.divider()

# --- Per-batch column-level detail ---
st.subheader("Column-Level Drift Detail")
selected_batch = st.selectbox("Select a batch to inspect", [r["batch"] for r in results])

selected_result = next(r for r in results if r["batch"] == selected_batch)
col_df = pd.DataFrame(selected_result["column_drift"])
col_df = col_df.sort_values("drift_score", ascending=False)

def highlight_drift(row):
    return ["background-color: #ffcccc" if row["drift_detected"] else "" for _ in row]

st.dataframe(col_df.style.apply(highlight_drift, axis=1), use_container_width=True)

st.divider()
st.caption("Built as part of an end-to-end MLOps drift monitoring pipeline — model, drift detection (Evidently), dashboard (Streamlit), CI/CD (GitHub Actions).")