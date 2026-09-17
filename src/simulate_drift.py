import pandas as pd
import numpy as np
import os

# 1. Load the cleaned reference data
df = pd.read_csv("data/reference/churn_reference_clean.csv")

os.makedirs("data/incoming", exist_ok=True)

NUM_BATCHES = 10          # how many "days" of incoming data to simulate
BATCH_SIZE = 200          # rows per batch

np.random.seed(42)

for i in range(NUM_BATCHES):
    batch = df.sample(n=BATCH_SIZE, replace=True, random_state=i).copy()

    # Gradually shift 'tenure' downward over time (simulating newer customers coming in)
    shift_factor = i * 2
    batch["tenure"] = (batch["tenure"] - shift_factor).clip(lower=0)

    # Gradually shift 'MonthlyCharges' upward over time (simulating a price increase)
    batch["MonthlyCharges"] = batch["MonthlyCharges"] + (i * 1.5)

    # Save this batch with a batch number so we can process them in order
    batch.to_csv(f"data/incoming/batch_{i:02d}.csv", index=False)
    print(f"Saved batch_{i:02d}.csv with {len(batch)} rows")

print("\nDone. Generated", NUM_BATCHES, "incoming batches in data/incoming/")