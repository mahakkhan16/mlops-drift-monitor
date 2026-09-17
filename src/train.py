import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# 1. Load data
df = pd.read_csv("data/reference/churn_reference.csv")

# 2. Basic cleaning
df = df.drop(columns=["customerID"])  # not a useful feature
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# 3. Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=["object", "str"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("F1 Score:", f1_score(y_test, preds))

os.makedirs("src/model", exist_ok=True)
joblib.dump(model, "src/model/churn_model.pkl")
joblib.dump(label_encoders, "src/model/label_encoders.pkl")

# 8. Save the cleaned reference data (evidently will need this later for drift comparison)
df.to_csv("data/reference/churn_reference_clean.csv", index=False)

print("Model and encoders saved to src/model/")
joblib.dump(scaler, "src/model/scaler.pkl")