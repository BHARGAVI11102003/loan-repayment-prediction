# loan-repayment-prediction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

app = FastAPI()

DATA_FILE = r"C:\Users\bhargavi k\Downloads\CSV Files\borrower_dataset_21.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"DATASET File {DATA_FILE} does not exist")

df = pd.read_csv(DATA_FILE, dtype={"borrower_id": int})

# Clean columns: strip spaces, lowercase, replace spaces with underscore
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("Columns after cleaning:", df.columns.tolist())

# Convert date columns to datetime
df['scheduled_repayment_date'] = pd.to_datetime(df['scheduled_repayment_date'], errors='coerce', dayfirst=True)
df['actual_repayment_date'] = pd.to_datetime(df['actual_repayment_date'], errors='coerce', dayfirst=True)

# Calculate repayment_percentage if not present
if 'repayment_percentage' not in df.columns:
    df['payment_on_time'] = (df['actual_repayment_date'] <= df['scheduled_repayment_date']).astype(int)
    df['total_payments'] = df.groupby('borrower_id')['payment_on_time'].transform('count')
    df['repayment_percentage'] = df.groupby('borrower_id')['payment_on_time'].transform('mean') * 100

# Define feature columns
feature_columns = [
    "loan_duration_days",
    "loan_amount",
    "borrower_credit_score",
    "borrower_income",
    "borrower_history_delays"
]

# Check for missing feature columns
missing = [col for col in feature_columns if col not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")

# Target variable
y = df['repayment_percentage']
X = df[feature_columns]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "repayment_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model training completed")

# Load model and scaler for prediction
model = joblib.load("repayment_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/test")
def test():
    return {"status": "FastAPI is working 🎉"}

class PredictRequest(BaseModel):
    borrower_id: int

@app.post("/predict-repayment")
def predict_repayment(request: PredictRequest) -> Dict:
    try:
        borrower_id = request.borrower_id
        record = df[df["borrower_id"] == borrower_id]

        if record.empty:
            raise HTTPException(status_code=404, detail="Borrower details not found")

        # Calculate delay in days
        record["days_late"] = (record["actual_repayment_date"] - record["scheduled_repayment_date"]).dt.days

        # Classify repayment status
        def classify_status(days_late):
            if pd.isna(days_late):
                return "unknown"
            elif days_late <= 0:
                return "on_time"
            elif days_late <= 30:
                return "late"
            else:
                return "defaulted"

        record["predicted_status"] = record["days_late"].apply(classify_status)

        # Count repayment statuses
        on_time_count = (record["predicted_status"] == "on_time").sum()
        late_count = (record["predicted_status"] == "late").sum()
        defaulted_count = (record["predicted_status"] == "defaulted").sum()
        total_payments = len(record)

        # ML Prediction
        X_input = record[feature_columns]
        X_scaled_input = scaler.transform(X_input)
        prediction = model.predict(X_scaled_input).mean()

        # Risk level based on prediction
        risk_level = (
            "low" if prediction >= 90 else
            "medium" if prediction >= 70 else
            "high"
        )

                summary = {
    # Summarize payment status counts (from categorical payment_status)
        "on_time_count" :int(on_time_count),
        "missed_count" : int(missed_count),
        "due_count" :int(due_count),
        "total_payments" :int(total_payments),
        "on_time_ratio": float(round(on_time_count / total_payments, 4)) if total_payments > 0 else 0.0,
        "missed_ratio": float(round(missed_count / total_payments, 4)) if total_payments > 0 else 0.0,
        "due_ratio": float(round(due_count / total_payments, 4)) if total_payments > 0 else 0.0,
    "average_delay_days": float(round(record["borrower_history_delays"].mean(), 2)) if "borrower_history_delays" in record.columns else None,
    "max_delay_days": int(record["borrower_history_delays"].max()) if "borrower_history_delays" in record.columns else None
}

        output = {
            "borrower_repayment_summary": summary,
            "predicted_repayment_percentage": float(round(prediction, 2)),
            "risk_level": risk_level
        }

        print(f"Output for Borrower {borrower_id}:\n{output}\n")
        return output

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

