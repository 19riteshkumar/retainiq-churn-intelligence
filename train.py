import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("[1/5] Downloading IBM Telco Customer Churn Dataset...")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("[2/5] Cleaning and Preprocessing Data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

customer_ids = df['customerID']
features = df.drop(columns=['customerID', 'Churn'])
features = pd.get_dummies(features, drop_first=True)

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    features, df['Churn'], customer_ids, test_size=0.20, random_state=42, stratify=df['Churn']
)
raw_test = df.loc[ids_test.index].copy()

print("[3/5] Scaling Numerical Features...")
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Handle Class Imbalance natively via XGBoost scale_pos_weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = float(neg_count) / float(pos_count)

print(f"[4/5] Training XGBoost Classifier (Class Weight: {scale_weight:.2f})...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_weight,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

def build_indian_demo_batch(feature_cols, scaler, row_count):
    """Create a deterministic, model-compatible Indian telecom demo batch."""
    rng = np.random.default_rng(2026)
    cities = [
        ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"),
        ("Bengaluru", "Karnataka"), ("Hyderabad", "Telangana"),
        ("Chennai", "Tamil Nadu"), ("Delhi", "Delhi"),
        ("Kolkata", "West Bengal"), ("Ahmedabad", "Gujarat"),
    ]
    city_rows = rng.integers(0, len(cities), size=row_count)
    city_names = [cities[index][0] for index in city_rows]
    states = [cities[index][1] for index in city_rows]
    tenure = rng.integers(1, 73, size=row_count)
    monthly = np.round(rng.uniform(399, 2499, size=row_count), 2)
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=row_count,
        p=[0.58, 0.25, 0.17],
    )
    internet = rng.choice(["Fiber optic", "DSL", "No"], size=row_count, p=[0.55, 0.30, 0.15])
    electronic_payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=row_count,
        p=[0.42, 0.18, 0.22, 0.18],
    )
    raw = pd.DataFrame({
        "customerID": [f"IN-{states[i][:2].upper()}-{i + 1:05d}" for i in range(row_count)],
        "City": city_names,
        "State": states,
        "gender": rng.choice(["Female", "Male"], size=row_count),
        "SeniorCitizen": rng.choice([0, 1], size=row_count, p=[0.88, 0.12]),
        "Partner": rng.choice(["Yes", "No"], size=row_count, p=[0.48, 0.52]),
        "Dependents": rng.choice(["Yes", "No"], size=row_count, p=[0.32, 0.68]),
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": rng.choice(["Yes", "No"], size=row_count, p=[0.38, 0.62]),
        "InternetService": internet,
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=row_count, p=[0.28, 0.57, 0.15]),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=row_count, p=[0.30, 0.55, 0.15]),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=row_count, p=[0.27, 0.58, 0.15]),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=row_count, p=[0.24, 0.61, 0.15]),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=row_count, p=[0.34, 0.51, 0.15]),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=row_count, p=[0.35, 0.50, 0.15]),
        "Contract": contracts,
        "PaperlessBilling": rng.choice(["Yes", "No"], size=row_count, p=[0.72, 0.28]),
        "PaymentMethod": electronic_payment,
        "MonthlyCharges": monthly,
    })
    raw["TotalCharges"] = np.round(raw["tenure"] * raw["MonthlyCharges"], 2)
    raw["Churn"] = np.where(
        (raw["Contract"] == "Month-to-month") & (raw["tenure"] < 18), "Yes", "No"
    )

    demo_features = pd.get_dummies(raw.drop(columns=["customerID", "Churn", "City", "State"]), drop_first=True)
    demo_features = demo_features.reindex(columns=feature_cols, fill_value=0).astype(float)
    demo_features[num_cols] = scaler.transform(demo_features[num_cols])
    return demo_features, raw


print("[5/5] Exporting Model Artifacts...")
feature_cols = list(features.columns)
X_demo, raw_demo = build_indian_demo_batch(feature_cols, scaler, len(X_test))

# Keep the demo batch compatible with both the current app and older versions.
sample_test_batch = {
    "X_test": X_demo,
    "y_test": raw_demo["Churn"].map({"Yes": 1, "No": 0}),
    "CustomerID": raw_demo["customerID"],
    "customer_ids": raw_demo["customerID"],
    "X_test_raw": raw_demo,
}

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_cols, 'columns.pkl')
joblib.dump(sample_test_batch, 'sample_test_batch.pkl')

print("\nSUCCESS: All 4 artifacts created successfully in expected dictionary format!")