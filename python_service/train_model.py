import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "training_data.csv"
model_path = BASE_DIR / "risk_model.joblib"

data = pd.read_csv(data_path)

X = data.drop(columns=["label"])
y = data["label"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X, y)

joblib.dump(model, model_path)
print(f"Model trained and saved to {model_path}.")
print("Rows:", len(data))
print("Labels:", y.value_counts().to_dict())
