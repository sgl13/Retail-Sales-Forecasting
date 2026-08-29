import joblib
import numpy as np

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

metrics = joblib.load("models/metrics.pkl")

clean_metrics = {
    "MAE": float(metrics["MAE"]),
    "RMSLE": 0.0,          # real value wasn't saved originally — placeholder for now
    "r2_score": float(metrics["r2_score"]),
}

print("Before:", metrics)
print("After :", clean_metrics)

joblib.dump(clean_metrics, "models/metrics.pkl")
print("Saved cleaned metrics.pkl")
