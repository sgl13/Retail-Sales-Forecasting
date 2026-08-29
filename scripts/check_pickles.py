import joblib
import numpy as np

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

for name in ["lightgbm_model.pkl", "label_encoders.pkl", "metrics.pkl"]:
    path = f"models/{name}"
    try:
        obj = joblib.load(path)
        print(f"{name}: OK -> {type(obj)}")
        if isinstance(obj, dict):
            for k, v in obj.items():
                print(f"   {k}: {type(v)}")
    except Exception as e:
        print(f"{name}: FAILED -> {e}")
