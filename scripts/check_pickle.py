import joblib

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