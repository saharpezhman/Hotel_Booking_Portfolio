from pathlib import Path
import joblib

def convert_to_string(X):
    return X.astype(str)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hotel_cancellation_model.pkl"

print("Checking saved model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Model type:", type(model).__name__)
print("Pipeline steps:", list(model.named_steps.keys()))