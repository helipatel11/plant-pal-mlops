# src/zenml_steps/evaluate_model_step.py

from zenml import step
from tensorflow.keras.models import load_model

@step
def evaluate_model_step(model_path: str, X_val, y_val) -> None:

    model = load_model(model_path)
    loss, acc = model.evaluate(X_val, y_val)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Loss: {loss:.4f}")
