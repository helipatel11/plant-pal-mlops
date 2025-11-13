# src/zenml_pipeline.py

from zenml import pipeline
from src.zenml_steps.load_data_step import load_data_step
from src.zenml_steps.train_model_step import train_model_step
from src.zenml_steps.evaluate_model_step import evaluate_model_step

@pipeline
def plant_pal_pipeline():
    X_train, y_train, X_val, y_val = load_data_step()

    model_path = train_model_step(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    evaluate_model_step(
        model_path=model_path,
        X_val=X_val,
        y_val=y_val
    )
