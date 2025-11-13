# src/zenml_steps/train_model_step.py

from zenml import step
from typing import Tuple
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import mlflow
import os

@step
def train_model_step(
    X_train,
    y_train,
    X_val,
    y_val
) -> str:

    mlflow.start_run()

    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))

    # os.makedirs("models", exist_ok=True)
    # model_path = "models/plant_model.h5"
    # model.save(model_path)

    os.makedirs("models/latest_model", exist_ok=True)
    model_path = "models/latest_model/plant_pal_model.keras"
    model.save(model_path)    

    mlflow.log_artifact(model_path)
    mlflow.end_run()

    return model_path
