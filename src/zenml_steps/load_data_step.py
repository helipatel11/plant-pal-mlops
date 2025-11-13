# src/zenml_steps/load_data_step.py
from zenml import step
from typing import Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

@step
def load_data_step() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        "data/train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        "data/train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        subset="validation"
    )

    X_train, y_train = next(train_gen)
    X_val, y_val = next(val_gen)

    return X_train, y_train, X_val, y_val
