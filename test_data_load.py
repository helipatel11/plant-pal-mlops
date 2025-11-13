from zenml import step
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

# Absolute path to your data folder
DATA_DIR = Path(os.path.expanduser("~/Documents/Plant Pal/data"))

def load_images_from_folder(folder: Path, label: int, img_size=(128, 128)):
    images, labels = [], []
    for img_path in folder.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels

@step
def load_data_step() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    healthy_dir = DATA_DIR / "healthy"
    unhealthy_dir = DATA_DIR / "unhealthy"

    if not healthy_dir.exists() or not unhealthy_dir.exists():
        raise FileNotFoundError(
            f"Check your data folders! Expected at:\n"
            f"Healthy: {healthy_dir}\nUnhealthy: {unhealthy_dir}"
        )

    healthy_imgs, healthy_labels = load_images_from_folder(healthy_dir, 0)
    unhealthy_imgs, unhealthy_labels = load_images_from_folder(unhealthy_dir, 1)

    X = np.array(healthy_imgs + unhealthy_imgs, dtype="float32") / 255.0
    y = np.array(healthy_labels + unhealthy_labels, dtype="float32")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return x_train, y_train, x_test, y_test
