# src/app/streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import pandas as pd
from datetime import datetime

# --- Compute model path relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "..", "models", "latest_model", "plant_pal_model.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)

FEEDBACK_CSV = os.path.join(SCRIPT_DIR, "..", "..", "feedback", "feedback.csv")
FEEDBACK_IMG_DIR = os.path.join(SCRIPT_DIR, "..", "..", "feedback", "images")

st.set_page_config(page_title="Plant Pal", layout="centered")

@st.cache_resource
def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Run training first.")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image: Image.Image, target_size=(224,224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, 0)  # batch dimension
    return arr

def ensure_feedback_dirs():
    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    os.makedirs(FEEDBACK_IMG_DIR, exist_ok=True)

def save_feedback(image_bytes, filename, predicted_label, user_label, comment=""):
    ensure_feedback_dirs()
    timestamp = datetime.utcnow().isoformat()
    img_path = os.path.join(FEEDBACK_IMG_DIR, filename)
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    # append to CSV
    df = pd.DataFrame([{
        "timestamp": timestamp,
        "image_path": img_path,
        "predicted": predicted_label,
        "user_label": user_label,
        "comment": comment
    }])
    if os.path.exists(FEEDBACK_CSV):
        df.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(FEEDBACK_CSV, mode='w', header=True, index=False)

# --- UI ---
st.title("🌿 Plant Pal")
st.write("Upload a leaf image to predict healthy vs unhealthy. Submit feedback to improve the model.")

model = load_model()
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img_bytes = uploaded.read()
    image = Image.open(io.BytesIO(img_bytes))
    st.image(image, caption="Uploaded image", use_column_width=True)

    if model is not None:
        x = preprocess_image(image)
        pred = model.predict(x)[0][0]
        conf = float(pred)
        label = "Unhealthy" if conf >= 0.5 else "Healthy"
        st.markdown(f"**Prediction:** `{label}` — **confidence:** {conf:.2f}")

        st.write("---")
        st.subheader("Was this prediction correct?")
        cols = st.columns(2)
        with cols[0]:
            if st.button("Yes — Correct"):
                save_feedback(img_bytes, f"fb_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg", label, label, comment="")
                st.success("Thanks — feedback recorded.")
        with cols[1]:
            if st.button("No — Incorrect"):
                user_label = st.radio("Select correct label", options=["Healthy", "Unhealthy"])
                comment = st.text_input("Optional comment")
                if st.button("Submit correction"):
                    save_feedback(img_bytes, f"fb_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg", label, user_label, comment=comment)
                    st.success("Thanks — correction recorded.")

st.write("")
st.markdown("---")
st.write("Feedback is stored locally under `feedback/` (images + CSV). Use these to retrain later.")
