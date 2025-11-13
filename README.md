# Plant Pal (MobileNetV2) — Local MLOps Starter

## What this project contains
- MobileNetV2-based binary classifier (healthy / unhealthy).
- MLflow logging of training runs and artifacts.
- ZenML pipeline stub to orchestrate training.
- Simple checks and a Streamlit app for inference + feedback collection.

## Setup (local)
1. Create virtual env and install:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
source .venv/bin/activate


# 🌿 Plant Pal — ML-powered Plant Disease Classification

## 🧠 Overview

**Plant Pal** is a machine learning application that identifies plant leaf diseases using deep learning.
The project integrates **ZenML** for MLOps orchestration, **MLflow** for experiment tracking, and **Streamlit** for an interactive web interface.

The model is built using **TensorFlow’s MobileNetV2** architecture and trained on a labeled dataset of healthy and diseased leaves.

---

## 🧰 Tools & Technologies

| Tool                              | Purpose                                     |
| --------------------------------- | ------------------------------------------- |
| **TensorFlow / Keras**            | Model training using MobileNetV2            |
| **ZenML**                         | Pipeline orchestration and step management  |
| **MLflow**                        | Tracking model runs, metrics, and artifacts |
| **Streamlit**                     | Interactive UI for testing trained models   |
| **NumPy / Pandas / scikit-learn** | Data preprocessing and evaluation           |
| **Python 3.10+**                  | Core programming language                   |
| **Virtual Environment**           | Dependency isolation                        |

---

## ⚙️ Architecture Overview

```
📦 Plant Pal
├── models/
│   └── latest_model/
│       └── plant_pal_model.keras      # Latest trained model
│
├── src/
│   ├── zenml_steps/
│   │   ├── load_data_step.py         # Loads and splits dataset
│   │   ├── train_model_step.py       # Trains and saves model
│   │   └── evaluate_model_step.py    # Evaluates model performance
│   │
│   ├── run_pipeline.py               # ZenML pipeline entry point
│   ├── utils/                        # Helper functions (optional)
│   └── streamlit_app.py              # Streamlit UI to test model
│
├── .venv/                            # Virtual environment (ignored in git)
├── requirements.txt
└── README.md
```

---

## 🧩 ZenML Pipeline Workflow

The ZenML pipeline includes 3 steps:

1. **`load_data_step`** — Loads and splits the dataset into train/validation sets.
2. **`train_model_step`** — Trains MobileNetV2 and saves model to `models/latest_model/plant_pal_model.keras`.
3. **`evaluate_model_step`** — Evaluates accuracy, loss, and logs metrics to MLflow.

---

## 🧑‍💻 Setup Instructions

### 1️⃣ Clone and navigate to the project

```bash
git clone https://github.com/<your-repo>/plant-pal.git
cd "Plant Pal"
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# or
.\.venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Initialize ZenML

```bash
zenml init
```

### 5️⃣ Run the ZenML pipeline

```bash
python -m src.run_pipeline
```

This will:

* Train your model
* Log metrics to MLflow
* Save the model to `models/latest_model/plant_pal_model.keras`

---

## 🔍 Experiment Tracking with MLflow

Start the MLflow UI to visualize training metrics and models:

```bash
mlflow ui
```

Then visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🌱 Running the Streamlit App

Once your model is trained, launch the app to test predictions:

```bash
streamlit run src/streamlit_app.py
```

**What it does:**

* Loads model from `models/latest_model/plant_pal_model.keras`
* Allows uploading a leaf image
* Displays predicted class (Healthy / Diseased)

---

## 🧾 Example Commands Summary

| Action                    | Command                              |
| ------------------------- | ------------------------------------ |
| Activate venv (Mac/Linux) | `source .venv/bin/activate`          |
| Activate venv (Windows)   | `.\.venv\Scripts\activate`           |
| Run ZenML pipeline        | `python -m src.run_pipeline`         |
| Launch Streamlit UI       | `streamlit run src/streamlit_app.py` |
| Start MLflow tracking UI  | `mlflow ui`                          |

---

## 🧱 Future Improvements

* ✅ Integrate AWS S3 artifact storage for models
* ✅ Add CI/CD using GitHub Actions for auto-deployment
* 🔄 Add data versioning using DVC
* ☁️ Deploy Streamlit app on AWS EC2 / Streamlit Cloud
* 🧠 Experiment with EfficientNetB0 and fine-tuning

---

## 🧑‍🎓 Author

**Heli Patel**

* M.Sc. Computer Science — Carleton University
* Specializing in MLOps, Cloud Infrastructure (AWS), and Machine Learning
* [LinkedIn](https://linkedin.com/in/heli-patel) | [GitHub](https://github.com/heli-patel)

---


