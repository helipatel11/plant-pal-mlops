# Plant Pal

A healthy/unhealthy leaf classifier built as my first MLOps pipeline (November 2025): MobileNetV2 transfer learning, a ZenML pipeline, MLflow tracking and a local Streamlit app.

> **Status:** an early learning project that runs locally. The goal was to learn the MLOps toolchain end to end, not to build an accurate model, so it uses a small sample dataset. See [Limitations](#limitations) for what I'd improve. My later work is the [LLMOps Support Assistant](https://helipatel.com/projects/llmops-support-assistant/).

## What it does

- **Model:** MobileNetV2 pre-trained on ImageNet, base frozen, with one sigmoid output trained on top (binary: healthy / unhealthy).
- **Pipeline (ZenML):** `load_data_step` → `train_model_step` → `evaluate_model_step` (`src/zenml_pipeline.py`).
- **Tracking (MLflow):** the training step logs the saved model file as a run artifact.
- **App (Streamlit):** upload a leaf image, get a prediction and a confidence score, and leave feedback, which is saved under `feedback/` (images + CSV).

## Project layout

| Path | What |
|---|---|
| `src/zenml_pipeline.py` | Pipeline definition |
| `src/run_pipeline.py` | Entry point: runs the pipeline |
| `src/zenml_steps/` | Load, train and evaluate steps |
| `src/app/streamlit_app.py` | Prediction app with feedback form |
| `src/checks/simple_checks.py` | Basic data-generator sanity checks |
| `src/utils/data_utils.py` | Generator helper with augmentation (not used by the pipeline) |

## Run it locally

Needs Python 3.10+. Images go in `data/train/healthy/` and `data/train/unhealthy/` (the `data/` folder is gitignored).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
zenml init
python -m src.run_pipeline                 # trains and saves models/latest_model/plant_pal_model.keras
streamlit run src/app/streamlit_app.py     # prediction app
mlflow ui                                  # http://127.0.0.1:5000
```

## Limitations

- **One batch per split.** `load_data_step` uses `next()` on the generators, which returns a single batch (`batch_size=32`). That covers the current dataset, but a larger one would need the full generator.
- **Evaluation is basic.** Validation loss and accuracy are printed; there's no separate test set or per-class metrics yet.
- **MLflow logs the model file,** but not training metrics or parameters, so runs can't be compared side by side yet.
- **Dependencies aren't pinned** in `requirements.txt`, so installs may differ over time.
- **The "No — Incorrect" feedback path may need a fix.** It nests Streamlit buttons, and Streamlit resets button state on each rerun, so corrections might not be saved. Not yet verified.

## Next steps

- Iterate over the full generator so the pipeline scales to larger datasets.
- Hold out a test set and report per-class precision, recall and a confusion matrix.
- Log metrics and parameters to MLflow.
- Pin dependencies.

