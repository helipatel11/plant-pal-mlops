from src.zenml_pipeline import plant_pal_pipeline

if __name__ == "__main__":
    run = plant_pal_pipeline()
    print(f"✅ Pipeline run completed: {run.name}")
