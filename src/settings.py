import os

from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "bank_marketing_classification"
)
MODELS_DIR = os.getenv("MODEL_DIR", "models")
API_MODEL_PATH = os.getenv("API_MODEL_PATH", f"{MODELS_DIR}/latest")
