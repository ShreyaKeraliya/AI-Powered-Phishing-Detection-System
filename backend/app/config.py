import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI-Powered Phishing Detection API"
    backend_cors_origins: list[str] = ["*"]

    # Base directory
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Models directory
    models_dir: str = os.path.join(base_dir, "models")

    # Model paths
    email_tfidf_rf_path: str = os.path.join(models_dir, "email_tfidf_rf.joblib")
    url_rf_path: str = os.path.join(models_dir, "url_rf.pkl")
    distilbert_email_dir: str = os.path.join(models_dir, "distilbert_email")
    
    # DistilBERT model path (can be overridden via DISTILBERT_MODEL_PATH env var)
    # If not set, get_distilbert_model_dir() will auto-detect from early_stop or root directory
    distilbert_model_path: str | None = os.environ.get("DISTILBERT_MODEL_PATH")
    
    # Email prediction threshold
    email_phishing_threshold: float = 0.6

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
