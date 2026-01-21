from functools import lru_cache
from typing import Any, Dict, Tuple
from pathlib import Path
import joblib
import os

from .config import get_settings


@lru_cache()
def load_email_tfidf_rf() -> Tuple[Any, Any]:
    """
    Load the TF-IDF + RandomForest email model and vectorizer.
    
    This model uses adversarial robustness techniques (character substitution,
    noise injection) for improved generalization against phishing attacks.
    
    Returns:
        Tuple[vectorizer, model]: TF-IDF vectorizer and RandomForest classifier
        
    Raises:
        FileNotFoundError: If model file does not exist at configured path
    """
    settings = get_settings()
    model_path = settings.email_tfidf_rf_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Email TF-IDF RandomForest model not found at {model_path}. "
            "Please train the model first using the training pipeline."
        )
    
    data: Dict[str, Any] = joblib.load(model_path)
    
    if "vectorizer" not in data or "model" not in data:
        raise ValueError(
            f"Invalid model file structure at {model_path}. "
            "Expected dict with 'vectorizer' and 'model' keys."
        )
    
    return data["vectorizer"], data["model"]


@lru_cache()
def load_url_model() -> Any:
    """Load the URL RandomForest model (and optional scaler/feature_names)."""
    settings = get_settings()
    if not os.path.exists(settings.url_rf_path):
        raise FileNotFoundError(
            f"URL model not found at {settings.url_rf_path}. "
            "Run backend/train/train_url_model.py first."
        )
    return joblib.load(settings.url_rf_path)


def get_distilbert_model_dir() -> str:
    """
    Get the absolute path to the DistilBERT email model directory.
    
    Tries to load from:
    1. DISTILBERT_MODEL_PATH env var (if set)
    2. early_stop subdirectory (if exists)
    3. Root distilbert_email directory (fallback)
    
    This directory must contain: config.json, pytorch_model.bin (or model.safetensors), and tokenizer files.
    
    Returns:
        str: Absolute path to the DistilBERT model directory
        
    Raises:
        FileNotFoundError: If the model directory or required files do not exist
    """
    settings = get_settings()
    
    # Try custom env var path first
    if settings.distilbert_model_path:
        candidate = os.path.abspath(settings.distilbert_model_path)
        if os.path.exists(candidate):
            model_dir = candidate
        else:
            model_dir = None
    else:
        model_dir = None
    
    # Fallback: try early_stop subdirectory, then root directory
    if not model_dir:
        base_dir = settings.distilbert_email_dir
        early_stop_path = os.path.join(base_dir, "early_stop")
        root_path = base_dir
        
        if os.path.exists(early_stop_path):
            model_dir = os.path.abspath(early_stop_path)
        elif os.path.exists(root_path):
            model_dir = os.path.abspath(root_path)
        else:
            model_dir = None
    
    if not model_dir:
        raise FileNotFoundError(
            f"DistilBERT model directory not found. Tried: "
            f"{settings.distilbert_model_path or 'N/A'}, "
            f"{os.path.join(settings.distilbert_email_dir, 'early_stop')}, "
            f"{settings.distilbert_email_dir}. "
            "Please ensure the model is trained and saved to one of these locations."
        )
    
    abs_path = os.path.abspath(model_dir)
    
    # Validate required files exist (accept either pytorch_model.bin or model.safetensors)
    required_files = ["config.json", "tokenizer.json"]
    optional_model_files = ["pytorch_model.bin", "model.safetensors"]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(abs_path, f))]
    
    # Check if at least one model file exists
    has_model_file = any(os.path.exists(os.path.join(abs_path, f)) for f in optional_model_files)
    
    if missing_files:
        raise FileNotFoundError(
            f"DistilBERT model directory exists but is missing required files: {missing_files}. "
            f"Directory: {abs_path}"
        )
    
    if not has_model_file:
        raise FileNotFoundError(
            f"DistilBERT model directory is missing model weights. "
            f"Expected one of: {optional_model_files} in {abs_path}"
        )
    
    return abs_path


