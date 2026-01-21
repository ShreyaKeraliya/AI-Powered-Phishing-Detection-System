import logging
from typing import Any, Dict
import time


from fastapi import APIRouter, HTTPException
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from ..schemas.email_schema import EmailRequest, EmailResponse, EmailModelType
from ..services.email_preprocessing import clean_email_text
from ..services.explainability import top_tfidf_phishing_terms
from ..utils import load_email_tfidf_rf, get_distilbert_model_dir
from ..config import get_settings

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api", tags=["email"])


_distilbert_model = None
_distilbert_tokenizer = None


def _load_distilbert() -> Dict[str, Any]:
    """
    Load DistilBERT model and tokenizer from local directory.
    
    Uses safetensors format (model.safetensors) and validates that all required
    files exist before attempting to load. Implements lazy loading with caching.
    """
    global _distilbert_model, _distilbert_tokenizer
    
    # Return cached models if already loaded
    if _distilbert_model is not None and _distilbert_tokenizer is not None:
        return {
            "model": _distilbert_model,
            "tokenizer": _distilbert_tokenizer,
        }

    # Get validated model directory path
    try:
        model_dir = get_distilbert_model_dir()
        logger.info(f"Loading DistilBERT model from: {model_dir}")
    except FileNotFoundError as e:
        logger.error(f"DistilBERT model directory validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"DistilBERT model not available: {e}"
        )

    # Load tokenizer and model from local directory
    try:
        _distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        _distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        _distilbert_model.eval()
        logger.info("DistilBERT model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load DistilBERT model from {model_dir}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load DistilBERT email model from {model_dir}: {str(e)}. "
                   "Please ensure all model files (config.json, model.safetensors, tokenizer.json) exist."
        )

    return {
        "model": _distilbert_model,
        "tokenizer": _distilbert_tokenizer,
    }


@router.post("/predict-email", response_model=EmailResponse)
def predict_email(request: EmailRequest) -> EmailResponse:
    processing_time = None
    """
    Predict phishing vs legitimate email using TF-IDF+RandomForest or DistilBERT.
    
    The TF-IDF+RandomForest model uses adversarial robustness techniques
    (character substitution, noise injection) for improved generalization
    against phishing attacks.
    """
    cleaned = clean_email_text(request.subject, request.body)
    if not cleaned:
        raise HTTPException(
            status_code=400, 
            detail="Email content is empty after preprocessing."
        )

    # Route to appropriate model based on enum
    if request.model_type == EmailModelType.tfidf_rf:
        # Load TF-IDF + RandomForest model
        try:
            vectorizer, model = load_email_tfidf_rf()
            logger.info("TF-IDF RandomForest model loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model loading failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Email phishing model not available: {e}"
            )
        
        start_time = time.time()

        # Transform text and predict
        X = vectorizer.transform([cleaned])
        proba = model.predict_proba(X)[0, 1]
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 4)
        # Use configurable threshold (default 0.6)
        settings = get_settings()
        threshold = settings.email_phishing_threshold
        label = "phishing" if proba >= threshold else "legitimate"
        
        # Extract suspicious indicators
        suspicious_indicators = top_tfidf_phishing_terms(
            model, vectorizer, cleaned, top_k=5
        )
        
        logger.info(
            f"Email prediction completed: label={label}, "
            f"probability={proba:.4f}, threshold={threshold}"
        )
        
        return EmailResponse(
            label=label,
            probability=float(proba),
            model_used="tfidf_rf_adversarial",
            explanations=suspicious_indicators,
            processing_time=f"{processing_time}s" if processing_time else None,
        )


    elif request.model_type == EmailModelType.distilbert:
        # Load DistilBERT model
        try:
            bundle = _load_distilbert()
        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except Exception as e:
            logger.error(f"Unexpected error loading DistilBERT: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load DistilBERT model: {str(e)}"
            )
        
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]

        start_time = time.time()

        encoded = tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            # Apply softmax: probs[0] = legitimate, probs[1] = phishing
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        end_time = time.time()
        processing_time = round(end_time - start_time, 4)


        # Model training mapping: 0 = legitimate, 1 = phishing
        legit_prob = float(probs[0])
        phish_prob = float(probs[1])
        
        # Use phishing probability for decision
        label = "phishing" if phish_prob >= 0.5 else "legitimate"
        # Return the probability of the predicted class
        probability = phish_prob if label == "phishing" else legit_prob

        # Use TF-IDF RandomForest for explanations (fallback to simple extraction if unavailable)
        try:
            vectorizer, rf_model = load_email_tfidf_rf()
            explanations = top_tfidf_phishing_terms(rf_model, vectorizer, cleaned, top_k=5)
        except FileNotFoundError:
            # Fallback if RF model not available
            explanations = []

        logger.info(
            f"DistilBERT prediction completed: label={label}, "
            f"probability={probability:.4f} (phishing_prob={phish_prob:.4f}, legit_prob={legit_prob:.4f})"
        )

        return EmailResponse(
            label=label,
            probability=probability,
            model_used="distilbert",
            explanations=explanations,
            processing_time=f"{processing_time}s" if processing_time else None,
        )

    
    else:
        # Invalid model type (shouldn't happen due to FastAPI validation, but safety check)
        raise HTTPException(
            status_code=422,
            detail=f"Invalid model_type: {request.model_type}. Must be 'tfidf_rf' or 'distilbert'"
        )


