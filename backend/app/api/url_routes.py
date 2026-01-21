from fastapi import APIRouter, HTTPException
import numpy as np
import time


from ..schemas.url_schema import UrlRequest, UrlResponse
from ..services.url_feature_extraction import extract_url_features
from ..services.explainability import url_feature_importance
from ..utils import load_url_model


router = APIRouter(prefix="/api", tags=["url"])


@router.post("/predict-url", response_model=UrlResponse)
def predict_url(request: UrlRequest) -> UrlResponse:
    processing_time = None

    """
    Predict whether a URL is phishing or legitimate using the trained URL model.
    """
    model_bundle = load_url_model()
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    start_time = time.time()
    feats_dict = extract_url_features(request.url)
    X = np.array([[feats_dict[name] for name in feature_names]])
    proba = model.predict_proba(X)[0, 1]
    label = "phishing" if proba >= 0.5 else "legitimate"

    end_time = time.time()
    processing_time = round(end_time - start_time, 4)
    
    important = url_feature_importance(model_bundle, feats_dict)


    # flatten important features into expected output shape
    simplified = {
        name: feats_dict[name]
        for name in feature_names
    }

    return UrlResponse(
        label=label,
        probability=float(proba),
        important_features=simplified,
        processing_time=f"{processing_time}s" if processing_time else None,
    )


