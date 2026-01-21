from typing import List, Dict, Any

import numpy as np


def top_tfidf_phishing_terms(
    model, vectorizer, text: str, top_k: int = 5
) -> List[str]:
    """
    Extract top suspicious TF-IDF features for a given text.
    
    For RandomForest models:
    - Uses TF-IDF weights of features present in the text
    - Combines with model's feature importances if available
    - Returns terms with highest TF-IDF values as suspicious indicators
    
    For Logistic Regression models:
    - Uses coefficient weights * TF-IDF values to find phishing-contributing terms
    
    Args:
        model: Trained classifier (RandomForest or LogisticRegression)
        vectorizer: Fitted TfidfVectorizer
        text: Input email text
        top_k: Number of top terms to return
        
    Returns:
        List of top suspicious feature terms
    """
    X = vectorizer.transform([text])
    if X.getnnz() == 0:  # Empty sparse matrix
        return []
    
    indices = X.nonzero()[1]
    values = X.data
    feature_names = vectorizer.get_feature_names_out()
    
    # For RandomForest: use TF-IDF weights + feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Combine TF-IDF value with feature importance
        scores = values * importances[indices]
        top_local_indices = np.argsort(-scores)[:top_k]
        top_feature_indices = indices[top_local_indices]
    # For Logistic Regression: use coefficients
    elif hasattr(model, "coef_"):
        coef = model.coef_[0]
        contrib = coef[indices] * values
        top_local_indices = np.argsort(-contrib)[:top_k]
        top_feature_indices = indices[top_local_indices]
    else:
        # Fallback: just use TF-IDF values
        top_local_indices = np.argsort(-values)[:top_k]
        top_feature_indices = indices[top_local_indices]
    
    return [feature_names[i] for i in top_feature_indices]


def url_feature_importance(
    model_bundle: Dict[str, Any], feature_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Given the RandomForest model bundle (with feature_names and model),
    return a mapping of feature_name -> value and importance.
    """
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    importances = getattr(model, "feature_importances_", None)

    important_features = {}
    for i, name in enumerate(feature_names):
        value = feature_values.get(name)
        importance = float(importances[i]) if importances is not None else None
        important_features[name] = {
            "value": value,
            "importance": importance,
        }
    return important_features


