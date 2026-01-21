"""
Train URL phishing detection models (Logistic Regression + Random Forest).

This script is designed to be **robust** to real-world datasets:

- Supports both CSV (`.csv`) and Excel (`.xlsx`)
- Resolves dataset paths relative to the project root
- Automatically detects URL and label columns with flexible naming
- Engineers lexical URL features when only raw URLs are given
- Prints clear, human-readable errors instead of crashing with stack traces
"""

import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.services.url_feature_extraction import extract_url_features
from app.config import get_settings


FEATURE_NAMES = [
    "url_length",
    "subdomains",
    "has_ip",
    "has_at",
    "uses_https",
    "dash_count",
    "digit_count",
    "query_count",
    "suspicious_word_count",
    "is_shortened",
    "url_entropy",
    "domain_length",
]




TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TRAIN_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)


def resolve_dataset_path(env_var: str, default_rel: str, description: str) -> str:
    """
    Resolve dataset path relative to the project root, honoring an optional env var.
    Prints clear guidance if the file is missing.
    """
    env_value = os.environ.get(env_var)
    if env_value:
        candidate = (
            env_value
            if os.path.isabs(env_value)
            else os.path.join(PROJECT_ROOT, env_value)
        )
    else:
        candidate = os.path.join(PROJECT_ROOT, default_rel)

    abs_path = os.path.abspath(candidate)
    print(f"Loading {description} dataset from: {abs_path}")

    if not os.path.exists(abs_path):
        print(f"Dataset not found. Please place the file at:\n  {abs_path}")
        print(
            f"Or set the environment variable {env_var} to point to your dataset "
            "(absolute path, or path relative to the project root)."
        )
        raise SystemExit(1)

    return abs_path


def load_tabular_file(path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame with helpful error messages.
    """
    _, ext = os.path.splitext(path)
    try:
        if ext.lower() == ".xlsx":
            df = pd.read_excel(path)
        elif ext.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            print(
                f"Unrecognized file extension '{ext}' for dataset. "
                "Attempting to read as CSV."
            )
            df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Dataset not found at: {path}")
        print("Please ensure the file exists at this location.")
        raise SystemExit(1)
    except Exception as e:
        print(f"Failed to read dataset at {path}: {e}")
        raise SystemExit(1)

    if df.empty:
        print(f"Dataset at {path} is empty. Please provide a non-empty file.")
        raise SystemExit(1)

    print(
        f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns. "
        f"Columns: {', '.join(map(str, df.columns))}"
    )
    return df


def _normalize_columns(df: pd.DataFrame) -> dict:
    return {col: str(col).strip().lower() for col in df.columns}


def _detect_url_and_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to automatically detect URL and label columns using flexible,
    substring-based matching on normalized column names.
    """
    norm = _normalize_columns(df)

    url_tokens = ["url", "link", "website", "domain"]
    label_tokens = ["label", "result", "phish", "target", "class", "status"]

    url_col = None
    label_col = None

    for original, lowered in norm.items():
        if url_col is None and any(t in lowered for t in url_tokens):
            url_col = original
        if label_col is None and any(t in lowered for t in label_tokens):
            label_col = original

    if url_col is None:
        print(
            "Could not automatically detect a URL column.\n"
            "Looked for columns containing any of: "
            f"{', '.join(url_tokens)}\n"
            f"Available columns: {list(df.columns)}"
        )
        raise SystemExit(1)

    if label_col is None:
        print(
            "Could not automatically detect a label column for phishing.\n"
            "Looked for columns containing any of: "
            f"{', '.join(label_tokens)}\n"
            f"Available columns: {list(df.columns)}"
        )
        raise SystemExit(1)

    return url_col, label_col


def _map_url_label(raw: object) -> int | None:
    """
    Map a raw label value to binary:
    - phishing/malicious-like values -> 1
    - benign/legitimate-like values -> 0
    Returns None if the value cannot be interpreted.
    """
    if pd.isna(raw):
        return None

    s = str(raw).strip().lower()

    positive = {
        "1",
        "true",
        "phish",
        "phishing",
        "malicious",
        "bad",
        "fraud",
        "scam",
        "spam",
        "phishing website",
    }
    negative = {
        "0",
        "false",
        "benign",
        "legit",
        "legitimate",
        "good",
        "normal",
        "safe",
        "clean",
    }

    if s in positive:
        return 1
    if s in negative:
        return 0

    try:
        num = float(s)
        if num > 0:
            return 1
        if num <= 0:
            return 0
    except Exception:
        pass

    return None


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and normalize a URL dataset from CSV/Excel into:
    - X: feature matrix from engineered URL features
    - y: binary numpy array (1 = phishing, 0 = legitimate)
    """
    df = load_tabular_file(path)
    url_col, label_col = _detect_url_and_label_columns(df)

    feature_rows: List[List[float]] = []
    labels: List[int] = []
    skipped_for_label = 0
    skipped_for_url = 0

    for _, row in df.iterrows():
        url_val = row[url_col]
        if pd.isna(url_val) or not str(url_val).strip():
            skipped_for_url += 1
            continue

        label_raw = row[label_col]
        mapped = _map_url_label(label_raw)
        if mapped is None:
            skipped_for_label += 1
            continue

        feats = extract_url_features(str(url_val))
        feature_rows.append([feats[name] for name in FEATURE_NAMES])
        labels.append(mapped)

    if not feature_rows:
        print(
            "After cleaning and label mapping, no valid URL training examples remain.\n"
            f"Rows skipped due to missing/empty URL: {skipped_for_url}\n"
            f"Rows skipped due to unrecognized labels: {skipped_for_label}"
        )
        raise SystemExit(1)

    X = np.array(feature_rows, dtype=float)
    y = np.array(labels, dtype=int)

    print(
        f"Prepared {len(labels)} URL examples "
        f"(skipped {skipped_for_url} for empty URL, "
        f"{skipped_for_label} for bad labels)."
    )
    return X, y


def main():
    settings = get_settings()
    data_path = resolve_dataset_path(
        env_var="URL_DATA",
        default_rel=os.path.join("data", "phishing_urls.csv"),
        description="URL",
    )
    X, y = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Logistic Regression baseline...")
    lr = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print("Logistic Regression performance:")
    print(classification_report(y_test, lr_pred, digits=4))

    print("Training RandomForest primary model...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("RandomForest performance:")
    print(classification_report(y_test, rf_pred, digits=4))

    # Save primary RF model and metadata
    os.makedirs(os.path.dirname(settings.url_rf_path), exist_ok=True)
    bundle = {
        "model": rf,
        "feature_names": FEATURE_NAMES,
        "baseline_lr": lr,
    }
    joblib.dump(bundle, settings.url_rf_path)
    print(f"Saved URL RandomForest model bundle to {settings.url_rf_path}")


if __name__ == "__main__":
    main()


