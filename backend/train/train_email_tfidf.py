"""
Train a TF-IDF + Logistic Regression model for email phishing detection.

This script is designed to be **robust** to real-world datasets:

- Supports both CSV (`.csv`) and Excel (`.xlsx`)
- Resolves dataset paths relative to the project root
- Automatically detects subject/body/label columns with flexible naming
- Converts spam/ham-like labels to binary 1/0
- Prints clear, human-readable errors instead of crashing with stack traces
"""

import os
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from app.services.email_preprocessing import clean_email_text
from app.config import get_settings


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


def _detect_email_columns(df: pd.DataFrame) -> Tuple[str | None, str, str]:
    """
    Try to automatically detect subject, body, and label columns using
    flexible, substring-based matching on normalized column names.
    """
    norm = _normalize_columns(df)

    subject_tokens = ["subject", "subj", "email_subject", "title"]
    body_tokens = ["body", "message", "content", "text", "email", "email_body"]
    label_tokens = ["label", "spam", "phish", "target", "category", "is_spam"]

    subject_col = None
    body_col = None
    label_col = None

    for original, lowered in norm.items():
        if subject_col is None and any(t in lowered for t in subject_tokens):
            subject_col = original
        if body_col is None and any(t in lowered for t in body_tokens):
            body_col = original
        if label_col is None and any(t in lowered for t in label_tokens):
            label_col = original

    if body_col is None:
        print(
            "Could not automatically detect an email body/message column.\n"
            "Looked for columns containing any of: "
            f"{', '.join(body_tokens)}\n"
            f"Available columns: {list(df.columns)}"
        )
        raise SystemExit(1)

    if label_col is None:
        print(
            "Could not automatically detect a label column for spam/phishing.\n"
            "Looked for columns containing any of: "
            f"{', '.join(label_tokens)}\n"
            f"Available columns: {list(df.columns)}"
        )
        raise SystemExit(1)

    return subject_col, body_col, label_col


def _map_email_label(raw: object) -> int | None:
    """
    Map a raw label value to binary:
    - spam/phishing-like values -> 1
    - ham/legitimate-like values -> 0
    Returns None if the value cannot be interpreted.
    """
    if pd.isna(raw):
        return None

    s = str(raw).strip().lower()

    positive = {
        "1",
        "true",
        "spam",
        "phish",
        "phishing",
        "malicious",
        "fraud",
        "scam",
        "bad",
        "spam_mail",
    }
    negative = {
        "0",
        "false",
        "ham",
        "legit",
        "legitimate",
        "benign",
        "not spam",
        "normal",
        "good",
        "clean",
    }

    if s in positive:
        return 1
    if s in negative:
        return 0

    # Some datasets use 0/1, -1/1, etc.
    try:
        num = float(s)
        if num > 0:
            return 1
        if num <= 0:
            return 0
    except Exception:
        pass

    return None


def load_dataset(path: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load and normalize an email dataset from CSV/Excel into:
    - texts: preprocessed combined subject+body
    - labels: binary pandas Series (1 = phishing, 0 = legitimate)
    """
    df = load_tabular_file(path)
    subject_col, body_col, label_col = _detect_email_columns(df)

    texts: List[str] = []
    labels: List[int] = []
    skipped_for_label = 0
    skipped_for_body = 0

    for _, row in df.iterrows():
        subject_val = str(row[subject_col]) if subject_col is not None else ""
        body_val = row[body_col]

        if pd.isna(body_val) or not str(body_val).strip():
            skipped_for_body += 1
            continue

        label_raw = row[label_col]
        mapped = _map_email_label(label_raw)
        if mapped is None:
            skipped_for_label += 1
            continue

        cleaned = clean_email_text(subject_val, str(body_val))
        if not cleaned:
            skipped_for_body += 1
            continue

        texts.append(cleaned)
        labels.append(mapped)

    if not texts:
        print(
            "After cleaning and label mapping, no valid training examples remain.\n"
            f"Rows skipped due to missing/empty body: {skipped_for_body}\n"
            f"Rows skipped due to unrecognized labels: {skipped_for_label}"
        )
        raise SystemExit(1)

    print(
        f"Prepared {len(texts)} email examples "
        f"(skipped {skipped_for_body} for empty body, "
        f"{skipped_for_label} for bad labels)."
    )
    return pd.Series(texts), pd.Series(labels, dtype=int)


def train_tfidf_lr(
    texts: pd.Series,
    labels: pd.Series,
) -> Tuple[TfidfVectorizer, LogisticRegression]:

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,      # ↓ from 20000
        min_df=5,                # ignore rare words
        max_df=0.9,              # ignore extremely common words
        stop_words="english",
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1000,
        C=0.5,                   # 🔥 stronger regularization
        solver="liblinear",
        class_weight="balanced",
    )

    model.fit(X, labels)
    return vectorizer, model


def main():
    settings = get_settings()
    data_path = resolve_dataset_path(
        env_var="EMAIL_TFIDF_DATA",
        default_rel=os.path.join("data", "phishing_email.csv"),
        description="email (TF-IDF)",
    )
    texts, labels = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Training TF-IDF + Logistic Regression model (fit ONLY on training set)...")
    vectorizer, model = train_tfidf_lr(X_train, y_train)

    # Evaluation
    print("Evaluating model...")

    # 🔹 Training accuracy
    X_train_vec = vectorizer.transform(X_train)
    train_pred = model.predict(X_train_vec)
    train_acc = accuracy_score(y_train, train_pred)

    # 🔹 Test accuracy
    X_test_vec = vectorizer.transform(X_test)
    test_pred = model.predict(X_test_vec)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    acc_gap = abs(train_acc - test_acc)
    if acc_gap < 0.005:
        print(
            "⚠️ Warning: Train/Test accuracy difference < 0.5%. "
            "Double-check for data leakage, duplicate rows, or label issues."
        )

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred, digits=4))


    # Save model and vectorizer
    os.makedirs(os.path.dirname(settings.email_tfidf_rf_path), exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "model": model},
        settings.email_tfidf_rf_path,
    )
    print(f"Saved TF-IDF RandomForest model to {settings.email_tfidf_rf_path}")


if __name__ == "__main__":
    main()


