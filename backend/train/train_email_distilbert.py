"""
Fine-tune DistilBERT for email phishing detection.

This script is designed to be **robust** to real-world datasets:

- Supports both CSV (`.csv`) and Excel (`.xlsx`)
- Resolves dataset paths relative to the project root
- Automatically detects subject/body/label columns with flexible naming
- Converts spam/ham-like labels to binary 1/0
- Prints clear, human-readable errors instead of crashing with stack traces

It uses HuggingFace Transformers and PyTorch for model fine-tuning.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

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

    try:
        num = float(s)
        if num > 0:
            return 1
        if num <= 0:
            return 0
    except Exception:
        pass

    return None


def load_dataset(path: str):
    """
    Load and normalize an email dataset from CSV/Excel into:
    - texts: preprocessed combined subject+body
    - labels: binary list (1 = phishing, 0 = legitimate)
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
    return texts, labels


class EmailDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    epochs: int = 3
    batch_size: int = 8              # CPU-friendly default
    cuda_batch_size: int = 16        # larger batch if GPU is available
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_length: int = 128            # shorter sequences for speed
    max_steps: int = 2000            # early stop after this many steps
    save_steps: int = 500            # save every N steps


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_model_and_tokenizer(model, tokenizer, out_dir: str, label: str):
    _ensure_dir(out_dir)
    print(f"[{label}] Saving model and tokenizer to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

def save_final_model(model, tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving DistilBERT model to: {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def _evaluate_loss(model, dataloader, device) -> float:
    """Lightweight eval on a small subset to monitor overfitting."""
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)


def train():
    cfg = TrainConfig()
    settings = get_settings()
    data_path = resolve_dataset_path(
        env_var="EMAIL_DISTILBERT_DATA",
        default_rel=os.path.join("data", "phishing_email.csv"),
        description="email (DistilBERT)",
    )
    texts, labels = load_dataset(data_path)

    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.model_name)
    dataset = EmailDataset(texts, labels, tokenizer, max_length=cfg.max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = cfg.cuda_batch_size if device.type == "cuda" else cfg.batch_size
    print(f"Using device: {device} | batch_size={batch_size} | max_length={cfg.max_length}")

    model = DistilBertForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
    )
    model.to(device)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    planned_steps = len(dataloader) * cfg.epochs
    total_steps = min(planned_steps, cfg.max_steps)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Prepare eval subset (small) for quick monitoring
    eval_cap = min(len(dataset), max(batch_size * 2, 32))
    eval_subset = Subset(dataset, list(range(eval_cap)))
    eval_loader = DataLoader(eval_subset, batch_size=min(batch_size, 16))

    out_dir = os.path.join(BACKEND_DIR, "models", "distilbert_email")
    _ensure_dir(out_dir)
    print(f"Checkpoints will be saved under: {out_dir}")

    model.train()
    global_step = 0

    print("Starting training...")

    try:
        for epoch in range(cfg.epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 100 == 0:
                    print(f"Step {global_step}/{cfg.max_steps} - loss={loss.item():.4f}")

                # 💾 Save every 500 steps
                if global_step % cfg.save_steps == 0:
                    save_final_model(model, tokenizer, out_dir)

                # ⛔ Early stop
                if global_step >= cfg.max_steps:
                    print(f"Early stopping at step {global_step}")
                    save_final_model(model, tokenizer, out_dir)
                    return

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed | avg_loss={avg_loss:.4f}")
            save_final_model(model, tokenizer, out_dir)

    finally:
        # ✅ GUARANTEED SAVE
        print("Final save (safety net)")
        save_final_model(model, tokenizer, out_dir)

        # Final save
    _save_model_and_tokenizer(model, tokenizer, out_dir, label="final")
    print("Training complete.")


if __name__ == "__main__":
    train()


