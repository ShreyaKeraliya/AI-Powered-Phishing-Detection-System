## AI-Powered Phishing Detection Backend (FastAPI + ML)

This backend provides REST APIs for detecting phishing in **emails** and **URLs** using:

- **Classical ML**: TF-IDF + Logistic Regression (emails) and RandomForest (URLs)
- **Transformer NLP**: DistilBERT fine-tuned for phishing vs legitimate emails

It is designed to be used by the Next.js frontend in this project or any other client.

---

### Project Structure

```text
backend/
 ├── app/
 │   ├── main.py                    # FastAPI application entrypoint
 │   ├── api/
 │   │   ├── email_routes.py        # /api/predict-email endpoint
 │   │   ├── url_routes.py          # /api/predict-url endpoint
 │   ├── models/
 │   │   ├── email_tfidf_lr.pkl     # Saved TF-IDF + LR model + vectorizer (created at train time)
 │   │   ├── url_rf.pkl             # Saved URL RandomForest bundle (created at train time)
 │   │   ├── distilbert_email/      # DistilBERT model + tokenizer directory (created at train time)
 │   ├── services/
 │   │   ├── email_preprocessing.py # Email cleaning pipeline
 │   │   ├── url_feature_extraction.py # URL feature engineering
 │   │   ├── explainability.py      # Simple explainability helpers
 │   ├── schemas/
 │   │   ├── email_schema.py        # Pydantic models for email API
 │   │   ├── url_schema.py          # Pydantic models for URL API
 │   ├── config.py                  # Settings & paths (uses .env if present)
 │   └── utils.py                   # Model loading helpers
 ├── train/
 │   ├── train_email_tfidf.py       # Train TF-IDF + LR email model
 │   ├── train_email_distilbert.py  # Fine-tune DistilBERT email model
 │   ├── train_url_model.py         # Train URL RandomForest model
 ├── requirements.txt               # Backend dependencies
 └── Dockerfile                     # Optional Docker image for backend
```

---

### High-Level Architecture (ASCII)

```text
[ Next.js Frontend ]
     |    \
     |     \ (HTTP POST JSON)
     v      v
 [ FastAPI Backend ]
  /api/predict-email   /api/predict-url
        |                    |
        |                    |
  [Email Models]        [URL Models]
   - TF-IDF + LR         - RandomForest (primary)
   - DistilBERT          - Logistic Regression (baseline)
        |                    |
  [Explainability]      [Feature Importance]
```

---

### Dataset Expectations

You must download and preprocess datasets yourself (for licensing reasons). The training scripts expect:

- **Email (Enron / similar spam dataset)**
  - CSV file (e.g. `data/emails_enron.csv`) with columns:
    - `subject` (string, can be empty)
    - `body` (string, required)
    - `label` (int, 1 = phishing/spam, 0 = legitimate/ham)

- **URL (Kaggle phishing URL dataset or similar)**
  - CSV file (e.g. `data/phishing_urls.csv`) with columns:
    - `url` (string)
    - `label` (int, 1 = phishing, 0 = legitimate)

You can override default paths via environment variables (see below).

---

### Environment Variables

Backend configuration is managed via `app.config.Settings` and `.env` (optional):

- **CORS / general**
  - `BACKEND_CORS_ORIGINS` is not strictly required; by default, `["*"]` is used for development.

- **Training data locations**
  - `EMAIL_TFIDF_DATA` (optional)  
    Path to CSV for TF-IDF + LR training.  
    Default: `../data/emails_enron.csv` (relative to `backend/`).
  - `EMAIL_DISTILBERT_DATA` (optional)  
    Path to CSV for DistilBERT fine-tuning.  
    Default: `../data/emails_enron.csv`.
  - `URL_DATA` (optional)  
    Path to CSV for URL model training.  
    Default: `../data/phishing_urls.csv`.

---

### Setup & Installation

1. **Create and activate a virtualenv (recommended)**

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac
```

2. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Training the Models

Ensure your datasets exist at the expected paths (or set the env vars).

#### 1. Train TF-IDF + Logistic Regression Email Model

```bash
cd backend
set EMAIL_TFIDF_DATA=..\data\emails_enron.csv  # Windows (PowerShell: $env:EMAIL_TFIDF_DATA="..\data\emails_enron.csv")
python -m train.train_email_tfidf
```

This will:

- Clean emails using `email_preprocessing.clean_email_text`
- Fit a TF-IDF vectorizer (unigram + bigram, `max_features=20000`)
- Train a Logistic Regression classifier
- Print a classification report
- Save model and vectorizer to `app/models/email_tfidf_lr.pkl`

#### 2. Fine-tune DistilBERT Email Model

```bash
cd backend
set EMAIL_DISTILBERT_DATA=..\data\emails_enron.csv  # or similar path
python -m train.train_email_distilbert
```

This will:

- Clean emails
- Tokenize with `DistilBertTokenizerFast` (`max_length=256`)
- Train `DistilBertForSequenceClassification` for 2 labels
- Use `AdamW` + linear warmup scheduler
- Run for 3 epochs with a modest batch size (suitable for CPU/GPU)
- Save the model and tokenizer to `app/models/distilbert_email/`

#### 3. Train URL RandomForest Model

```bash
cd backend
set URL_DATA=..\data\phishing_urls.csv
python -m train.train_url_model
```

This will:

- Compute features:
  - `url_length`
  - `subdomains`
  - `has_ip`
  - `has_at`
  - `uses_https`
  - `dash_count`
- Train:
  - Logistic Regression (baseline)
  - RandomForest (primary model)
- Print performance for both
- Save a bundle (RF + LR + feature_names) to `app/models/url_rf.pkl`

---

### Running the FastAPI Backend

After training, ensure the model files exist:

- `app/models/email_tfidf_lr.pkl`
- `app/models/url_rf.pkl`
- `app/models/distilbert_email/` (contains `config.json`, `pytorch_model.bin`, `tokenizer.json`, etc.)

Then start the API:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

---

### API Endpoints

#### `POST /api/predict-email`

**Request JSON**

```json
{
  "subject": "Your account has been suspended",
  "body": "Dear user, please click the link below to verify your account...",
  "model_type": "tfidf_lr"
}
```

`model_type` can be:

- `"tfidf_lr"` — TF-IDF + Logistic Regression (fast baseline, explainable weights)
- `"distilbert"` — DistilBERT transformer (deeper NLP understanding)

**Response JSON**

```json
{
  "label": "phishing",
  "probability": 0.93,
  "model_used": "tfidf_lr",
  "explanations": ["verify account", "urgent", "click link"]
}
```

- `label`: `"phishing"` or `"legitimate"`
- `probability`: model confidence for the phishing class
- `explanations`: top phishing-weighted keywords from the TF-IDF model

For DistilBERT, the backend still uses the TF-IDF model to derive interpretable keyword explanations.

#### `POST /api/predict-url`

**Request JSON**

```json
{
  "url": "http://192.168.0.10/secure-login/verify"
}
```

**Response JSON**

```json
{
  "label": "phishing",
  "probability": 0.88,
  "important_features": {
    "url_length": 85,
    "subdomains": 3,
    "has_ip": 1,
    "has_at": 0,
    "uses_https": 0,
    "dash_count": 4
  }
}
```

- `important_features` mirrors the engineered feature values that influenced the prediction.

---

### Explainability Details

- **Emails (TF-IDF + LR)**
  - `explainability.top_tfidf_phishing_terms`:
    - Computes the TF-IDF vector for the given email.
    - Multiplies each feature’s TF-IDF value by the LR coefficient for the phishing class.
    - Returns the top few tokens with highest positive contribution.

- **URLs**
  - `explainability.url_feature_importance`:
    - Wraps RandomForest’s `feature_importances_` along with actual feature values.
    - The `url_routes` API returns a simplified mapping of feature names to their values, and can be extended to surface raw importances if desired.

---

### Docker Usage (Optional)

A simple Dockerfile is provided under `backend/Dockerfile`.

Build the image:

```bash
cd backend
docker build -t phishing-backend .
```

Run the container (assuming models are already baked into the image or mounted):

```bash
docker run -p 8000:8000 phishing-backend
```

For a production deployment, consider:

- Using a process manager (e.g. `gunicorn` with `uvicorn.workers.UvicornWorker`)
- Health checks and logging
- Mounting a volume for model files if retraining in-place

---

### Frontend Integration

The Next.js frontend (in the `frontend/` directory) is configured to call this backend at:

- `http://localhost:8000/api/predict-email`
- `http://localhost:8000/api/predict-url`

You can override the backend URL from the frontend by setting:

- `NEXT_PUBLIC_BACKEND_URL=http://your-backend-host:8000`

in `frontend/.env.local`.


