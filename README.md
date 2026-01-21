## AI-Powered Phishing Detection for Emails and URLs

End-to-end project combining **FastAPI**, **classical ML**, and **DistilBERT-based NLP** with a **Next.js frontend** to detect phishing in emails and URLs.

---

### Repository Layout

```text
.
├── backend/         # FastAPI backend + ML training scripts and models
└── frontend/        # Next.js frontend dashboard
```

See `backend/README.md` for detailed backend documentation.

---

### Architecture Overview

```text
[ User Browser ]
       |
       v
[ Next.js Frontend ]  <---->  [ FastAPI Backend ]
       |                          |
       |                          +--> Email TF-IDF + LR model
       |                          +--> Email DistilBERT model
       |                          +--> URL RandomForest + features
       v
[ Visual Results + Explanations ]
```

---

### Quick Start

1. **Backend**
   - Install Python 3.10+ and dependencies (`backend/requirements.txt`)
   - Train models using the scripts in `backend/train/` with your datasets
   - Run the FastAPI backend:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Frontend**
   - From the project root:

```bash
cd frontend
npm install
npm run dev
```

- Open `http://localhost:3000` in your browser.
- Optionally configure `NEXT_PUBLIC_BACKEND_URL` in `frontend/.env.local` if the backend is not at `http://localhost:8000`.

---

### Screens & Features

- **Dashboard (`/`)**
  - Overview with navigation to email and URL analyzers.

- **Email Phishing Checker (`/email`)**
  - Inputs:
    - Subject
    - Body
    - Model selector: `TF-IDF + Logistic Regression` or `DistilBERT`
  - Output:
    - Label: `phishing` / `legitimate`
    - Probability
    - Highlighted suspicious phrases (keywords)

- **URL Phishing Analyzer (`/url`)**
  - Input:
    - URL string
  - Output:
    - Label: `phishing` / `legitimate`
    - Probability
    - Key engineered features (URL length, subdomains, IP presence, etc.)

---

### API Summary

- `POST /api/predict-email`
- `POST /api/predict-url`

For full request/response examples and training instructions, see `backend/README.md`.

---

### Screenshot Placeholders

- `docs/screenshot-dashboard.png` — Dashboard view
- `docs/screenshot-email.png` — Email phishing detection results
- `docs/screenshot-url.png` — URL phishing analysis results

You can capture your own screenshots and store them under `docs/` using these filenames.


"# AI-Powered-Phishing-Detection-System" 
