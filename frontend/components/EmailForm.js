import { useState } from "react";
import ResultCard from "./ResultCard";

const API_BASE =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function EmailForm() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [modelType, setModelType] = useState("tfidf_rf");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!body.trim()) {
      setError("Email body is required.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/predict-email`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          subject,
          body,
          model_type: modelType,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Prediction failed.");
      }

      const data = await res.json();
      // Debug logging for response structure
      console.log("API Response:", JSON.stringify(data, null, 2));
      
      // Ensure we're setting the result object correctly
      if (data && typeof data === 'object') {
        setResult({
          risk: data.label === "phishing" ? "High Risk" : "Low Risk",
          confidence: data.probability ?? 0,
          model: data.model_used ?? "ML Detection Engine",
          time: data.processing_time ?? "—",
          reasons: data.explanations ?? [],
          originalText: body, // 🔥 REQUIRED
        });
      } else {
        setResult(data);
      }
    } catch (err) {
      setError(err.message || "Unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <form className="panel" onSubmit={handleSubmit}>
        <h2>Email Phishing Checker</h2>
        <p className="panel-subtitle">
          Analyze email content for phishing indicators using ML models.
        </p>

        <label className="field">
          <span>Subject (optional)</span>
          <input
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="Account Verification Required"
          />
        </label>

        <label className="field">
          <span>Body</span>
          <textarea
            rows={8}
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder="Paste the full email content here..."
          />
        </label>

        <label className="field">
          <span>Model</span>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
          >
            <option value="tfidf_rf">TF-IDF + RandomForest (Adversarial)</option>
            <option value="distilbert">DistilBERT (Transformer NLP)</option>
          </select>
        </label>

        {error && <p className="error">{error}</p>}

        <button type="submit" className="btn" disabled={loading}>
          {loading ? "Analyzing..." : "Scan Email"}
        </button>
      </form>

      <ResultCard result={result} type="email" />
    </>
  );
}


