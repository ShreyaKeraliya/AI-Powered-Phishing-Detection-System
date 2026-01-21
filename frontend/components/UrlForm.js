import { useState } from "react";
import ResultCard from "./ResultCard";

const API_BASE =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function UrlForm() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!url.trim()) {
      setError("URL is required.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/predict-url`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Prediction failed.");
      }

      const data = await res.json();

    // Map URL backend label to match email convention
    let normalizedRisk;
    if (data.label.toLowerCase() === "phishing") {
      normalizedRisk = "High Risk";
    } else {
      normalizedRisk = "Low Risk";
    }

    const normalizedResult = {
      type: "url",
      risk: normalizedRisk, // now compatible with ResultCard
      confidence: typeof data.probability === "number" ? data.probability : 0,
      model: data.model || "RandomForest URL Detector",
      time: data.process_time || "N/A",
      originalText: url,
      reasons: data.important_features
        ? Object.entries(data.important_features).map(
            ([feature, value]) => `${feature}: ${value}`
          )
        : [],
    };



      setResult(normalizedResult);
    } catch (err) {
      setError(err.message || "Unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <form className="panel" onSubmit={handleSubmit}>
        <h2>URL Phishing Analyzer</h2>
        <p className="panel-subtitle">
          Inspect URLs for phishing using lexical features and RandomForest.
        </p>

        <label className="field">
          <span>URL</span>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://secure-login.example.com/verify"
          />
        </label>

        {error && <p className="error">{error}</p>}

        <button type="submit" className="btn" disabled={loading}>
          {loading ? "Analyzing..." : "Scan URL"}
        </button>
      </form>

      <ResultCard result={result} />
    </>
  );
}
