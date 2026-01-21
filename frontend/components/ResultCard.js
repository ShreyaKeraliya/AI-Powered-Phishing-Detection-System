import { highlightText } from "../utils/highlight";

export default function ResultCard({ result }) {
  if (!result) return null;

  // Determine badge color
  const riskColor = result.risk === "High Risk" ? "bg-red-500" : "bg-green-500";

  // Determine displayed label
  let displayRisk = result.risk;
  if (result.type === "url") {
    // Map email-style risk to URL-style labels
    displayRisk = result.risk === "High Risk" ? "Phishing" : "Legitimate";
  }



  // Safe defaults
  const reasons = result.reasons || [];
  const confidence = typeof result.confidence === "number" ? result.confidence : 0;
  const model = result.model || "N/A";
  const time = result.time || "N/A";
  const originalText = result.originalText || "";
  const type = result.type || "item";

  return (
    <div className="panel mt-6">
      <div className="flex-between">
        <h3>Security Analysis Report</h3>
        <span className={`badge ${riskColor}`}>{displayRisk || "Unknown Risk"}</span>
      </div>

      {/* Confidence */}
      <div className="mt-3">
        <p className="label">Detection Confidence :</p>
        <span className="value">{(confidence * 100).toFixed(1)}%</span>
        <div className="progress">
          <div
            className={`progress-bar ${riskColor}`}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Meta */}
      <div className="grid-2 mt-4">
        <div>
          <p className="label">Detection Engine :</p>
          <span>{model}</span>
        </div>
        <div>
          <p className="label">Processing Time</p>
          <span>{time}</span>
        </div>
      </div>

      {/* Highlighted Content */}
      {originalText && (
        <div className="mt-6">
          <p className="label">
            Analyzed {type === "url" ? "URL" : "Email"} Content
          </p>
          <div
            className="email-preview"
            dangerouslySetInnerHTML={{
              __html: highlightText(originalText, reasons),
            }}
          />
        </div>
      )}

      {/* Reasons */}
      {reasons.length > 0 && (
        <details className="mt-4">
          <summary>
            Why was this {type === "url" ? "URL" : "email"} flagged?
          </summary>
          <ul>
            {reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
