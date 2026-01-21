import { useState } from "react";

export function DetectionResult({ result }) {
  const [showDetails, setShowDetails] = useState(false);

  const riskColor =
    result.label === "High Risk"
      ? "bg-red-500"
      : result.label === "Suspicious"
      ? "bg-yellow-400"
      : "bg-green-500";

  return (
    <div className="mt-6 rounded-xl border border-slate-700 bg-slate-900 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Threat Assessment Report</h3>
        <span
          className={`px-3 py-1 rounded-full text-xs font-semibold text-black ${riskColor}`}
        >
          {result.label}
        </span>
      </div>

      {/* Confidence */}
      <div className="mb-4">
        <p className="text-sm text-slate-400 mb-1">
          Detection Confidence
        </p>
        <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${riskColor}`}
            style={{ width: `${result.confidence * 100}%` }}
          />
        </div>
        <p className="text-xs text-slate-400 mt-1">
          High confidence based on learned phishing patterns
        </p>
      </div>

      {/* Meta */}
      <div className="grid grid-cols-2 gap-4 text-sm mb-4">
        <div>
          <p className="text-slate-400">Detection Engine</p>
          <p>{result.model}</p>
        </div>
        <div>
          <p className="text-slate-400">Analysis Time</p>
          <p>{result.time}</p>
        </div>
      </div>

      {/* Toggle */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="text-sm text-sky-400 hover:underline"
      >
        {showDetails ? "Hide explanation" : "Why was this email flagged?"}
      </button>

      {/* Hidden details */}
      {showDetails && (
        <div className="mt-4 border-t border-slate-700 pt-4">
          <ul className="list-disc list-inside text-sm text-slate-300 space-y-1">
            {result.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
