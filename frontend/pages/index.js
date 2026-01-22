import Link from "next/link";

export default function Home() {
  return ( 
    <main className="container animated-bg ">
      {/* HERO SECTION */}
      <header className="hero">
        <h1>AI-Powered Phishing Detection</h1>
        <p>
          Protect users from phishing attacks using machine learning and
          transformer-based NLP — built with a cybersecurity-first mindset.
        </p>

        <p
          style={{
            marginTop: "0.8rem",
            fontSize: "0.9rem",
            opacity: 0.8,
          }}
        >
          ⚠️ Built for awareness, research, and educational use
        </p>
      </header>

      {/* WHY THIS MATTERS */}
      <section className="panel">
        <h2>Why Phishing Detection Matters</h2>
        <p className="panel-subtitle">
          Phishing is one of the most common and effective cyberattacks today.
        </p>

        <p>
          Attackers exploit human trust by impersonating banks, payment apps,
          companies, and even colleagues. A single phishing email can bypass
          traditional security controls and lead to serious consequences.
        </p>
      </section>

      {/* PROBLEMS CAUSED */} 
      <section className="cards" style={{ marginBottom: "1.2rem" }}>
        <div className="card">
          <h2>💸 Financial Loss</h2>
          <p>
            Phishing emails often trick users into sharing banking details,
            OTPs, or payment credentials, leading to direct financial theft.
          </p>
        </div>

        <div className="card">
          <h2>🪪 Identity Theft</h2>
          <p>
            Stolen credentials can be reused across platforms, allowing
            attackers to impersonate victims and access multiple services.
          </p>
        </div>

        <div className="card">
          <h2>🏢 Organizational Breaches</h2>
          <p>
            In corporate environments, phishing is a primary entry point for
            ransomware, data leaks, and large-scale security incidents.
          </p>
        </div>
      </section>

      {/* AWARENESS SECTION */}
      <section className="panel">
        <h2>Awareness Is the First Line of Defense</h2>
        <p className="panel-subtitle">
          Technology alone cannot stop phishing - understanding the attack
          patterns is equally important.
        </p>

        <p>
          This platform not only detects phishing attempts using machine
          learning, but also helps users understand:
        </p>

        <ul style={{ marginTop: "0.8rem", paddingLeft: "1.2rem" }}>
          <li>How attackers use urgency, fear, and authority</li>
          <li>Which words and patterns raise red flags</li>
          <li>Why some phishing emails look completely legitimate</li>
        </ul>
      </section>

      {/* DETECTION TOOLS */}
      <section className="cards" style={{ marginBottom: "1.2rem" }}>
        <Link href="/email" className="card card-link">
          <h2>Email Phishing Detector</h2>
          <p>
            Analyze suspicious email content using NLP-based machine learning
            models such as TF-IDF and DistilBERT.
          </p>
        
          <p className="card-meta">
            Focus: language patterns, social engineering, intent detection
          </p>
        
          <button className="card-btn">
            Analyze Email →
          </button>
        </Link>


        <Link href="/url" className="card card-link">
          <h2>URL Phishing Analyzer</h2>
          <p>
            Inspect URLs using lexical and structural features with a
            Random-Forest-based model.
          </p>
        
          <p className="card-meta">
            Focus: interpretable features and explainable predictions
          </p>
        
          <button className="card-btn">
            Analyze URL →
          </button>
        </Link>

      </section>

      {/* CLOSING TRUST SECTION */}
      <section className="panel">
        <h2>Built with Security in Mind</h2>
        <p>
          This project combines <strong>Machine Learning</strong> with
          <strong> Cyber Security </strong>, focusing not
          only on prediction accuracy but also on explainability, awareness, and
          real-world attack understanding.
        </p>
      </section>

    </main>
  );
}
