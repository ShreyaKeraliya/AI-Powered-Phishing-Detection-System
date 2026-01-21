export default function Awareness() {
  return (
    <main className="container">
      <header className="hero">
        <h1>Phishing Awareness</h1>
        <p>
          Understanding how phishing works is the first step toward preventing
          it. Technology helps — but awareness protects.
        </p>
      </header>

      <section className="panel">
        <h2>What Is Phishing?</h2>
        <p className="panel-subtitle">
          A social engineering attack, not just a technical one.
        </p>

        <p>
          Phishing is a cyberattack where attackers impersonate trusted entities
          such as banks, payment apps, companies, or colleagues to trick users
          into revealing sensitive information.
        </p>

        <p>
          Unlike malware, phishing targets human psychology — urgency, fear,
          trust, and authority — making it highly effective even against
          technically aware users.
        </p>
      </section>

      <section className="cards" style={{ marginBottom: "1.2rem" }}>
        <div className="card">
          <h2>⚠️ Urgency & Fear</h2>
          <p>
            Messages like “Your account will be blocked” or “Immediate action
            required” pressure users into acting without thinking.
          </p>
        </div>

        <div className="card">
          <h2>🏦 Impersonation</h2>
          <p>
            Attackers pose as banks, UPI apps, HR departments, or government
            services to gain credibility.
          </p>
        </div>

        <div className="card">
          <h2>🔗 Malicious Links</h2>
          <p>
            Phishing emails often contain links that look legitimate but redirect
            to fake login pages or credential-harvesting sites.
          </p>
        </div>
      </section>

      <section className="panel" style={{ marginTop: "1.2rem" }}>
        <h2>What Can Phishing Lead To?</h2>
        <p className="panel-subtitle">
          One click is often enough to cause serious damage.
        </p>

        <ul style={{ paddingLeft: "1.2rem" }}>
          <li>Financial loss and unauthorized transactions</li>
          <li>Account takeover and identity theft</li>
          <li>Corporate data breaches and ransomware attacks</li>
          <li>Loss of trust in digital platforms</li>
        </ul>
      </section>

      <section className="panel" >
        <h2>Why Awareness Matters</h2>
        <p>
          No detection system is perfect. Attackers constantly adapt their
          language, tone, and strategies to bypass filters.
        </p>

        <p>
          This platform focuses on combining <strong>machine learning</strong>{" "}
          with <strong>user awareness</strong>, helping people understand not
          only <em>what</em> was detected, but <em>why</em> it was flagged.
        </p>
      </section>
    </main>
  )
}
