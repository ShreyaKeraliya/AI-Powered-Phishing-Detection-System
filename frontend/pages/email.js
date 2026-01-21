import Link from "next/link";
import EmailForm from "../components/EmailForm";

export default function EmailPage() {
  return (
    <main className="container">
      <nav className="top-nav">
        <Link href="/">← Dashboard</Link>
      </nav>
      <EmailForm />
      <section className="panel">
  <h2>Sample Phishing Email</h2>
  <p className="panel-subtitle">
    Common red flags found in real-world phishing attacks
  </p>

  <div className="result-card">
    <p><strong>Subject:</strong> Urgent: Your Account Will Be Suspended</p>
    <p style={{ marginTop: "0.6rem" }}>
      Dear Customer,<br />
      We detected unusual activity on your account. Please verify
      immediately to avoid suspension.
    </p>

    <ul style={{ marginTop: "0.8rem", paddingLeft: "1.2rem" }}>
      <li>Urgent language creating fear</li>
      <li>Generic greeting instead of your name</li>
      <li>Pressure to act immediately</li>
    </ul>
  </div>
</section>

    </main>
  );
}


