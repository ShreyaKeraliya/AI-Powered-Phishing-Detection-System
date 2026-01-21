export default function Footer() {
  return (
    <footer
      style={{
        marginTop: "4rem",
        padding: "2rem 0 1.5rem",
        // borderTop: "1px solid rgba(148,163,184,0.2)",
        fontSize: "0.85rem",
        opacity: 0.75,
      }}
    >
      <div className="container">
        <p>
          © {new Date().getFullYear()} Phishing Detection Platform - Built for cybersecurity
          awareness and research.
        </p>

        <p style={{ marginTop: "0.4rem" }}>
          This platform is intended for educational and awareness purposes only.
        </p>
      </div>
    </footer>
  );
}
