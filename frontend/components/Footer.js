export default function Footer() {
  return (
    <footer
      style={{
        marginTop: "4rem",
        padding: "2.5rem 0 1.5rem",
        fontSize: "0.85rem",
        opacity: 0.8,
      }}
    >
      <div className="container">
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.6rem",
          }}
        >
          <p>
            © {new Date().getFullYear()} Phishing Detection Platform — A learning-driven
            cybersecurity project.
          </p>

          <p>
            Designed & developed by <strong>Shreya Keraliya</strong> as part of an
            exploration into ML-based phishing detection and security awareness.
          </p>

          <div
            style={{
              display: "flex",
              gap: "1.2rem",
              flexWrap: "wrap",
              marginTop: "0.4rem",
            }}
          >
            <a
              href="https://github.com/ShreyaKeraliya"
              target="_blank"
              rel="noopener noreferrer"
              style={{ textDecoration: "underline" }}
            >
              GitHub
            </a>

            <a
              href="https://www.linkedin.com/in/shreya-keraliya-5737b3279/"
              target="_blank"
              rel="noopener noreferrer"
              style={{ textDecoration: "underline" }}
            >
              LinkedIn
            </a>

            <a
              href="/about"
              style={{ textDecoration: "underline" }}
            >
              About Project
            </a>
          </div>

          <p style={{ marginTop: "0.6rem", opacity: 0.7 }}>
            This platform is intended strictly for educational and awareness
            purposes. No real-world misuse is encouraged or supported.
          </p>
        </div>
      </div>
    </footer>
  );
}
