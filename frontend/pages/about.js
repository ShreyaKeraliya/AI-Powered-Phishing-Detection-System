export default function About() {
  return (
    <main className="container">
      <header className="hero">
        <h1>About This Project</h1>
        <p>
          A learning-driven attempt to bridge machine learning with real-world
          cybersecurity problems.
        </p>
      </header>

      <section className="panel">
        <h2>Project Motivation</h2>
        <p className="panel-subtitle">
          Why phishing detection - and why machine learning?
        </p>

        <p>
          Phishing remains one of the most successful cyberattacks because it
          targets people rather than systems. While many security tools focus on
          infrastructure-level defenses, phishing often bypasses them entirely.
        </p>

        <p>
          This project was built to explore how machine learning and NLP can help
          detect phishing attempts by analyzing language patterns, intent, and
          social engineering techniques.
        </p>
      </section>

      <section className="panel">
        <h2>Learning Journey</h2>
        <p className="panel-subtitle">
          This was not a plug-and-play ML project.
        </p>

        <p>
          My primary interest lies in <strong>Information & Network Security
          (INS)</strong>, not pure machine learning or deep learning. As a result,
          most of the ML concepts used here were learned from scratch while
          building this system.
        </p>

        <ul style={{ paddingLeft: "1.2rem", marginTop: "0.8rem" }}>
          <li>Understanding text preprocessing and feature extraction</li>
          <li>Training and comparing classical ML models</li>
          <li>Exploring transformer-based models like DistilBERT</li>
          <li>Dealing with overfitting, accuracy drops, and trade-offs</li>
        </ul>
      </section>

      <section className="panel">
        <h2>What This Project Is - and Isn’t</h2>
        <p>
          This project is not advanced research or a production-grade security
          product. It is a learning-focused system built to understand:
        </p>

        <ul style={{ paddingLeft: "1.2rem" }}>
          <li>How phishing language differs from legitimate communication</li>
          <li>How ML models interpret and classify such patterns</li>
          <li>Why explainability matters in cybersecurity systems</li>
        </ul>

        <p style={{ marginTop: "0.8rem" }}>
          Accuracy alone is not the goal - understanding the decision-making
          process is.
        </p>
      </section>

      <section className="panel">
        <h2>Ethical Note</h2>
        <p>
          All phishing examples and simulations used in this platform are
          strictly for awareness and educational purposes. This system is not
          intended to assist in real-world attacks or misuse.
        </p>
      </section>
    </main>
  );
}
