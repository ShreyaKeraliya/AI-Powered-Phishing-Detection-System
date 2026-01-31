# AI-Powered Phishing Detection System  
## Case Study: Detecting Phishing Emails and URLs Using ML & NLP


## Executive Summary

Phishing remains one of the most common initial attack vectors in cybersecurity incidents, exploiting human trust rather than system vulnerabilities.

This project presents an **end-to-end AI-powered phishing detection system** capable of analyzing both **emails and URLs**, combining **classical machine learning** with **transformer-based NLP models**.

The system is deployed as a real-world web application with a **Next.js frontend** and a **FastAPI backend**, emphasizing not only prediction accuracy but also **deployment practicality, latency, and security considerations**.


## Problem Statement

Traditional rule-based phishing filters struggle against:

- Rapidly evolving phishing templates
- Semantic manipulation of email content
- URL obfuscation techniques (shorteners, IP-based URLs, deep subdomains)

The goal of this project was to design a system that:

- Detects phishing in **real time**
- Balances **accuracy vs inference cost**
- Is suitable for **practical deployment**, not just offline evaluation


## Threat Model

| Asset            | Threat            | Potential Impact          |
|------------------|-------------------|---------------------------|
| User credentials | Phishing emails   | Account takeover          |
| Browser session  | Malicious URLs    | Malware delivery          |
| Trust in system  | False positives   | User fatigue / bypass     |

The system prioritizes minimizing false negatives while controlling false positives to maintain user trust.



Attackers are assumed to:

- Use social engineering language
- Obfuscate URLs
- Adapt phrasing to evade keyword-based filters


## System Architecture

```text
[ User Browser ]
       |
       v
[ Next.js Frontend ]
       |
       v
[ FastAPI Backend ]
       |
       +--> Email TF-IDF + Logistic Regression
       +--> Email DistilBERT (Transformer NLP)
       +--> URL RandomForest + engineered features

```

### Design Rationale

- Frontend handles user interaction and visualization
- Backend isolates ML inference and security-sensitive logic
- Multiple models allow **accuracy vs latency** trade-offs


## AI & Model Selection Strategy

### Email Detection

Two approaches were implemented:

#### 1. TF-IDF + Logistic Regression
- Fast inference
- Strong baseline
- Lower compute cost

#### 2. DistilBERT (Transformer)
- Captures semantic intent
- Better at detecting paraphrased phishing
- Higher inference cost


This dual-model design allows comparison between **classical ML** and **modern NLP** in a security context.


### URL Detection

- Random Forest classifier
- Feature-engineered inputs:
  - URL length
  - Subdomain depth
  - IP address usage
  - Special characters
  - Suspicious tokens

This approach mirrors real-world URL reputation systems while remaining **interpretable and explainable**.


## Security Considerations

- Input validation on all user-submitted data
- Model inference isolated behind API endpoints
- No raw model exposure to the client

The system design accounts for:
- Model abuse
- False positives as a usability and trust risk
- Dataset bias affecting real-world performance


## Implementation Challenges & Lessons Learned

### Deployment Issues
- CORS misconfiguration between frontend and backend
- Environment variable handling in production
- Cold-start latency for transformer models

### Lessons Learned
- Deployment constraints matter as much as model accuracy
- Smaller transformer models offer better real-world trade-offs
- Security systems must consider **user trust**, not just detection rate


## Results & Evaluation

- Both email models achieved reliable phishing detection on known datasets
- Transformer-based model performed better on semantically rewritten phishing attempts
- URL model successfully identified common obfuscation techniques

Evaluation focused on **practical effectiveness**, not benchmark chasing.

No claims are made about state-of-the-art performance, as the focus was on deployability and robustness.


## Limitations & Future Work

### Current Limitations
- Vulnerable to highly novel (zero-day) phishing patterns
- No adversarial training implemented yet
- Explainability can be improved further

### Planned Improvements
- Adversarial phishing simulation
- Browser extension integration
- Explainable AI (XAI) techniques for security decisions


## Demo & Resources

- 🔗 **Live Demo:** *https://ai-powered-phishing-detection-system-2.onrender.com*
- 📦 **Source Code:** *https://github.com/ShreyaKeraliya/AI-Powered-Phishing-Detection-System*


## Final Note

This project demonstrates the **end-to-end design of an AI-driven security system**, emphasizing not just model performance but also **threat modeling, deployment challenges, and real-world constraints**.
