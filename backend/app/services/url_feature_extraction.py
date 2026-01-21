import re
import math
from collections import Counter
from urllib.parse import urlparse


def sanitize_url(url: str) -> str:
    url = url.strip()
    url = url.replace("[.]", ".")
    url = url.replace("(.)", ".")
    url = url.replace(" ", "")
    return url

IP_RE = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?!$)|$)){4}"
)

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update",
    "bank", "confirm", "password", "signin", "wp-admin"
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"
]


def has_ip(url: str) -> int:
    return 1 if IP_RE.search(url) else 0


def count_subdomains(parsed) -> int:
    if not parsed.netloc:
        return 0
    parts = parsed.netloc.split(".")
    if len(parts) <= 2:
        return 0
    return len(parts) - 2


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [n / len(s) for n in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs)


def extract_url_features(url: str) -> dict:
    """
    Extract enhanced lexical and structural URL features
    for phishing detection.
    """
    url = sanitize_url(url)
    url = url.strip()
    url_lower = url.lower()
    parsed = urlparse(url)

    return {
        # 🔹 Existing features (unchanged)
        "url_length": len(url),
        "subdomains": count_subdomains(parsed),
        "has_ip": has_ip(url),
        "has_at": 1 if "@" in url else 0,
        "uses_https": 1 if parsed.scheme == "https" else 0,
        "dash_count": url.count("-"),

        # 🔥 New high-impact features
        "digit_count": sum(c.isdigit() for c in url),
        "query_count": url.count("?"),
        "suspicious_word_count": sum(
            word in url_lower for word in SUSPICIOUS_WORDS
        ),
        "is_shortened": 1 if any(
            s in url_lower for s in URL_SHORTENERS
        ) else 0,
        "url_entropy": shannon_entropy(url),
        "domain_length": len(parsed.netloc),
    }
