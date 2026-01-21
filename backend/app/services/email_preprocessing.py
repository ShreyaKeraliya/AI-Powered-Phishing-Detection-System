import re
from bs4 import BeautifulSoup


HTML_TAG_RE = re.compile(r"<.*?>")
URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
EMAIL_HEADER_RE = re.compile(
    r"^(from|to|subject|cc|bcc|reply-to|return-path|received|date):.*$",
    re.IGNORECASE | re.MULTILINE,
)
PUNCT_RE = re.compile(r"[^\w\s]")


def clean_email_text(subject: str, body: str) -> str:
    """
    Basic email preprocessing:
    - combine subject and body
    - lowercase
    - strip HTML
    - remove URLs and headers
    - remove punctuation (keep numbers)
    """
    combined = f"{subject}\n{body}" if subject else body

    # Remove HTML tags using BeautifulSoup then regex fallback
    try:
        soup = BeautifulSoup(combined, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        text = re.sub(HTML_TAG_RE, " ", combined)

    text = text.lower()
    text = re.sub(EMAIL_HEADER_RE, " ", text)
    text = re.sub(URL_RE, " ", text)
    text = re.sub(PUNCT_RE, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


