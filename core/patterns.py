\
import re

# Built-in regex patterns for India-centric invoices (extend as needed).
# These target **values** rather than labels.
PATTERNS = {
    "GSTIN": re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\dZ[0-9A-Z]\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "Email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "Phone": re.compile(r"\b(?:\+?\d{1,3}[- ]?)?(?:\d{10})\b"),
    "PIN": re.compile(r"\b\d{6}\b"),
    "Date": re.compile(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
    "Amount": re.compile(r"\b(?:INR|Rs\.?|₹)?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b"),
}

# Default label synonyms for anchoring
LABEL_SYNONYMS = {
    "GSTIN": ["gstin", "gst no", "gst number", "gst", "goods and services tax"],
    "PAN": ["pan", "pan no", "pan number"],
    "Email": ["email", "e-mail", "mail id"],
    "Phone": ["phone", "mobile", "contact"],
    "PIN": ["pin", "pincode", "postal code", "zip"],
    "Date": ["date", "invoice date", "bill date"],
    "Amount": ["total", "amount", "grand total", "invoice total", "balance due"],
}

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

