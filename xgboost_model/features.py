"""Feature extraction for XGBoost scam detection.

Converts a raw transcript + sender_type into a fixed-order numeric vector.
Keyword lists mirror rulebased/engine.py for consistency, but here they
produce counts/probabilities rather than hard rule triggers.
"""
import re
from typing import List

# Same keywords as rulebased/engine.py, used as numeric feature counts here.
_REMOTE_ACCESS_TOOLS = [
    "anydesk", "teamviewer", "ultraviewer", "remote desktop",
    "screen share", "screen sharing", "remote access",
]

_URGENCY_PHRASES = [
    "urgent action required", "act now", "act immediately",
    "immediately respond", "within 24 hours", "within 2 hours",
    "within 1 hour", "within 30 minutes", "within 10 minutes",
    "last warning", "final warning", "account will be blocked",
    "account will be deactivated", "account suspended",
    "your account is blocked", "your account has been suspended",
    "account will be locked", "do it now or",
    "failure to comply", "failure to respond",
]

_SENSITIVE_INFO_KEYWORDS = [
    "pin", "cvv", "card number", "card no", "password",
    "net banking", "netbanking", "internet banking",
    "aadhaar", "pan number",
]

_IMPERSONATION_KEYWORDS = [
    "bank", "police", "cyber crime", "government", "irda", "rbi",
    "reserve bank", "sebi", "customs", "tax department", "income tax",
    "sbi", "hdfc", "icici", "axis", "kotak", "pnb", "bob",
    "canara", "union bank", "indusind", "idbi",
]

_ACCOUNT_THREAT_KEYWORDS = [
    "account will be", "account has been", "account will be closed",
    "account will be blocked", "legal action", "police complaint",
    "arrest warrant", "court notice", "deactivated", "blocked",
]

_SCAM_KEYWORDS = (
    _REMOTE_ACCESS_TOOLS + _SENSITIVE_INFO_KEYWORDS +
    _IMPERSONATION_KEYWORDS + _ACCOUNT_THREAT_KEYWORDS
)

_SENDER_MAP = {"dlt_registered": 0, "plain_number": 1, "unknown": 2}

# Feature names in output order (used by train.py and predict.py)
FEATURE_NAMES = [
    "word_count", "char_count", "question_density", "exclamation_density",
    "uppercase_density", "scam_keyword_count", "urgency_count",
    "sensitive_info_count", "remote_access_count", "impersonation_count",
    "account_threat_count", "otp_mentioned", "sender_type",
]


def _count_matches(text: str, patterns: list) -> int:
    count = 0
    for p in patterns:
        count += len(re.findall(re.escape(p), text))
    return count


def _count_regex(text: str, patterns: list) -> int:
    count = 0
    for p in patterns:
        count += len(re.findall(p, text))
    return count


def extract_features(transcript: str, sender_type: str = "unknown") -> List[float]:
    """Return a fixed-length numeric feature vector from a transcript.

    The order matches FEATURE_NAMES above. All features are numeric so
    they can feed directly into XGBoost without further encoding.
    """
    text = transcript.lower().strip()
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    total_chars = max(char_count, 1)

    # Punctuation densities
    question_density = text.count("?") / total_chars
    exclamation_density = text.count("!") / max(word_count, 1)

    # Uppercase ratio in original (non-lowered) text
    upper = sum(1 for c in transcript if c.isupper())
    uppercase_density = upper / max(len(transcript), 1)

    # Keyword / phrase counts
    scam_kw_count = _count_matches(text, _SCAM_KEYWORDS)
    urgency_count = _count_matches(text, _URGENCY_PHRASES)
    sensitive_count = _count_regex(text, [
        r"\b(?:share|send|tell|give|enter|what.{0,10}is)\b.{0,30}\b(?:pin|cvv|card\s*number|password|netbanking|internet\s*banking)\b",
        r"\bcard\s*(?:number|no)\b",
        r"\bcvv\b",
        r"\bnet\s*banking\s*(?:password|pin|credentials?)\b",
    ])
    remote_count = _count_matches(text, _REMOTE_ACCESS_TOOLS)
    impersonation_count = _count_regex(text, [
        r"\b(?:this is|calling from|i am from).{0,40}\b(?:bank|police|cyber crime|government|irda|rbi|reserve bank|sebi|customs|tax department|income tax)\b",
        r"\b(?:sbi|hdfc|icici|axis|kotak|pnb|bob|canara|union bank|indusind|idbi)\b.{0,30}\b(?:customer care|helpline|support team|branch)\b",
    ])
    account_threat_count = _count_matches(text, _ACCOUNT_THREAT_KEYWORDS)

    # OTP flag (regex for patterns like "share OTP", "tell me the OTP")
    otp_mentioned = 1 if re.search(r"\botp\b", text) else 0

    sender_val = _SENDER_MAP.get(sender_type, 2)

    return [
        float(word_count), float(char_count),
        question_density, exclamation_density, uppercase_density,
        float(scam_kw_count), float(urgency_count), float(sensitive_count),
        float(remote_count), float(impersonation_count),
        float(account_threat_count), float(otp_mentioned), float(sender_val),
    ]
