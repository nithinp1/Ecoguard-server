"""Standalone rule-based scam detection engine.

Zero dependencies beyond the stdlib. Returns a deterministic, fully
explainable risk score for a transcript.

Scoring (capped at 100):
  Impersonation         +30
  Money transfer req    +30
  Fee/payment request   +25
  OTP request           +40
  Remote access tool    +40
  Sensitive info req    +25
  Account threat        +20
  Urgency/coercion      +15
  Investment bait       +20
  Prize/lottery scam    +25
  Job scam              +20
  Family emergency      +25

Each fired rule appends a human-readable flag. The highest-scoring
single reason is promoted to ``reason``.
"""
import re
from typing import List, Dict

__all__ = ["score_transcript"]

# ---------------------------------------------------------------------------
# Individual rule checks -- each returns (points, flag) or (0, None)
# ---------------------------------------------------------------------------

def _otp_request(text: str):
    """OTP / one-time-password solicitation."""
    # Strip legitimate protective warnings before checking for solicitations:
    # e.g., "Do not share your OTP with anyone", "Never disclose your OTP"
    clean_text = re.sub(r"\b(?:do\s+not|never|don'?t|not)\s+(?:share|disclose|reveal|tell|give)\b[^.!?\n]*", "", text, flags=re.I)
    patterns = [
        r"\b(?:share|send|tell|give|enter|type|read back|what.{0,10}is)\b.{0,30}\botp\b",
        r"\botp\s+(?:batao|bhej|de)\b",
        r"\bshare\s+otp\b", r"\bsend\s+otp\b", r"\benter\s+(?:the\s+)?otp\b",
        r"\btell\s+me\s+(?:the\s+)?otp\b", r"\botp\s+number\b",
        r"\bwhat.{0,10}is.{0,10}otp\b",
        r"\botp\b.{0,20}\b(?:share|send|tell)\b",
    ]
    for p in patterns:
        if re.search(p, clean_text):
            return 40, "OTP request detected"
    return 0, None


def _remote_access(text: str):
    """AnyDesk / TeamViewer / remote desktop mentions."""
    tools = ["anydesk", "teamviewer", "ultraviewer", "remote desktop",
             "screen share", "screen sharing", "remote access"]
    for t in tools:
        if t in text:
            return 40, f"Remote access tool mentioned ({t})"
    return 0, None


def _impersonation(text: str):
    """Claims to be bank / police / government / authority."""
    patterns = [
        (r"\b(?:this is|calling from|i am from|i am calling from|I'm calling from|I'm from|I am officer|we are from|this is officer)\b.{0,50}\b(?:bank|police|police station|police department|cyber crime|cyber cell|government|irda|rbi|reserve bank|sebi|customs|tax department|income tax|department)\b",
         "Authority impersonation detected"),
        (r"\b(?:officer|inspector|constable|sergeant|lieutenant|captain)\b.{0,30}\b(?:police|department)\b",
         "Police impersonation detected"),
        (r"\b(?:sbi|hdfc|icici|axis|kotak|pnb|bob|canara|union bank|indusind|idbi)\b.{0,30}\b(?:customer care|helpline|support team|branch|bank)\b",
         "Bank impersonation detected"),
        (r"\b(?:bank|police|government|tax|customs|rbi)\b.{0,30}\b(?:officer|department|authority|ministry|bureau)\b",
         "Institution impersonation detected"),
    ]
    for pat, flag in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return 30, flag
    return 0, None


def _money_transfer(text: str):
    """Asking to transfer / send money to a specific account."""
    patterns = [
        r"\btransfer\s+(?:the\s+)?(?:money|funds|amount|money|rupees|rs)\b",
        r"\b(?:send|transfer|deposit|move)\s+(?:the\s+)?(?:money|funds|amount|payment)\b.{0,40}\b(?:to|into|account|secure)\b",
        r"\b(?:wire|remit|send|transfer)\s+(?:money|funds|payment|amount)\b",
        r"\baccount\s+(?:number|details|transfer)\b",
        r"\b(?:payment|money|funds)\s+(?:has been|is|to be)\s+(?:sent|transferred|done|received)\b",
        r"\b(?:i have|I've|you need to|please)\s+(?:sent?|transfer|move)\s+(?:the\s+)?(?:money|funds|amount)\b",
        r"\bsecure\s+(?:account|money|fund)\b",
        r"\btransfer\s+(?:details|number|information)\b",
    ]
    for p in patterns:
        if re.search(p, text):
            return 30, "Money transfer request detected"
    return 0, None


def _fee_request(text: str):
    """Requesting upfront fees / processing charges / registration fees."""
    patterns = [
        r"\b(?:registration|processing|processing|service|documentation|activation|membership|handling|cancellation|crossing)\s+(?:fee|charge|cost|amount)\b",
        r"\b(?:fee|charge|cost)\s+(?:of|is|will be|amounting)\s+(?:rs\.?|rupees|inr)?\s*\d+",
        r"\b(?:pay|paying|paid|payment)\s+(?:a\s+)?(?:small\s+)?(?:fee|charge|amount|sum)\b",
        r"\b(?:upfront|advance|deposit)\s+(?:fee|charge|payment|amount)\b",
        r"\b(?:small|nominal|minimal|one-time)\s+(?:fee|charge|amount|payment|sum|cost)\b",
        r"\b(?:fee|charge|payment)\s+(?:for|to|towards|of)\s+(?:documentation|processing|verification|registration|activation)\b",
    ]
    for p in patterns:
        if re.search(p, text):
            return 25, "Upfront fee request detected"
    return 0, None


def _sensitive_info(text: str):
    """Asking for card number, PIN, CVV, password, bank credentials."""
    clean_text = re.sub(r"\b(?:do\s+not|never|don'?t|not)\s+(?:share|disclose|reveal|tell|give)\b[^.!?\n]*", "", text, flags=re.I)
    patterns = [
        (r"\b(?:share|send|tell|give|enter|what.{0,10}is|provide|confirm)\b.{0,30}\b(?:pin|cvv|card\s*number|password|netbanking|internet\s*banking|otp|account\s*number|aadhaar|pan|upi|atm)\b",
         "Request for sensitive financial info"),
        (r"\bcard\s*(?:number|no)\b", "Card number requested"),
        (r"\bcvv\b", "CVV requested"),
        (r"\bnet\s*banking\s*(?:password|pin|credentials?)\b", "Net-banking credentials requested"),
        (r"\b(?:your|your)\s+(?:pin|cvv|password|account\s*number|aadhaar|pan)\s+(?:number|is|please|tell|share|send)\b",
         "Sensitive info solicitation"),
        r"\bverify.{0,20}\b(?:account|identity|details|information|credentials|number)\b",
        r"\bneed.{0,20}\b(?:account|account\s*number|card|details)\s+(?:number|details|info)\b",
    ]
    for pat in patterns:
        if isinstance(pat, tuple):
            if re.search(pat[0], clean_text, re.IGNORECASE):
                return 25, pat[1]
        else:
            if re.search(pat, clean_text, re.IGNORECASE):
                return 25, "Sensitive info request detected"
    return 0, None


def _account_threat(text: str):
    """Threats about account closure / freezing / deactivation."""
    patterns = [
        r"\baccount\s+(?:will be|has been|is|may be|could be)\s+(?:closed|blocked|frozen|deactivated|suspended|locked|disabled|seized)\b",
        r"\b(?:your\s+)?account\s+(?:is|has been)\s+(?:compromised|flagged|under\s+(?:review|investigation|scrutiny))\b",
        r"\b(?:legal|police)\s+(?:action|complaint|case)\b",
        r"\b(?:arrest|court)\s+(?:warrant|notice|summon)\b",
        r"\b(?:aadhaar|pan|sim|phone|mobile)\s+(?:will be|has been|is)\s+(?:blocked|deactivated|suspended|disabled)\b",
        r"\b(?:consequences|penalty|fine)\s+(?:if|for|of)\b",
        r"\b(?:face|facing)\s+(?:legal|criminal|police)\b",
    ]
    for p in patterns:
        if re.search(p, text):
            return 20, "Account/legal threat detected"
    return 0, None


def _urgency_coercion(text: str):
    """Phrases designed to force immediate action or pressure."""
    patterns = [
        r"\b(?:urgent|immediately|asap|right\s+away|right\s+now|act\s+now|act\s+fast|hurry|hurry\s+up)\b",
        r"\bwithin\s+(?:the\s+)?(?:next\s+)?(?:\d+\s+)?(?:hour|minute|day|week|night|time)\b",
        r"\b(?:don'?t|do\s+not)\s+(?:delay|wait|waste\s+time|miss\s+(?:out|this))\b",
        r"\b(?:last|final)\s+(?:warning|chance|offer|call|notice)\b",
        r"\b(?:expires?|expiring|valid|offer)\s+(?:in|within|only|just|today|now)\b",
        r"\b(?:if\s+you\s+delay|too\s+late|before\s+it'?s?\s+too\s+late|running\s+out)\b",
        r"\bact\s+(?:now|immediately|fast|quickly|right\s+away)\b",
        r"\b(?:press|reply|respond|call\s+back)\s+(?:now|immediately|right\s+away|asap)\b",
    ]
    for p in patterns:
        if re.search(p, text):
            return 15, "Urgency/coercion language detected"
    return 0, None


def _investment_bait(text: str):
    """Investment opportunity / guaranteed returns scam."""
    patterns = [
        r"\b(?:invest|investment|trading|stock|mutual\s+fund)\s+(?:opportunity|scheme|plan|offer|returns?)\b",
        r"\b(?:guaranteed|assured|fixed|sure)\s+(?:returns?|profit|income|earning|money)\b",
        r"\b\d+\s*(?:x|times|fold)\s+return",
        r"\b(?:double|triple|quadruple)\s+(?:your\s+)?(?:money|investment|amount|returns?)\b",
        r"\b(?:lakh|crore|hundred|thousand)\s+(?:return|profit|earn)\b",
        r"\b(?:exclusive|special|limited)\s+(?:opportunity|offer|deal|investment)\b",
        r"\bwe\s+(?:handle|manage|do)\s+(?:everything|trading|investment)\s+(?:for\s+you|on\s+your\s+behalf)\b",
    ]
    for p in patterns:
        if re.search(p, text):
            return 20, "Investment bait detected"
    return 0, None


def _prize_lottery(text: str):
    """Prize / lottery / lottery / winning notification scam."""
    patterns = [
        r"\b(?:congratulations|congrats|you(?:'ve| have)\s+(?:been|won|selected|chosen|chosen|picked|shortlisted))\b",
        r"\b(?:won|winning|prize|award|reward|lottery|contest|raffle|lucky\s+(?:winner|draw|draw))\b",
        r"\b(?:all[\s-]+expense[\s-]+paid|free\s+(?:trip|flight|vacation|holiday|ticket|package))\b",
        r"\b(?:trip|vacation|holiday|flight|tour)\s+(?:to|package|deal|offer)\b",
        r"\b(?:selected|chosen|picked|shortlisted|lucky)\s+(?:for|from|individual|winner|candidate)\b",
        r"\b(?:claim|collect|claiming)\s+(?:your|the|prize|reward|award|money|trip)\b",
    ]
    for p in patterns:
        if re.search(p, text):
            return 25, "Prize/lottery scam detected"
    return 0, None


def _job_scam(text: str):
    """Fake job / work-from-home / recruitment scam."""
    patterns = [
        r"\b(?:work[\s-]+from[\s-]+home|wfh|data\s+entry|typing\s+job|part[\s-]+time\s+(?:job|work))\b",
        r"\b(?:high[\s-]+paying|good\s+(?:salary|pay|money|income|earning))\s+(?:job|offer|work|role|opportunity)\b",
        r"\b(?:earn|earning|make|making)\s+(?:rs\.?|inr|rupees)?\s*\d+\s*(?:k|lakh|thousand|per\s+month|monthly|weekly|daily)\b",
        r"\bno\s+experience\s+(?:required|needed|necessary)\b",
        r"\b(?:salary|pay|income|earning|money|amount)\s+(?:per|each|every)\s+(?:day|week|month|hour)\b",
        r"\b(?:registration|enrollment|joining|sign[\s-]+up)\s+(?:fee|charge|amount|cost)\b",
    ]
    for p in patterns:
        if re.search(p, text):
            return 20, "Job/recruitment scam detected"
    return 0, None


def _task_earning_scam(text: str):
    """Detect paid-like/review/task bait that moves victims to WhatsApp or Telegram.

    This requires the combination of a low-effort online task, a payment promise, and an
    off-platform contact request. It deliberately does not flag normal YouTube, WhatsApp, or job
    mentions on their own.
    """
    has_task = bool(re.search(
        r"\b(?:like|liking|subscribe|subscribing|review|rating|watch(?:ing)?|youtube|video(?:s)?|"
        r"online\s+tasks?|simple\s+(?:online\s+)?tasks?)\b", text))
    has_payment = bool(re.search(
        r"(?:₹|rs\.?|inr)?\s*\d[\d,]*\s*(?:/|per)\s*(?:day|daily|week|weekly|month|monthly)\b|"
        r"\b(?:earn|earning|income|paid)\b", text))
    has_off_platform_contact = bool(re.search(r"\b(?:whats\s*app|whatsapp|telegram)\b", text))
    if has_task and has_payment and has_off_platform_contact:
        return 65, "Task-and-earn scam pattern detected"
    return 0, None


def _carrier_spam_warning(text: str):
    """Honor a carrier-provided spam label when it is present in the inspected SMS text."""
    if re.search(r"\b(?:jio|airtel|vi|vodafone)\s+alert\s*:\s*(?:spam|suspected\s+spam)\b", text):
        return 50, "Carrier spam warning detected"
    return 0, None


def _family_emergency(text: str):
    """Family member in trouble / emergency bail / hospital scam."""
    patterns = [
        r"\b(?:family\s+member|relative|loved\s+one|son|daughter|husband|wife|brother|sister|father|mother|mom|dad|uncle|aunt)\s+(?:is\s+)?(?:in\s+)?(?:trouble|danger|hospital|jail|custody|arrested|missing)\b",
        r"\b(?:bail|hospital|medical|emergency|surgery|accident|injured|hurt)\s+(?:money|fee|charge|fund|payment)\b",
        r"\b(?:need|needs|send|sending|required|urgent)\s+(?:money|fund|funds|cash|payment)\s+(?:now|immediately|right\s+away|fast|asap|quickly)\b",
        r"\b(?:transfer|send|wire|pay)\s+(?:the\s+)?(?:money|funds|cash|amount|payment)\s+(?:now|immediately|right\s+away|urgent|fast|asap)\b",
        r"\b(?:your|a)\s+(?:loved\s+one|family\s+member|close\s+relative)\s+(?:is\s+in\s+)?(?:trouble|danger|jail|custody|hospital)\b",
    ]
    for p in patterns:
        if re.search(p, text):
            return 25, "Family emergency scam detected"
    return 0, None


# All rules in evaluation order
_RULES = [
    _otp_request,
    _remote_access,
    _impersonation,
    _money_transfer,
    _fee_request,
    _sensitive_info,
    _account_threat,
    _urgency_coercion,
    _investment_bait,
    _prize_lottery,
    _job_scam,
    _task_earning_scam,
    _carrier_spam_warning,
    _family_emergency,
]

# Sender-type amplifiers (added once, not per-rule)
_SENDER_BONUS = {
    "dlt_registered": 0,
    "plain_number": 10,
    "unknown": 10,
}


def is_legitimate_bank_otp(text: str, sender_type: str = "unknown") -> bool:
    if sender_type != "dlt_registered":
        return False
    lower = text.lower()
    has_otp_delivery = bool(re.search(
        r"\b(?:your\s+)?otp\s+(?:for|to|is|code)\b|\botp\s*(?:is|:)\s*\d{4,8}\b|\buse\s+otp\s+\d{4,8}\b|\bis\s+\d{4,8}\b",
        lower
    ))
    if not has_otp_delivery:
        return False
    has_url = bool(re.search(r"https?://|bit\.ly|tinyurl|\.xyz\b|\.top\b|\.tk\b|\.link\b", lower))
    if has_url:
        return False
    has_threat = bool(re.search(r"\b(?:blocked|suspended|deactivated|closed|frozen|legal action|arrest|unblock)\b", lower))
    if has_threat:
        return False
    return True


def score_transcript(transcript: str, sender_type: str = "unknown") -> dict:
    """Score a transcript and return a risk assessment.

    Returns:
        {"risk_score": int 0-100, "reason": str, "flags": [str], "source": "rulebased"}
    """
    if is_legitimate_bank_otp(transcript, sender_type):
        return {
            "risk_score": 0,
            "reason": "Legitimate transactional OTP delivery (DLT registered bank)",
            "flags": ["DLT verified transactional route"],
            "source": "rulebased"
        }

    text = transcript.lower()
    total = 0
    flags = []

    max_pts = 0
    best_flag = ""

    for rule in _RULES:
        points, flag = rule(text)
        if points > 0:
            flags.append(flag)
            total += points
            if points > max_pts:
                max_pts = points
                best_flag = flag or ""

    # Sender-type amplifier
    bonus = _SENDER_BONUS.get(sender_type, 10)
    if bonus and total > 0:
        flags.append(f"Non-registered sender ({sender_type})")
        total += bonus

    total = min(total, 100)

    if total == 0:
        return {
            "risk_score": 0,
            "reason": "No scam indicators found",
            "flags": [],
            "source": "rulebased",
        }

    return {
        "risk_score": total,
        "reason": best_flag,
        "flags": flags,
        "source": "rulebased",
    }


# ---------------------------------------------------------------------------
# Quick standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("Please share your OTP so I can verify your account", "unknown"),
        ("Hi, how are you doing today?", "dlt_registered"),
        ("Install AnyDesk and share the code with me", "unknown"),
        ("Your account will be blocked within 2 hours. Share your PIN now.", "plain_number"),
        ("I am officer Danny from the local police department. We received reports of fraudulent activities. Transfer funds to a secure account.", "unknown"),
        ("Your profile was shortlisted for a high-paying work from home job. Registration fee of 2000 to confirm your spot.", "plain_number"),
        ("Congratulations, you won an all expense paid trip to Dubai. Pay a small crossing fee of 5000 rupees.", "unknown"),
        ("We have an exclusive stock market opportunity. Invest just 10000 and get one lakh returns.", "plain_number"),
        ("A close family member is in trouble and needs bail money. Send the money now.", "unknown"),
        ("Hello sir, how are you today?", "dlt_registered"),
    ]
    for text, sender in tests:
        result = score_transcript(text, sender)
        label = "SCAM" if result["risk_score"] >= 70 else "LEGIT"
        print(f"[{label}] risk={result['risk_score']:3d} flags={result['flags']}")
        print(f"       text={text[:80]}...")
        print()
