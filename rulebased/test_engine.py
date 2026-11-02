"""Standalone test script for the rule-based engine.

Run: python -m rulebased.test_engine (from project root)
   or: python test_engine.py (from rulebased/ directory with PYTHONPATH)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rulebased.engine import score_transcript


SCAM_TESTS = [
    # OTP requests
    ("Please share your OTP so I can verify your account", "unknown", 40),
    ("Send me the OTP you received on your phone", "plain_number", 50),
    ("Tell me the OTP number", "dlt_registered", 40),

    # Remote access tools
    ("Install AnyDesk and share the code with me", "unknown", 50),
    ("Download TeamViewer so I can help you", "unknown", 50),
    ("We need remote access to fix your computer", "plain_number", 60),

    # Sensitive info requests
    ("What is your CVV number?", "unknown", 30),
    ("Share your card number and PIN", "unknown", 30),
    ("Enter your net banking password", "plain_number", 40),

    # Urgency phrases
    ("Urgent action required within 2 hours", "unknown", 20),
    ("Act immediately or your account will be blocked", "unknown", 40),
    ("Last warning - respond within 30 minutes", "dlt_registered", 20),

    # Account threats
    ("Your account will be blocked if you don't comply", "unknown", 20),
    ("Legal action will be taken against you", "plain_number", 30),
    ("Your SIM will be deactivated tomorrow", "unknown", 20),

    # Impersonation
    ("Calling from SBI customer care, we need your KYC update", "unknown", 30),
    ("This is the police, we have an arrest warrant", "unknown", 30),
    ("I am from RBI, your account has suspicious activity", "plain_number", 40),

    # Combined high-score cases
    ("SBI customer care here. Share OTP and CVV immediately or account blocked", "unknown", 100),
    ("Install AnyDesk now. Send OTP. Your account will be deactivated in 10 minutes.", "plain_number", 100),
    # Task scams: low-effort engagement work with a payment claim and WhatsApp/Telegram hand-off.
    ("3,000/day by liking YouTube videos or completing simple online tasks. Message us on WhatsApp to start.", "unknown", 70),
    ("Jio Alert: SPAM. Earn 3000/day by reviewing videos; contact us on WhatsApp.", "unknown", 100),
]

LEGIT_TESTS = [
    ("Hi, how are you doing today?", "dlt_registered", 0),
    ("Meeting tomorrow at 10am, don't forget to bring the report", "dlt_registered", 0),
    ("Your OTP for login is 123456. Do not share with anyone.", "dlt_registered", 0),
    ("Thank you for your purchase. Your order will arrive by 5pm.", "plain_number", 0),
    ("Hello, this is a reminder for your dentist appointment", "dlt_registered", 0),
    ("Watch our YouTube video and message us on WhatsApp if you have questions", "dlt_registered", 0),
]


def run_tests():
    print("=" * 60)
    print("SCAM TESTS (expect high scores)")
    print("=" * 60)
    passed = 0
    for transcript, sender, expected_min in SCAM_TESTS:
        r = score_transcript(transcript, sender)
        status = "PASS" if r["risk_score"] >= expected_min else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"[{status}] score={r['risk_score']:>3} expect>={expected_min} sender={sender}")
        print(f"       reason: {r['reason']}")
        print(f"       flags: {r['flags']}")
        print(f"       text: {transcript[:80]}...")
        print()

    print("=" * 60)
    print("LEGIT TESTS (expect 0 or very low scores)")
    print("=" * 60)
    legit_passed = 0
    for transcript, sender, expected_max in LEGIT_TESTS:
        r = score_transcript(transcript, sender)
        status = "PASS" if r["risk_score"] <= expected_max else "FAIL"
        if status == "PASS":
            legit_passed += 1
        print(f"[{status}] score={r['risk_score']:>3} expect<={expected_max} sender={sender}")
        print(f"       reason: {r['reason']}")
        print(f"       flags: {r['flags']}")
        print(f"       text: {transcript[:80]}...")
        print()

    total = len(SCAM_TESTS) + len(LEGIT_TESTS)
    ok = passed + legit_passed
    print(f"Result: {ok}/{total} tests passed")
    return ok == total


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
