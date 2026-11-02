# Rule-Based Scam Detection Engine

Standalone, zero-dependency scam detector. Pure deterministic string/pattern matching — fully explainable and reproducible.

## Usage

```python
from rulebased.engine import score_transcript

result = score_transcript(
    "Please share your OTP immediately or your account will be blocked",
    sender_type="plain_number"  # or "dlt_registered", "unknown"
)
# {
#   "risk_score": 80,
#   "reason": "OTP request detected",
#   "flags": ["OTP request detected", "Urgency phrase: \"act immediately\"",
#             "Account/legal threat: \"your account will be blocked\"",
#             "Non-registered sender (plain_number)"],
#   "source": "rulebased"
# }
```

## Rule List & Point Values

| Rule | Points | Trigger Examples |
|------|--------|------------------|
| **OTP request** | +40 | "share your OTP", "send me the OTP", "tell me the OTP", "enter the OTP" |
| **Remote access tool** | +50 | AnyDesk, TeamViewer, UltraViewer, "remote desktop", "screen share" |
| **Sensitive info request** | +30 | "share CVV", "card number", "PIN", "net banking password" |
| **Urgency phrase** | +20 | "urgent action required", "act now", "within 2 hours", "within 30 minutes", "last warning" |
| **Account/legal threat** | +20 | "account will be blocked", "legal action", "police complaint", "SIM will be deactivated" |
| **Authority impersonation** | +30 | "calling from SBI", "I am from RBI/police/government", "bank customer care" |

### Sender-Type Amplifier

Added once (not per-rule) when any rule fires:

| sender_type | Bonus |
|-------------|-------|
| `dlt_registered` | +0 |
| `plain_number` | +10 |
| `unknown` | +10 |

**Max score is capped at 100.**

## Adding New Rules

1. Create a new function in `engine.py` following the pattern:

```python
def _my_new_rule(text: str):
    """Short docstring."""
    if "trigger phrase" in text:
        return 25, "Human-readable flag"
    return 0, None
```

2. Add it to the `_RULES` list (order determines priority for `reason`):

```python
_RULES = [
    _otp_request,
    _remote_access,
    _my_new_rule,   # <-- add here
    _sensitive_info,
    ...
]
```

3. Add test cases in `test_engine.py` to validate.

## Running Tests

```bash
cd rulebased
python -m test_engine
```

## Design Notes

- **No ML, no LLMs, no external calls** — pure Python + `re`
- **Deterministic** — same input always produces same output
- **Explainable** — every point has a corresponding flag string
- **Extensible** — add rules by writing a function, no framework needed
- **Isolated** — zero imports from the main server code