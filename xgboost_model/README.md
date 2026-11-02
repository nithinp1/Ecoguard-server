# XGBoost Scam Detection Module

Standalone ML-based scam classifier. Fully decoupled from `server.py` and `/rulebased` — no cross-imports.

## Files

| File | Purpose |
|------|---------|
| `features.py` | Extracts 13 numeric features from a transcript |
| `train.py` | Trains the XGBoost model, saves to `model.json` |
| `predict.py` | Loads `model.json`, exposes `score_transcript()` |
| `test_predict.py` | Smoke test with example transcripts |
| `synthetic_data.csv` | Placeholder labeled dataset (generated on first train) |
| `model.json` | Saved trained model |

## Features (13 total)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `word_count` | count | Number of words |
| 2 | `char_count` | count | Number of characters |
| 3 | `question_density` | ratio | `?` per character |
| 4 | `exclamation_density` | ratio | `!` per word |
| 5 | `uppercase_density` | ratio | Uppercase chars / total chars |
| 6 | `scam_keyword_count` | count | Matches against known scam keyword list |
| 7 | `urgency_count` | count | Urgency phrases detected |
| 8 | `sensitive_info_count` | count | Requests for PIN/CVV/password |
| 9 | `remote_access_count` | count | AnyDesk/TeamViewer mentions |
| 10 | `impersonation_count` | count | Bank/authority impersonation patterns |
| 11 | `account_threat_count` | count | Account closure / legal threat phrases |
| 12 | `otp_mentioned` | binary | Whether "OTP" appears in text |
| 13 | `sender_type` | categorical | 0=DLT registered, 1=plain number, 2=unknown |

Keyword lists mirror `rulebased/engine.py` for consistency between the two engines.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train (generates synthetic_data.csv if missing, saves model.json)
python train.py

# Smoke test
python test_predict.py

# Use in code
from predict import score_transcript
result = score_transcript("Share your OTP now", "plain_number")
# → {"risk_score": 87, "reason": "...", "flags": [...], "source": "xgboost"}
```

## Retraining with real data

Replace `synthetic_data.csv` with a CSV containing columns: `transcript`, `sender_type`, `label` (0=normal, 1=scam). Then:

```bash
python train.py --data your_real_data.csv
```

The new model saves to the same `model.json` path, immediately usable by `predict.py`.

## Caveat

**The included synthetic dataset is a placeholder.** It has only ~40 hand-crafted rows and the model should NOT be used for real decisions until trained on a proper labeled corpus from the project's test data.
