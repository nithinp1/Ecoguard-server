"""Predict scam risk for a transcript using the trained XGBoost model.

Load the model once at import time; score_transcript() is the public API.
"""
import os
import sys
import xgboost as xgb

# Ensure xgboost_model dir is on path for sibling imports
_dir = os.path.dirname(__file__)
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from features import extract_features, FEATURE_NAMES

MODEL_PATH = os.path.join(_dir, "model.json")

# Load model once at import time
_model = xgb.Booster()
_model.load_model(MODEL_PATH)


def score_transcript(transcript: str, sender_type: str = "unknown") -> dict:
    """Score a transcript and return a risk assessment.

    Returns:
        {"risk_score": int 0-100, "reason": str, "flags": [str], "source": "xgboost"}
    """
    features = extract_features(transcript, sender_type)
    dmat = xgb.DMatrix([features], feature_names=FEATURE_NAMES)
    prob = float(_model.predict(dmat)[0])
    risk_score = int(round(prob * 100))

    # Build human-readable flags from features that fired
    flags = []
    reasons = []
    feat_vals = dict(zip(FEATURE_NAMES, features))

    if feat_vals["otp_mentioned"] > 0:
        flags.append("OTP keyword detected")
        reasons.append("OTP mentioned")
    if feat_vals["remote_access_count"] > 0:
        flags.append(f"Remote access tool(s) mentioned ({int(feat_vals['remote_access_count'])}x)")
        reasons.append("remote access tool")
    if feat_vals["sensitive_info_count"] > 0:
        flags.append("Sensitive info request detected")
        reasons.append("sensitive info request")
    if feat_vals["urgency_count"] > 0:
        flags.append(f"Urgency language ({int(feat_vals['urgency_count'])}x)")
        reasons.append("urgency language")
    if feat_vals["impersonation_count"] > 0:
        flags.append("Authority/bank impersonation detected")
        reasons.append("impersonation")
    if feat_vals["account_threat_count"] > 0:
        flags.append(f"Account/legal threat ({int(feat_vals['account_threat_count'])}x)")
        reasons.append("account threat")
    if feat_vals["sender_type"] >= 1 and risk_score > 20:
        flags.append(f"Non-registered sender ({sender_type})")

    reason = "Likely scam: " + ", ".join(reasons) if reasons else "No strong scam indicators"

    return {
        "risk_score": risk_score,
        "reason": reason,
        "flags": flags,
        "source": "xgboost",
    }
