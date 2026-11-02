"""Train an XGBoost classifier on labeled transcripts.

CAVEAT: The synthetic CSV shipped here is a PLACEHOLDER — it exists only
to get the pipeline working end-to-end. Replace it with real labeled
data from the project's test corpus before any production use.

Usage:
    python train.py                       # train on synthetic data
    python train.py --data my_labels.csv  # train on your own CSV
"""
import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from features import extract_features, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")
DATA_PATH = os.path.join(os.path.dirname(__file__), "synthetic_data.csv")

# ── Placeholder synthetic dataset (30-40 rows) ──────────────────────────
# REPLACE THIS with real labeled data before production use.
SYNTHETIC_DATA = [
    # ── Scam examples (label=1) ──
    ("Please share your OTP so I can verify your account", "unknown", 1),
    ("Install AnyDesk and share the code with me immediately", "plain_number", 1),
    ("Your account will be blocked within 2 hours. Share your PIN now.", "plain_number", 1),
    ("This is SBI customer care. We need your CVV to update KYC.", "unknown", 1),
    ("Urgent: your account has been suspended. Call back now.", "plain_number", 1),
    ("Give me your net banking password so I can fix the issue", "unknown", 1),
    ("I am from cyber crime department. Send your Aadhaar number.", "plain_number", 1),
    ("Your PAN will be deactivated. Act now or face legal action.", "unknown", 1),
    ("Tell me the OTP that was just sent to your phone", "plain_number", 1),
    ("Install TeamViewer on your PC, I will fix it remotely", "unknown", 1),
    ("Share your card number and CVV for verification", "plain_number", 1),
    ("Final warning: account will be closed within 1 hour", "unknown", 1),
    ("I am calling from HDFC bank helpline. Enter your PIN to proceed.", "plain_number", 1),
    ("Your SIM will be deactivated today. Press 1 to confirm.", "unknown", 1),
    ("Send your OTP to 12345 immediately or lose your account", "plain_number", 1),
    ("We need your internet banking password to process the refund", "unknown", 1),
    ("Police complaint will be filed if you do not share OTP now", "plain_number", 1),
    ("This is income tax department. Your PAN is blocked. Share Aadhaar.", "unknown", 1),
    ("Act immediately: your account has been compromised. Send PIN.", "plain_number", 1),
    ("Install Ultraviewer and let me access your computer screen", "unknown", 1),
    # ── Normal examples (label=0) ──
    ("Hi, how are you doing today?", "dlt_registered", 0),
    ("Meeting tomorrow at 10am, don't forget to bring the report", "dlt_registered", 0),
    ("Your package has been delivered to your doorstep", "dlt_registered", 0),
    ("Happy birthday! Hope you have a wonderful day", "dlt_registered", 0),
    ("Reminder: dentist appointment on Friday at 3pm", "dlt_registered", 0),
    ("The quarterly report is ready for review", "dlt_registered", 0),
    ("Lunch at noon? Let me know if you are free", "dlt_registered", 0),
    ("Your order #1234 has shipped and will arrive Tuesday", "dlt_registered", 0),
    ("Team standup in 5 minutes, please join the call", "dlt_registered", 0),
    ("Thanks for helping with the project last week", "dlt_registered", 0),
    ("Can you send me the updated spreadsheet?", "dlt_registered", 0),
    ("Movie tonight? There is a new thriller at the cinema", "dlt_registered", 0),
    ("Your subscription renews on the 15th of next month", "dlt_registered", 0),
    ("Great presentation today, the client was impressed", "dlt_registered", 0),
    ("Happy anniversary! Wishing you both all the best", "dlt_registered", 0),
    ("The weather looks great this weekend for a hike", "dlt_registered", 0),
    ("Please review the attached document when you get a chance", "dlt_registered", 0),
    ("Your flight is confirmed for tomorrow at 8am", "dlt_registered", 0),
    ("Just checking in — are we still on for Thursday?", "dlt_registered", 0),
    ("The kids had a great time at the party yesterday", "dlt_registered", 0),
]


def _generate_csv(path: str):
    """Write the placeholder synthetic dataset to CSV."""
    df = pd.DataFrame(SYNTHETIC_DATA, columns=["transcript", "sender_type", "label"])
    df.to_csv(path, index=False)
    print(f"Wrote placeholder dataset ({len(df)} rows) to {path}")


def train(data_path: str = DATA_PATH):
    """Train XGBoost model and save to MODEL_PATH."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    # Extract features
    X = [extract_features(row["transcript"], row["sender_type"]) for _, row in df.iterrows()]
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.3,
        "subsample": 0.8,
        "seed": 42,
    }

    model = xgb.train(params, dtrain, num_boost_round=50, evals=[(dtest, "test")])

    # Evaluate
    preds = (model.predict(dtest) > 0.5).astype(int)
    print("\n" + classification_report(y_test, preds, target_names=["normal", "scam"]))

    # Save
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Print feature importances
    importance = model.get_score(importance_type="weight")
    print("\nFeature importances (weight):")
    for feat in sorted(importance, key=importance.get, reverse=True):
        print(f"  {feat}: {importance[feat]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost scam detection model")
    parser.add_argument("--data", default=DATA_PATH, help="Path to labeled CSV")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        _generate_csv(DATA_PATH)
    train(args.data)
