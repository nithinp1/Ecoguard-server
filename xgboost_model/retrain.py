"""Retrain XGBoost on evaluation_48.csv (real scam/legitimate transcripts).

Usage:
    cd xgboost_model/
    python retrain.py
"""
import csv
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_DIR, "..", "..", "..", ".."))
EVAL_CSV = os.path.join(_PROJ, "benchmark", "datasets", "evaluation_48.csv")
MODEL_PATH = os.path.join(_DIR, "model.json")

sys.path.insert(0, _DIR)
import xgboost as xgb
from features import extract_features, FEATURE_NAMES

def main():
    with open(EVAL_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X, y = [], []
    for row in rows:
        label = 1 if row["expected_label"] == "SCAM" else 0
        features = extract_features(row["input"], "unknown")
        X.append(features)
        y.append(label)

    print(f"Loaded {len(rows)} samples ({sum(y)} scam, {len(y)-sum(y)} legitimate)")
    print(f"Feature count: {len(FEATURE_NAMES)}")

    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.1,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.5,
    }

    model = xgb.train(params, dtrain, num_boost_round=100)

    preds = model.predict(dtrain)
    predictions = [1 if p >= 0.5 else 0 for p in preds]
    tp = sum(1 for p, l in zip(predictions, y) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, y) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, y) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, y) if p == 0 and l == 1)
    acc = (tp + tn) / len(y)
    print(f"\nTraining results:")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")

    model.save_model(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
