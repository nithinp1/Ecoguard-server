"""Quick smoke test for score_transcript()."""
from predict import score_transcript


def main():
    tests = [
        ("Please share your OTP so I can verify your account", "unknown"),
        ("Hi, how are you doing today?", "dlt_registered"),
        ("Install AnyDesk and share the code with me", "unknown"),
        ("Your account will be blocked within 2 hours. Share your PIN now.", "plain_number"),
        ("This is SBI customer care. We need your CVV to update KYC.", "unknown"),
        ("Meeting tomorrow at 10am, don't forget to bring the report", "dlt_registered"),
        ("Urgent: install TeamViewer and give me remote access to fix your computer", "plain_number"),
        ("Your package has been delivered to your doorstep", "dlt_registered"),
    ]

    print(f"{'Score':>5}  {'Source':<9}  Reason")
    print("-" * 70)
    for transcript, sender in tests:
        r = score_transcript(transcript, sender)
        print(f"{r['risk_score']:>5}  {r['source']:<9}  {r['reason']}")
        if r["flags"]:
            for f in r["flags"]:
                print(f"        -> {f}")
        print()


if __name__ == "__main__":
    main()
