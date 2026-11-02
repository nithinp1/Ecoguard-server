import os
import json
import pytest
from fastapi.testclient import TestClient
from server import (
    app, split_into_sentences, is_minimum_sentence,
    save_sms_record, load_saved_sms, delete_saved_sms, clear_all_saved_sms,
    SMS_FILE
)

@pytest.fixture(autouse=True)
def clean_sms_db(tmp_path, monkeypatch):
    test_file = str(tmp_path / "test_sms.json")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump([], f)
    monkeypatch.setattr("server.SMS_FILE", test_file)
    yield

def test_split_into_sentences():
    # Multi-sentence paragraph from Sarvam AI / ASR
    text = "Hello sir I am calling from State Bank of India. Your debit card has been suspended. Please confirm the OTP."
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello sir I am calling from State Bank of India."
    assert sentences[1] == "Your debit card has been suspended."
    assert sentences[2] == "Please confirm the OTP."

def test_split_into_sentences_preserves_abbreviations():
    text = "Transfer Rs. 50,000 to Dr. Sharma immediately. Your account has suspicious activity."
    sentences = split_into_sentences(text)
    assert len(sentences) == 2
    assert "Rs. 50,000" in sentences[0]
    assert "Dr. Sharma" in sentences[0]
    assert sentences[1] == "Your account has suspicious activity."

def test_is_minimum_sentence():
    # Sub-sentence fragment should NOT trigger LLM
    assert not is_minimum_sentence("hello sir")
    assert not is_minimum_sentence("yes ok")
    assert not is_minimum_sentence("uh huh")

    # Complete punctuated sentence with >= 3 words
    assert is_minimum_sentence("Please confirm the OTP.")
    assert is_minimum_sentence("Your card is blocked!")

    # Explicit scam keyword with >= 3 words
    assert is_minimum_sentence("send me the otp")

    # Stream of words without punctuation reaching >= 6 words
    assert is_minimum_sentence("i am calling from mumbai police regarding arrest warrant")

def test_sms_persistence_and_endpoints():
    client = TestClient(app)

    # 1. Initially empty
    res = client.get("/api/sms")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["total"] == 0
    assert data["records"] == []

    # 2. Add an SMS via POST
    sms_payload = {
        "sender": "VM-SBIINB",
        "sender_type": "dlt_registered",
        "body": "Dear customer, your SBI account has been credited with Rs 5000 on 01-Nov-2026."
    }
    post_res = client.post("/api/sms", json=sms_payload)
    assert post_res.status_code == 200
    record = post_res.json()["record"]
    assert record["sender"] == "VM-SBIINB"
    assert record["sender_type"] == "dlt_registered"
    assert record["risk_score"] < 40
    assert record["classification"] == "SAFE"

    # 3. Add a smishing threat SMS
    scam_payload = {
        "sender": "+919876543210",
        "sender_type": "plain_number",
        "body": "URGENT: Your bank account is suspended. Click http://bit.ly/sbi-verify immediately and enter OTP."
    }
    post_scam_res = client.post("/api/sms", json=scam_payload)
    assert post_scam_res.status_code == 200
    scam_record = post_scam_res.json()["record"]
    assert scam_record["risk_score"] >= 50
    assert scam_record["classification"] in ("CRITICAL SMISHING", "SUSPICIOUS")

    # 4. Check list endpoint counts
    res2 = client.get("/api/sms")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["total"] == 2
    assert data2["scams"] >= 1

    # 5. Delete single SMS
    del_res = client.delete(f"/api/sms/{record['id']}")
    assert del_res.status_code == 200

    # Verify only 1 remains
    res3 = client.get("/api/sms")
    assert res3.json()["total"] == 1

    # 6. Clear all
    clear_res = client.delete("/api/sms")
    assert clear_res.status_code == 200
    res4 = client.get("/api/sms")
    assert res4.json()["total"] == 0


def test_legitimate_bank_otp_scoring():
    client = TestClient(app)
    # Legitimate DLT Bank OTP message
    otp_payload = {
        "sender": "VM-HDFCBK",
        "sender_type": "dlt_registered",
        "body": "Your OTP for login to HDFC NetBanking is 591832. Valid for 10 minutes. Do not share your OTP with anyone, including bank staff."
    }
    res = client.post("/api/sms", json=otp_payload)
    assert res.status_code == 200
    record = res.json()["record"]
    assert record["risk_score"] == 0
    assert record["classification"] == "SAFE"
    assert "legitimate" in record["reason"].lower() or "verified" in record["reason"].lower()

    # Android integrations may send the sender type in a different case or
    # with a hyphen; it must still take the verified-bank OTP path.
    variant_payload = {
        "sender": "VM-HDFCBK",
        "sender_type": "DLT-REGISTERED",
        "body": "Your OTP for login to HDFC NetBanking is 123456. Valid for 10 minutes. Do not share it with anyone."
    }
    variant_res = client.post("/api/sms", json=variant_payload)
    assert variant_res.status_code == 200
    variant_record = variant_res.json()["record"]
    assert variant_record["risk_score"] == 0
    assert variant_record["classification"] == "SAFE"

    # Phishing bank SMS posing as bank with link must be caught
    phish_payload = {
        "sender": "AX-HDFCBK",
        "sender_type": "dlt_registered",
        "body": "Dear customer, your bank account has been suspended due to pending KYC. Click http://hdfc-verify-kyc.com/update immediately to unblock."
    }
    phish_res = client.post("/api/sms", json=phish_payload)
    assert phish_res.status_code == 200
    phish_record = phish_res.json()["record"]
    assert phish_record["risk_score"] >= 40
    assert phish_record["classification"] in ("CRITICAL SMISHING", "SUSPICIOUS")


