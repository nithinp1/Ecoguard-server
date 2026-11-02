"""ScamShield AI Server.
Production-ready with Web UI Dashboard, Call Recording, Transcripts History,
Persistent Last-Call State, and Multi-Engine Scoring (LLM + XGBoost + Rule-Based).
"""
import asyncio
import base64
import gc
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import urllib.parse
import wave
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Ensure NVIDIA CUDA / cuBLAS / cuDNN DLLs are discoverable on Windows (Python 3.8+)
if sys.platform == "win32":
    import site
    _dirs_to_check = set()
    if hasattr(site, "getsitepackages"):
        _dirs_to_check.update(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        _dirs_to_check.add(site.getusersitepackages())
    _dirs_to_check.add(os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages"))
    for _p in sys.path:
        if "site-packages" in _p:
            _dirs_to_check.add(_p)

    for _sp in _dirs_to_check:
        for _pkg in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "torch/lib"):
            _dll_dir = os.path.join(_sp, _pkg.replace("/", os.sep))
            if os.path.isdir(_dll_dir):
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(_dll_dir)
                    except Exception:
                        pass
                if _dll_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

import anyio
import requests
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

from runtime_config import RuntimeConfigError, RuntimeConfigStore

# ---------------------------------------------------------------------------
# Logging & Polling Filter (Silences repetitive /api/status & /api/last-call spam)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scamshield-server")


class EndpointFilter(logging.Filter):
    """Filter out frequent status and polling requests from access logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("/api/status" in msg or "/api/last-call" in msg)


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# ---------------------------------------------------------------------------
# Directories & Persistent Storage Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
CALLS_FILE = os.path.join(DATA_DIR, "calls.json")
LAST_CALL_FILE = os.path.join(DATA_DIR, "last_call.json")
TEST_SAMPLES_DIR = os.path.join(BASE_DIR, "test_samples")
RUNTIME_CONFIG_STORE = RuntimeConfigStore(os.path.join(DATA_DIR, "runtime_config.json"))
config_lock = threading.RLock()

os.makedirs(RECORDINGS_DIR, exist_ok=True)
db_lock = threading.Lock()

if not os.path.exists(CALLS_FILE):
    with open(CALLS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

SMS_FILE = os.path.join(DATA_DIR, "sms_records.json")
if not os.path.exists(SMS_FILE):
    with open(SMS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# ---------------------------------------------------------------------------
# Environment & Secret Management (.env)
# ---------------------------------------------------------------------------
def _load_env_secrets():
    """Load variables from .env file into os.environ without logging or printing secrets."""
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_secrets()
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except Exception:
    pass

def _load_dashboard_config():
    """Saved dashboard choices override deployment defaults, never storage paths."""
    try:
        saved = RUNTIME_CONFIG_STORE.load()
    except RuntimeConfigError:
        logger.warning("[Config] Saved dashboard settings could not be loaded; using deployment defaults.")
        return
    os.environ.update(saved)
    if saved:
        logger.info("[Config] Loaded saved dashboard settings from persistent storage.")


_load_dashboard_config()

# ---------------------------------------------------------------------------
# Server Runtime State & Settings (Parity with CLI Options)
# ---------------------------------------------------------------------------
DETECTION_MODE = os.getenv("DETECTION_MODE", "all").lower()

# STT Provider Options: "whisper", "nemotron", "sarvam"
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper").lower().strip()
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium").lower().strip()
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto").lower().strip()

# Sarvam AI STT Settings
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v4").strip()
SARVAM_MODE = os.getenv("SARVAM_MODE", "transcribe").strip()
SARVAM_LANGUAGE = os.getenv("SARVAM_LANGUAGE", "auto").strip()

# NVIDIA Nemotron ASR Settings (Local In-Process via NeMo toolkit)
NVIDIA_ASR_MODEL = os.getenv("NVIDIA_ASR_MODEL", "nvidia/parakeet-ctc-0.6b").strip()
NVIDIA_ASR_ENDPOINT = os.getenv("NVIDIA_ASR_ENDPOINT", "local_in_process").strip()

# Dynamic Model Operational State ("loading" | "hot" | "error")
MODEL_STATE: Dict = {
    "status": "loading",
    "stt": "loading",
    "llm": "loading",
    "llm_error": None,
    "device": "cpu",
    "last_warmed_at": None,
}
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5")
ACTIVE_CONNECTIONS = 0

# Persistent Representation of the Last Call Data
LAST_CALL_DATA: Dict = {
    "call_id": None,
    "status": "idle",
    "start_time": None,
    "duration_seconds": 0.0,
    "current_risk_score": 0,
    "peak_risk_score": 0,
    "avg_risk_score": 0.0,
    "raw_risk_score": 0,
    "classification": "IDLE",
    "reason": "Waiting for incoming call stream...",
    "flags": [],
    "chunks_count": 0,
    "latest_transcript": "",
    "rolling_window": [],
    "engine_breakdown": {},
    "audio_sha256": None,
    "hash_algorithm": "SHA-256",
    "tamper_status": "none",
    "tamper_verified": False,
    "is_partial": False,
    "latency_ms": 0,
    "stt_provider": STT_PROVIDER,
}


def compute_file_sha256(filepath: str) -> Optional[str]:
    """Compute cryptographic SHA-256 hash of a recorded audio file for tamper detection."""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.error(f"Error computing SHA-256 for {filepath}: {e}")
        return None


def verify_call_audio_integrity(call_id: str) -> dict:
    """Verify the audio file on disk against the recorded hash to detect tampering."""
    wav_path = os.path.join(RECORDINGS_DIR, f"{call_id}.wav")
    calls = load_saved_calls()
    call_record = next((c for c in calls if c.get("call_id") == call_id), None)

    if not os.path.exists(wav_path):
        return {
            "call_id": call_id,
            "status": "missing_audio",
            "verified": False,
            "tamper_detected": False,
            "message": "Audio file not found on disk",
        }

    current_hash = compute_file_sha256(wav_path)
    file_size = os.path.getsize(wav_path)
    recorded_hash = call_record.get("audio_sha256") if call_record else None

    # If recorded hash wasn't stored previously, set it now
    if call_record and not recorded_hash and current_hash:
        call_record["audio_sha256"] = current_hash
        call_record["hash_algorithm"] = "SHA-256"
        call_record["tamper_status"] = "verified"
        call_record["tamper_verified"] = True
        save_call_record(call_record)
        recorded_hash = current_hash

    is_verified = (recorded_hash is not None and current_hash == recorded_hash)
    tamper_detected = (recorded_hash is not None and current_hash != recorded_hash)

    return {
        "call_id": call_id,
        "algorithm": "SHA-256",
        "recorded_sha256": recorded_hash,
        "current_sha256": current_hash,
        "verified": is_verified,
        "tamper_detected": tamper_detected,
        "tamper_status": "tampered" if tamper_detected else ("verified" if is_verified else "unhashed"),
        "file_size_bytes": file_size,
        "message": "File integrity intact. Recording has not been tampered with since creation." if is_verified else ("ALERT: Audio file hash mismatch! Possible tampering detected." if tamper_detected else "Audio recorded.")
    }


def load_saved_calls() -> List[dict]:
    with db_lock:
        if not os.path.exists(CALLS_FILE):
            return []
        try:
            with open(CALLS_FILE, "r", encoding="utf-8") as f:
                calls = json.load(f)
            dirty = False
            for c in calls:
                if c.get("audio_file") and not c.get("audio_sha256"):
                    wav_p = os.path.join(RECORDINGS_DIR, c["audio_file"])
                    sha = compute_file_sha256(wav_p)
                    if sha:
                        c["audio_sha256"] = sha
                        c["hash_algorithm"] = "SHA-256"
                        c["tamper_status"] = "verified"
                        c["tamper_verified"] = True
                        dirty = True
            if dirty:
                try:
                    tmp_fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                        json.dump(calls, f, indent=2)
                    os.replace(tmp_path, CALLS_FILE)
                except Exception:
                    pass
            return calls
        except Exception as e:
            logger.error(f"Failed to load calls DB: {e}")
            return []


def save_last_call_data():
    """Persist LAST_CALL_DATA so dashboard restores state upon page refresh."""
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(LAST_CALL_DATA, f, indent=2)
        os.replace(tmp_path, LAST_CALL_FILE)
    except Exception as e:
        logger.warning(f"Could not persist last_call.json: {e}")


def init_last_call_data():
    """Restore last call data from disk on startup."""
    global LAST_CALL_DATA
    if os.path.exists(LAST_CALL_FILE):
        try:
            with open(LAST_CALL_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict) and saved.get("call_id"):
                    LAST_CALL_DATA.update(saved)
                    logger.info(f"[State] Restored last call data: {LAST_CALL_DATA.get('call_id')}")
                    return
        except Exception:
            pass

    # If no last_call.json, hydrate from calls.json
    saved_calls = load_saved_calls()
    if saved_calls:
        recent = saved_calls[0]
        LAST_CALL_DATA.update({
            "call_id": recent.get("call_id"),
            "status": "completed",
            "start_time": recent.get("formatted_time", recent.get("timestamp")),
            "duration_seconds": recent.get("duration_seconds", 0.0),
            "current_risk_score": recent.get("final_risk_score", 0),
            "peak_risk_score": recent.get("peak_risk_score", 0),
            "avg_risk_score": recent.get("avg_risk_score", 0.0),
            "raw_risk_score": recent.get("raw_risk_score", 0),
            "classification": recent.get("classification", "LEGITIMATE"),
            "reason": recent.get("reason", ""),
            "flags": recent.get("flags", []),
            "chunks_count": recent.get("chunks_count", 0),
            "latest_transcript": recent.get("transcript", "")[-200:],
            "rolling_window": [s.get("text", "") for s in recent.get("segments", [])][-10:] if recent.get("segments") else [recent.get("transcript", "")],
            "engine_breakdown": recent.get("engine_breakdown", {}),
            "audio_sha256": recent.get("audio_sha256"),
            "hash_algorithm": recent.get("hash_algorithm", "SHA-256"),
            "tamper_status": recent.get("tamper_status", "verified" if recent.get("audio_sha256") else "none"),
            "tamper_verified": recent.get("tamper_verified", bool(recent.get("audio_sha256"))),
        })


init_last_call_data()

# ---------------------------------------------------------------------------
# Load Detection Engines (Rule-based & XGBoost)
# ---------------------------------------------------------------------------
_rulebased_score = None
_xgboost_score = None

for search_path in [
    BASE_DIR,
    os.path.join(BASE_DIR, "Advanced Versions", "Advanced Versions", "New version - XGBOOST - RULE")
]:
    if search_path not in sys.path and os.path.isdir(search_path):
        sys.path.insert(0, search_path)

try:
    from rulebased.engine import score_transcript as _rulebased_score
    logger.info("[Engine] Rule-based heuristic engine loaded successfully.")
except Exception as e:
    logger.warning(f"[Engine] Rule-based engine unavailable: {e}")

try:
    from xgboost_model.predict import score_transcript as _xgboost_score
    logger.info("[Engine] XGBoost machine-learning engine loaded successfully.")
except Exception as e:
    logger.warning(f"[Engine] XGBoost engine unavailable: {e}")

# ---------------------------------------------------------------------------
# Speech-To-Text Model Loader (Faster-Whisper with CUDA/CPU Graceful Fallback)
# ---------------------------------------------------------------------------
_whisper_model: Optional[WhisperModel] = None
_device_used = "cpu"
# Reload holds this lock while calling the shared model loader.
model_lock = threading.RLock()
_local_asr_lock = threading.RLock()


def get_model():
    global _whisper_model, _device_used, MODEL_STATE
    with model_lock:
        if _whisper_model is None:
            try:
                logger.info(f"[Whisper] Attempting to load '{WHISPER_MODEL_SIZE}' on CUDA (float16)...")
                _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
                _device_used = "cuda"
                logger.info(f"[Whisper] Loaded '{WHISPER_MODEL_SIZE}' successfully on CUDA GPU.")
            except Exception as e:
                logger.warning(f"[Whisper] CUDA initialization failed ({e}). Falling back to CPU (int8)...")
                _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
                _device_used = "cpu"
                logger.info(f"[Whisper] Loaded '{WHISPER_MODEL_SIZE}' successfully on CPU.")
            MODEL_STATE["device"] = _device_used
    return _whisper_model


def reload_whisper_model(new_size: str):
    global _whisper_model, WHISPER_MODEL_SIZE, MODEL_STATE
    with model_lock:
        MODEL_STATE["status"] = "loading"
        MODEL_STATE["stt"] = "loading"
        WHISPER_MODEL_SIZE = new_size.lower()
        _whisper_model = None
        get_model()
        MODEL_STATE["stt"] = "hot"
        MODEL_STATE["status"] = "hot"
        MODEL_STATE["device"] = _device_used
    logger.info(f"[Whisper] Reloaded STT model to '{WHISPER_MODEL_SIZE}' on {_device_used}.")


# ---------------------------------------------------------------------------
# System Prompts & Heuristic Patterns
# ---------------------------------------------------------------------------
GREETINGS_AND_FILLERS = {
    "hello", "hi", "hey", "good", "morning", "afternoon", "evening",
    "can", "you", "hear", "me", "yes", "no", "yeah", "nope", "who", "is",
    "this", "speaking", "thanks", "thank", "bye", "goodbye", "okay", "ok",
    "am", "i", "audible", "sir", "maam", "madam", "call", "calling"
}

DEFAULT_AUDIO_SYSTEM = (
    "You are a strict, highly accurate phone scam detector.\n"
    "Given a transcribed phone call, score how likely it is a scam from 0 (legitimate) to 100 (scam).\n\n"
    "CRITICAL RULES FOR LEGITIMATE CALLS (Score 0):\n"
    "1. IT Support, Tech Support, Customer Service (e.g., office printer issues, computer help, network troubleshooting, package delivery, routine business calls) are LEGITIMATE calls. Score 0.\n"
    "2. Greetings ('hello', 'hi', 'good morning'), small talk, silence, or garbled/mis-transcribed background words (e.g. isolated words like 'monies', 'wife') MUST be scored 0 unless an explicit scam request is made.\n"
    "3. Simply mentioning 'money' or 'payment' in conversation is NOT a scam unless the caller explicitly demands/requests the listener to send money, transfer funds, pay upfront fees, or provide credit card/bank details.\n\n"
    "REQUIREMENTS FOR SCAM WARNING (Score >= 60):\n"
    "Only flag as a scam if at least ONE explicit scam tactic is present:\n"
    "- Demanding OTP / one-time passwords / verification codes\n"
    "- Demanding wire transfers, UPI payments, gift cards, or crypto\n"
    "- Impersonating Police, CBI, Cyber Cell, Bank Security, or Tax Authorities threatening arrest/account freezing\n"
    "- Instructing user to install remote access tools (AnyDesk, TeamViewer, QuickSupport)\n\n"
    'Reply ONLY in JSON format: {"risk_score": int, "reason": str, "flags": [str]}'
)

DEFAULT_SMS_SYSTEM = (
    "You detect smishing (SMS phishing). Given an SMS body and its sender type, "
    "score how likely it is a scam from 0 to 100. "
    "Sender types: 'dlt_registered' (a registered business sender ID, e.g. VM-HDFCBK), "
    "'plain_number' (a normal phone number), 'unknown' (anything else). "
    "Urgent/OTP/payment language from a 'plain_number' or 'unknown' sender is "
    "significantly more suspicious than identical language from a 'dlt_registered' sender. "
    'Reply with JSON only: {"risk_score": int, "reason": str, "flags": [str]}'
)

# Prompts are independently configurable at runtime. Keep the defaults in
# code so a missing/blank deployment setting remains safe and deterministic.
AUDIO_SYSTEM = os.getenv("AUDIO_SYSTEM_PROMPT", DEFAULT_AUDIO_SYSTEM).strip() or DEFAULT_AUDIO_SYSTEM
SMS_SYSTEM = os.getenv("SMS_SYSTEM_PROMPT", DEFAULT_SMS_SYSTEM).strip() or DEFAULT_SMS_SYSTEM

SCAM_INDICATOR_PATTERNS = [
    # 1. OTP Request
    r"\b(?:share|send|tell|give|enter|read back|what.{0,10}is)\b.{0,30}\botp\b",
    r"\botp\s+(?:batao|bhej|de)\b", r"\bshare\s+otp\b", r"\bsend\s+otp\b", r"\benter\s+(?:the\s+)?otp\b",
    # 2. Remote Access Tools
    r"\b(?:anydesk|teamviewer|ultraviewer|quicksupport|screen\s*share|remote\s*access|remote\s*desktop)\b",
    # 3. Impersonation (Bank, Police, Government, Support)
    r"\b(?:this is|calling from|i am from|i am calling from|officer|inspector)\b.{0,50}\b(?:bank|police|cyber crime|cyber cell|government|rbi|sbi|hdfc|icici|axis|kotak|tax department|customs)\b",
    # 4. Money Transfer Request
    r"\b(?:transfer|send|deposit|move|wire|remit)\s+(?:the\s+)?(?:money|funds|amount|payment|rupees|rs)\b",
    r"\baccount\s+(?:number|details|transfer)\b",
    # 5. Upfront Fee Request
    r"\b(?:registration|processing|service|documentation|activation|cancellation|crossing)\s+(?:fee|charge|cost|amount)\b",
    r"\b(?:upfront|advance|deposit)\s+(?:fee|charge|payment|amount)\b",
    # 6. Sensitive Financial Info
    r"\b(?:share|send|tell|give|enter|provide|confirm)\b.{0,30}\b(?:pin|cvv|card\s*number|password|netbanking|aadhaar|pan|upi)\b",
    r"\bcard\s*number\b", r"\bcvv\b", r"\bnet\s*banking\s*password\b",
    # 7. Account Threat / Legal Action
    r"\baccount\s+(?:will be|has been|is)\s+(?:closed|blocked|frozen|deactivated|suspended|locked|seized)\b",
    r"\b(?:arrest|court|police|legal)\s+(?:warrant|notice|case|action|complaint)\b",
    r"\b(?:aadhaar|pan|sim|phone|mobile)\s+(?:will be|has been|is)\s+(?:blocked|deactivated|suspended)\b",
    # 8. Urgency / Coercion
    r"\b(?:urgent|immediately|asap|right\s+away|act\s+now|act\s+fast|hurry)\b",
    r"\b(?:last|final)\s+(?:warning|chance|notice|call)\b",
    # 9. Investment / Guaranteed Returns
    r"\b(?:invest|investment|trading|stock)\s+(?:opportunity|scheme|returns?|profit)\b",
    r"\b(?:guaranteed|assured|fixed)\s+(?:returns?|profit|income|money)\b",
    # 10. Prize / Lottery Scam
    r"\b(?:congratulations|congrats|you(?:'ve| have)\s+(?:won|selected|chosen))\b",
    r"\b(?:won|winning|prize|award|lottery|lucky\s*draw)\b",
    # 11. Fake Job / Recruitment Scam
    r"\b(?:work[\s-]+from[\s-]+home|wfh|data\s+entry|typing\s+job|part[\s-]+time\s+job)\b",
    # 12. Family Emergency / Hospital Bail Scam
    r"\b(?:family\s+member|relative|son|daughter|husband|wife|brother|sister|father|mother)\s+(?:is\s+)?(?:in\s+)?(?:trouble|danger|hospital|jail|custody)\b",
    r"\b(?:bail|hospital|medical|emergency|surgery|accident)\s+(?:money|fee|charge|payment|fund)\b",
]


def clean_and_deduplicate_transcript(text: str) -> str:
    """Deduplicate consecutive repeated words/phrases from Whisper transcript."""
    if not text:
        return ""
    words = text.split()
    deduped = []
    for w in words:
        clean_w = re.sub(r"[^\w]", "", w.lower())
        if not deduped:
            deduped.append(w)
        else:
            prev_clean = re.sub(r"[^\w]", "", deduped[-1].lower())
            if clean_w and clean_w == prev_clean:
                continue
            deduped.append(w)
    return " ".join(deduped)


def is_insufficient_or_greeting_only(text: str) -> bool:
    cleaned = clean_and_deduplicate_transcript(text)
    words = [re.sub(r"[^\w]", "", w.lower()) for w in cleaned.split() if re.sub(r"[^\w]", "", w.lower())]
    if len(words) < 3:
        return True
    non_greeting_words = [w for w in words if w not in GREETINGS_AND_FILLERS]
    if len(set(non_greeting_words)) < 3 and len(words) < 8:
        return True
    return False


def has_scam_indicator(text: str) -> bool:
    if not text:
        return False
    if _rulebased_score:
        try:
            res = _rulebased_score(text)
            if res.get("risk_score", 0) > 0:
                return True
        except Exception:
            pass
    lower = text.lower()
    for pat in SCAM_INDICATOR_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def split_into_sentences(text: str) -> List[str]:
    """Split transcript text or paragraph into individual clean sentences.
    Prevents transcripts from rendering as single unformatted paragraphs and
    ensures proper sentence-level granularity.
    """
    if not text or not text.strip():
        return []
    cleaned = clean_and_deduplicate_transcript(text.strip())
    if not cleaned:
        return []

    # Protect common abbreviations from causing false sentence splits
    abbrevs = [
        (r"\bMr\.", "Mr<DOT>"),
        (r"\bMrs\.", "Mrs<DOT>"),
        (r"\bMs\.", "Ms<DOT>"),
        (r"\bDr\.", "Dr<DOT>"),
        (r"\bProf\.", "Prof<DOT>"),
        (r"\bRs\.", "Rs<DOT>"),
        (r"\bNo\.", "No<DOT>"),
        (r"\bvs\.", "vs<DOT>"),
        (r"\be\.g\.", "eg<DOT>"),
        (r"\bi\.e\.", "ie<DOT>"),
        (r"\bapprox\.", "approx<DOT>"),
    ]
    protected = cleaned
    for pat, rep in abbrevs:
        protected = re.sub(pat, rep, protected, flags=re.IGNORECASE)

    # Split on terminal punctuation followed by whitespace or newline, or Hindi danda (।)
    parts = re.split(r"(?<=[.?!।\n])\s+", protected)
    sentences = []
    for p in parts:
        restored = p.replace("<DOT>", ".").strip()
        if restored:
            sentences.append(restored)
    return sentences if sentences else [cleaned]


def is_minimum_sentence(text: str) -> bool:
    """Check if text is a minimum complete sentence for LLM analysis.
    Prevents sub-sentence 100ms fragments from prematurely triggering heavy LLM reasoning.
    """
    if not text or not text.strip():
        return False
    cleaned = clean_and_deduplicate_transcript(text.strip())
    words = [w for w in re.findall(r"\b\w+\b", cleaned) if w]
    if not words:
        return False

    has_terminal_punct = bool(re.search(r"[.?!।]\s*$", cleaned))
    # 1. Ends with terminal punctuation and contains at least 3 words (e.g. "Send the OTP.")
    if has_terminal_punct and len(words) >= 3:
        return True

    # 2. Contains explicit scam keywords with at least 3 words
    if has_scam_indicator(cleaned) and len(words) >= 3:
        return True

    # 3. Stream accumulated sufficient words for a coherent sentence without punctuation
    non_fillers = [w for w in words if w.lower() not in GREETINGS_AND_FILLERS]
    if len(words) >= 6 and len(non_fillers) >= 4:
        return True

    return False


class SentenceAccumulator:
    """Accumulates streamed audio ASR transcripts and yields completed minimum sentences.
    Ensures 100ms ASR chunks are parsed sentence by sentence before reaching LLM.
    """
    def __init__(self):
        self.buffer = ""

    def add_transcript(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        cleaned = clean_and_deduplicate_transcript(text.strip())
        if not cleaned:
            return []

        # Combine with active buffer
        if self.buffer:
            combined = f"{self.buffer} {cleaned}".strip()
        else:
            combined = cleaned

        sentences = split_into_sentences(combined)
        completed: List[str] = []

        if len(sentences) > 1:
            for s in sentences[:-1]:
                if is_minimum_sentence(s):
                    completed.append(s)
                elif completed:
                    completed[-1] = f"{completed[-1]} {s}"
                else:
                    completed.append(s)

            tail = sentences[-1]
            if is_minimum_sentence(tail):
                completed.append(tail)
                self.buffer = ""
            else:
                self.buffer = tail
        else:
            candidate = sentences[0]
            if is_minimum_sentence(candidate):
                completed.append(candidate)
                self.buffer = ""
            else:
                self.buffer = candidate

        return completed

    def flush(self) -> Optional[str]:
        """Flush any trailing words in buffer at call termination."""
        leftover = self.buffer.strip()
        self.buffer = ""
        words = leftover.split()
        if len(words) >= 2:
            return leftover
        return None


def parse_json(text: str) -> dict:
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            result = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            result = None
    if not isinstance(result, dict):
        return {"risk_score": 0, "reason": f"Unparsable model output: {text[:200]}", "flags": []}
    try:
        result["risk_score"] = int(result.get("risk_score", 0))
    except (TypeError, ValueError):
        result["risk_score"] = 0
    return result


def _ollama_base_url(host: str) -> str:
    """Build an Ollama base URL from bare hostnames or explicit URLs."""
    value = (host or "localhost").strip().rstrip("/")
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"http://{value}:11434"
    return value


def _ollama_tags(host: Optional[str] = None) -> list[str]:
    response = requests.get(f"{_ollama_base_url(host or OLLAMA_HOST)}/api/tags", timeout=3)
    response.raise_for_status()
    payload = response.json()
    return [str(item.get("name", "")) for item in payload.get("models", []) if item.get("name")]


def ask_ollama(system: str, user_text: str) -> dict:
    ollama_url = f"{_ollama_base_url(OLLAMA_HOST)}/api/chat"
    try:
        resp = requests.post(
            ollama_url,
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "format": "json",
                "options": {
                    "num_predict": 256,
                    "temperature": 0.1,
                },
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
            },
            timeout=12,
        )
        resp.raise_for_status()
        return parse_json(resp.json()["message"]["content"])
    except requests.exceptions.Timeout:
        logger.warning(f"[Ollama] Request timed out after 12s on {ollama_url} ({OLLAMA_MODEL}). Falling back...")
        return {"risk_score": 0, "reason": "Ollama LLM timed out (>12s)", "flags": []}
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 404:
            try:
                available = _ollama_tags()
            except Exception:
                available = []
            suffix = f" Available models: {', '.join(available)}" if available else " Check `ollama list`."
            reason = f"Ollama model '{OLLAMA_MODEL}' was not found (HTTP 404).{suffix}"
            logger.warning(f"[Ollama] {reason}")
            return {"risk_score": 0, "reason": reason, "flags": []}
        logger.warning(f"[Ollama] HTTP error from {ollama_url}: {e}")
        return {"risk_score": 0, "reason": f"Ollama HTTP error: {e}", "flags": []}
    except Exception as e:
        logger.warning(f"[Ollama] Connection error to {ollama_url}: {e}")
        return {"risk_score": 0, "reason": f"Ollama unreachable: {e}", "flags": []}


def _combine_all_results(results: dict, transcript: str) -> dict:
    valid = {k: v for k, v in results.items() if v.get("risk_score") is not None}
    if not valid:
        return {"risk_score": 0, "reason": "All engines failed", "flags": [], "engine_breakdown": results, "mode": "all"}

    best_engine = max(valid, key=lambda k: valid[k]["risk_score"])
    best = valid[best_engine]

    rb_score = valid.get("rulebased", {}).get("risk_score", 0)
    if rb_score == 0 and not has_scam_indicator(transcript) and best["risk_score"] >= 40:
        return {
            "risk_score": 0,
            "reason": "Suppressed LLM hallucination: No explicit scam indicators found in transcript",
            "flags": [],
            "engine_breakdown": results,
            "mode": "all",
            "transcript": transcript,
        }

    all_flags = []
    for engine, res in valid.items():
        prefix = f"[{engine.upper()}]"
        for f in res.get("flags", []):
            all_flags.append(f"{prefix} {f}")

    return {
        "risk_score": best["risk_score"],
        "reason": best["reason"],
        "flags": all_flags,
        "engine_breakdown": {k: v for k, v in results.items()},
        "mode": "all",
        "transcript": transcript,
    }


def is_legitimate_bank_otp(text: str, sender_type: str = "unknown") -> bool:
    """Detect authentic, non-phishing transactional OTP notifications from verified DLT bank headers."""
    # Android clients and older integrations have sent casing/spacing variants.
    # Normalize them before applying the safety exemption so a genuine bank OTP
    # is not sent through the generic OTP-request heuristics.
    normalized_sender_type = re.sub(r"[\s-]+", "_", str(sender_type or "").strip().lower())
    if normalized_sender_type not in {"dlt_registered", "dlt", "registered"}:
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


def _score_with_mode(transcript: str, sender_type: str, system_prompt: str, context: str = "", skip_llm: bool = False) -> dict:
    cleaned = clean_and_deduplicate_transcript(transcript)
    eval_text = context if context else cleaned

    if is_insufficient_or_greeting_only(eval_text):
        return {
            "risk_score": 0,
            "reason": "Normal call greeting or small talk",
            "flags": [],
            "transcript": cleaned,
            "engine_breakdown": {},
        }

    # Authentic DLT bank OTP notification exemption (Standard banking delivery)
    if is_legitimate_bank_otp(eval_text, sender_type):
        return {
            "risk_score": 0,
            "reason": "Legitimate transactional OTP delivery from verified DLT bank",
            "flags": ["DLT verified transactional route", "Legitimate OTP notification"],
            "transcript": cleaned,
            "engine_breakdown": {
                "rulebased": {"risk_score": 0, "reason": "Verified bank OTP notification", "flags": ["DLT verified transactional route"], "source": "rulebased"},
                "xgboost": {"risk_score": 0, "reason": "Verified bank OTP notification", "flags": [], "source": "xgboost"},
                "llm": {"risk_score": 0, "reason": "Verified bank OTP notification", "flags": [], "source": "llm"},
            },
            "mode": DETECTION_MODE,
        }

    mode = DETECTION_MODE
    results = {}

    if mode in ("llm_only", "all") and not skip_llm:
        try:
            if system_prompt == SMS_SYSTEM:
                prompt = f"SMS Sender: {sender_type}\nSMS Message Body:\n{cleaned}"
            else:
                prompt = f"Transcript so far:\n{context}" if context else f"Full call transcript:\n{cleaned}"
            results["llm"] = ask_ollama(system_prompt, prompt)
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")
            results["llm"] = {"risk_score": 0, "reason": f"LLM error: {e}", "flags": []}

    if mode in ("rulebased_only", "all") and _rulebased_score:
        try:
            results["rulebased"] = _rulebased_score(eval_text, sender_type)
        except Exception as e:
            logger.warning(f"Rule-based scoring failed: {e}")
            results["rulebased"] = {"risk_score": 0, "reason": f"Rule error: {e}", "flags": []}
    elif mode in ("rulebased_only", "all") and not _rulebased_score:
        results["rulebased"] = {"risk_score": 0, "reason": "Rule-based engine unavailable", "flags": []}

    if mode in ("xgboost_only", "all") and _xgboost_score:
        try:
            results["xgboost"] = _xgboost_score(eval_text, sender_type)
        except Exception as e:
            logger.warning(f"XGBoost scoring failed: {e}")
            results["xgboost"] = {"risk_score": 0, "reason": f"XGBoost error: {e}", "flags": []}
    elif mode in ("xgboost_only", "all") and not _xgboost_score:
        results["xgboost"] = {"risk_score": 0, "reason": "XGBoost engine unavailable", "flags": []}

    if mode == "llm_only":
        res = dict(results.get("llm", {"risk_score": 0, "reason": "LLM output empty", "flags": []}))
        res["engine_breakdown"] = {k: v for k, v in results.items()}
        return res
    elif mode == "rulebased_only":
        res = dict(results.get("rulebased", {"risk_score": 0, "reason": "Rule engine output empty", "flags": []}))
        res["engine_breakdown"] = {k: v for k, v in results.items()}
        return res
    elif mode == "xgboost_only":
        res = dict(results.get("xgboost", {"risk_score": 0, "reason": "XGBoost engine output empty", "flags": []}))
        res["engine_breakdown"] = {k: v for k, v in results.items()}
        return res
    elif mode == "all":
        return _combine_all_results(results, eval_text)
    else:
        res = dict(results.get("llm", {"risk_score": 0, "reason": f"Unknown mode {mode}", "flags": []}))
        res["engine_breakdown"] = {k: v for k, v in results.items()}
        return res


def _slice_wav_chunks(path: str, max_chunk_sec: float = 25.0) -> List[bytes]:
    """Slice a WAV file into smaller in-memory WAV chunks to respect API length constraints."""
    chunks = []
    try:
        with wave.open(path, "rb") as w:
            params = w.getparams()
            framerate = w.getframerate()
            n_frames = w.getnframes()
            chunk_frames = int(max_chunk_sec * framerate)
            if n_frames <= chunk_frames:
                with open(path, "rb") as f:
                    return [f.read()]
            while True:
                frames = w.readframes(chunk_frames)
                if not frames:
                    break
                buf = io.BytesIO()
                with wave.open(buf, "wb") as out_w:
                    out_w.setparams(params)
                    out_w.writeframes(frames)
                chunks.append(buf.getvalue())
    except Exception as e:
        logger.warning(f"Audio chunking failed: {e}; falling back to raw file")
        with open(path, "rb") as f:
            return [f.read()]
    return chunks


def transcribe_whisper(path: str) -> str:
    """Transcribe audio using local Faster-Whisper on CUDA/CPU."""
    with _local_asr_lock:
        model = get_model()
        segments, _ = model.transcribe(
            path,
            language=None if WHISPER_LANGUAGE in ("", "auto") else WHISPER_LANGUAGE,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        raw_text = " ".join(s.text.strip() for s in segments)
        return clean_and_deduplicate_transcript(raw_text)


# ---------------------------------------------------------------------------
# Sarvam AI Realtime Streaming WebSocket Client (saaras:v4 / saaras:v3-realtime)
# ---------------------------------------------------------------------------
class AsyncSarvamStreamingClient:
    """Asynchronous bidirectional WebSocket client for Sarvam AI Realtime Speech-to-Text."""

    def __init__(
        self,
        api_key: str,
        model: str = "saaras:v4",
        mode: str = "transcribe",
        language_code: str = "auto",
        sample_rate: int = 16000,
        stream_type: str = "fast",
        endpointing: str = "vad",
        on_partial: Optional[Callable[[str, int, Optional[str]], Any]] = None,
        on_final: Optional[Callable[[str, int, Optional[str]], Any]] = None,
    ):
        self.api_key = api_key
        self.model = model or "saaras:v4"
        self.mode = mode or "transcribe"
        self.language_code = language_code if language_code not in ("auto", "unknown", "") else "auto"
        self.sample_rate = str(sample_rate)
        self.stream_type = stream_type or "fast"
        self.endpointing = endpointing or "vad"
        self.on_partial = on_partial
        self.on_final = on_final
        self.ws = None
        self._rx_task = None
        self._is_connected = False
        self._ready = asyncio.Event()
        self._session_ended = asyncio.Event()
        self.failed = False
        self.last_error = None
        self.partial_count = 0
        self.final_count = 0

    def _fail(self, message: str):
        self.failed = True
        self.last_error = message
        self._is_connected = False
        self._ready.set()
        logger.warning(f"[Sarvam Realtime WS] {message}")

    async def connect(self) -> bool:
        try:
            import websockets
        except ImportError:
            logger.error("[Sarvam Realtime WS] 'websockets' library is not installed.")
            return False

        query = {
            "model": self.model,
            "mode": self.mode,
            "language_code": self.language_code,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
            "stream_type": self.stream_type,
            "endpointing": self.endpointing,
        }
        qs = urllib.parse.urlencode(query)
        url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{qs}"
        headers = {"Api-Subscription-Key": self.api_key}

        try:
            connect_kwargs = {"open_timeout": 10, "close_timeout": 2, "ping_interval": 20, "ping_timeout": 10}
            try:
                self.ws = await websockets.connect(url, additional_headers=headers, **connect_kwargs)
            except TypeError:
                self.ws = await websockets.connect(url, extra_headers=headers, **connect_kwargs)

            self._rx_task = asyncio.create_task(self._listen_loop())
            await asyncio.wait_for(self._ready.wait(), timeout=10)
            if self.failed or not self._is_connected:
                await self.close()
                return False
            logger.info(f"[Sarvam Realtime WS] Ready: {self.model}, {self.language_code}, {self.stream_type}, {self.endpointing}; sending 100ms PCM frames.")
            return True
        except Exception as e:
            self._fail(f"Session setup failed: {e}")
            await self.close()
            return False

    async def send_audio_chunk(self, pcm_bytes: bytes) -> bool:
        if self.failed or not self._is_connected or not self.ws:
            return False
        # Fast mode rejects PCM frames above 16,000 bytes. Mobile WAV messages
        # contain several seconds; split the normalized PCM without altering it.
        frame_bytes = max(2, min(16000, int(self.sample_rate) * 2 // 10))
        try:
            for offset in range(0, len(pcm_bytes), frame_bytes):
                if self.failed or not self._is_connected:
                    return False
                b64 = base64.b64encode(pcm_bytes[offset:offset + frame_bytes]).decode("ascii")
                await self.ws.send(json.dumps({"event": "audio_input", "audio": b64}))
            return True
        except Exception as e:
            self._fail(f"Frame dispatch failed: {e}")
            return False

    async def _listen_loop(self):
        try:
            async for raw in self.ws:
                try:
                    data = json.loads(raw)
                    event = data.get("event")
                    if event == "session.begin":
                        self._is_connected = True
                        self._ready.set()
                    elif event == "session.end":
                        logger.info(f"[Sarvam Realtime WS] Session complete: audio={data.get('audio_duration_s')}s, utterances={data.get('total_utterances')}, partials={self.partial_count}, finals={self.final_count}.")
                        self._session_ended.set()
                        break
                    elif event == "error":
                        # Nonfatal errors can still mean this audio was discarded.
                        self._fail(f"Provider error {data.get('code', 'unknown')}: {data.get('message', '')}")
                        break
                    elif event == "transcript.partial":
                        self.partial_count += 1
                        text = data.get("text", "")
                        idx = data.get("utterance_idx", 0)
                        lang = data.get("language")
                        if self.on_partial and text:
                            res = self.on_partial(text, idx, lang)
                            if asyncio.iscoroutine(res):
                                await res
                    elif event == "transcript.final":
                        self.final_count += 1
                        text = data.get("text", "")
                        idx = data.get("utterance_idx", 0)
                        lang = data.get("language")
                        if self.on_final and text:
                            res = self.on_final(text, idx, lang)
                            if asyncio.iscoroutine(res):
                                await res
                    elif event not in ("vad.speech_start", "vad.speech_end", "config.updated", "pong"):
                        logger.debug(f"[Sarvam Realtime WS] Unhandled event: {event}")
                except Exception as parse_e:
                    self._fail(f"Event processing failed: {parse_e}")
                    break
        except Exception as e:
            self._fail(f"Receiver terminated: {e}")
        finally:
            self._is_connected = False
            if not self._session_ended.is_set() and not self.failed:
                self._fail("Receiver stopped before session.end")
            self._ready.set()

    async def flush(self):
        # The SDK defines flush for manual endpointing only; end finalizes VAD.
        if self.endpointing == "manual" and self._is_connected and self.ws:
            try:
                await self.ws.send(json.dumps({"event": "flush"}))
            except Exception as e:
                self._fail(f"Flush failed: {e}")

    async def close(self) -> bool:
        try:
            if self.ws and self._is_connected and not self.failed and not self._session_ended.is_set():
                async def _end_and_drain():
                    await self.ws.send(json.dumps({"event": "end"}))
                    # Final callbacks run before session.end is consumed.
                    await self._session_ended.wait()
                # Include a backpressured end-send in the finalization deadline.
                await asyncio.wait_for(_end_and_drain(), timeout=5)
        except Exception as e:
            self._fail(f"Finalization failed: {type(e).__name__}: {e}")
        finally:
            self._is_connected = False
            try:
                if self.ws:
                    await self.ws.close()
            except Exception as e:
                self._fail(f"Transport closure failed: {e}")
            finally:
                if self._rx_task:
                    if not self._rx_task.done():
                        self._rx_task.cancel()
                    await asyncio.gather(self._rx_task, return_exceptions=True)
        return self._session_ended.is_set() and not self.failed


def transcribe_sarvam(path: str) -> str:
    """Transcribe audio using Sarvam AI Speech-to-Text API (saaras:v4 / saaras:v3) with graceful Whisper fallback."""
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        logger.warning("[Sarvam AI] No SARVAM_API_KEY configured. Falling back to local Faster-Whisper.")
        return transcribe_whisper(path)

    try:
        # Slice into <= 25-second chunks to respect Sarvam's 30-second API limit
        wav_chunks = _slice_wav_chunks(path, max_chunk_sec=25.0)
        transcripts = []

        # Check if official SDK is available
        sdk_available = False
        try:
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=api_key)
            sdk_available = True
        except Exception:
            sdk_available = False

        lang_code = "unknown" if SARVAM_LANGUAGE.lower() in ("auto", "unknown", "") else SARVAM_LANGUAGE

        for idx, chunk_bytes in enumerate(wav_chunks):
            chunk_text = ""
            # 1. Try SDK first
            if sdk_available:
                try:
                    buf = io.BytesIO(chunk_bytes)
                    res = client.speech_to_text.transcribe(
                        file=buf,
                        model=SARVAM_MODEL,
                        mode=SARVAM_MODE,
                        language_code=lang_code
                    )
                    chunk_text = getattr(res, "transcript", None) or (res.get("transcript") if isinstance(res, dict) else "")
                except Exception as e:
                    logger.warning(f"[Sarvam SDK] Chunk #{idx+1} call failed: {e}; trying direct REST...")

            # 2. REST fallback
            if not chunk_text:
                url = "https://api.sarvam.ai/speech-to-text"
                headers = {"api-subscription-key": api_key}
                files = {"file": ("chunk.wav", io.BytesIO(chunk_bytes), "audio/wav")}
                data = {
                    "model": SARVAM_MODEL,
                    "mode": SARVAM_MODE,
                    "language_code": SARVAM_LANGUAGE
                }
                r = requests.post(url, headers=headers, files=files, data=data, timeout=35)
                if r.status_code != 200:
                    raise RuntimeError(f"Sarvam API error (HTTP {r.status_code}): {r.text}")
                chunk_text = r.json().get("transcript", "")

            if chunk_text:
                transcripts.append(chunk_text.strip())

        full_text = " ".join(transcripts)
        logger.info(f"[Sarvam AI] Transcribed {len(wav_chunks)} segment(s) with {SARVAM_MODEL}: {full_text[:80]}...")
        return clean_and_deduplicate_transcript(full_text)
    except Exception as e:
        logger.warning(f"[Sarvam AI] Transcription failed: {e}. Falling back to local Faster-Whisper to keep stream active.")
        return transcribe_whisper(path)


# ---------------------------------------------------------------------------
# NVIDIA NeMo (Parakeet-CTC 0.6B & Nemotron) In-Process Local ASR Model Loader
# ---------------------------------------------------------------------------
_nemotron_model = None
_current_nemo_model_name = None
_failed_nemo_target = None
_failed_nemo_error = None
nemotron_lock = threading.Lock()


def _release_local_model_memory():
    """Release superseded local ASR models before loading another CUDA model."""
    global _whisper_model, _nemotron_model, _current_nemo_model_name
    _whisper_model = None
    _nemotron_model = None
    _current_nemo_model_name = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        logger.debug(f"[ASR] CUDA cache cleanup skipped: {e}")


def get_nemotron_model(model_name: Optional[str] = None):
    """Load local in-process NVIDIA NeMo ASR model (Parakeet-CTC 0.6B or Nemotron) with CUDA GPU acceleration."""
    global _nemotron_model, _current_nemo_model_name, _failed_nemo_target, _failed_nemo_error
    model_target = model_name or NVIDIA_ASR_MODEL or "nvidia/parakeet-ctc-0.6b"
    with nemotron_lock:
        if _failed_nemo_target == model_target:
            raise RuntimeError(_failed_nemo_error or f"Previous load failed for {model_target}")
        if _nemotron_model is None or _current_nemo_model_name != model_target:
            logger.info(f"[NeMo ASR] Loading local in-process model '{model_target}'...")
            try:
                if _nemotron_model is not None or _whisper_model is not None:
                    _release_local_model_memory()
                import nemo.collections.asr as nemo_asr
                import torch
                from nemo.utils import logging as nemo_logging
                import logging

                # Suppress noisy Lhotse and NeMo dataloader warnings
                nemo_logging.setLevel(logging.ERROR)
                logging.getLogger("lhotse").setLevel(logging.ERROR)
                logging.getLogger("nemo_logger").setLevel(logging.ERROR)

                model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_target)
                if torch.cuda.is_available():
                    try:
                        model = model.cuda()
                        logger.info(f"[NeMo ASR] Model '{model_target}' loaded successfully on CUDA GPU.")
                    except Exception as ce:
                        logger.warning(f"[NeMo ASR] CUDA placement failed ({ce}); running on CPU.")
                else:
                    logger.info(f"[NeMo ASR] Model '{model_target}' loaded on CPU.")
                model.eval()
                try:
                    model.change_decoding_strategy(decoder_type="ctc", decoding_cfg={"strategy": "greedy_batch"})
                except Exception:
                    pass
                _nemotron_model = model
                _current_nemo_model_name = model_target
                _failed_nemo_target = None
                _failed_nemo_error = None
            except Exception as e:
                _failed_nemo_target = model_target
                _failed_nemo_error = str(e)
                logger.error(f"[NeMo ASR] Failed to load local model '{model_target}': {e}")
                raise
    return _nemotron_model


def _prepare_nemotron_audio(path: str):
    """Ensure audio is mono for NeMo FastConformer model. Returns (audio_path, is_temp)."""
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(path)
        if len(data.shape) > 1 and data.shape[1] > 1:
            mono_data = np.mean(data, axis=1)
            fd, tmp_mono = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(tmp_mono, mono_data, sr)
            return tmp_mono, True
    except Exception as e:
        logger.warning(f"[Nemotron ASR] Mono check failed ({e}); using original file.")
    return path, False


def transcribe_nemotron(path: str, model_name: Optional[str] = None) -> str:
    """Transcribe audio locally using in-process NVIDIA Nemotron ASR with graceful Faster-Whisper fallback."""
    clean_audio_path = path
    is_temp = False
    try:
        clean_audio_path, is_temp = _prepare_nemotron_audio(path)
        model = get_nemotron_model(model_name)
        transcripts = model.transcribe([clean_audio_path], verbose=False)
        text = ""
        if transcripts:
            first = transcripts[0]
            if hasattr(first, "text"):
                text = first.text
            elif isinstance(first, list):
                text = " ".join(str(x) for x in first)
            else:
                text = str(first)
        active_name = model_name or _current_nemo_model_name or NVIDIA_ASR_MODEL
        logger.info(f"[NeMo ASR] Local in-process inference ({active_name}): {text[:80]}...")
        return clean_and_deduplicate_transcript(text)
    except Exception as e:
        logger.warning(
            f"[Nemotron ASR] Local in-process transcription failed ({e}). "
            f"Falling back to local Faster-Whisper to keep stream active."
        )
        return transcribe_whisper(path)
    finally:
        if is_temp and os.path.exists(clean_audio_path):
            try:
                os.unlink(clean_audio_path)
            except Exception:
                pass


def transcribe_wav(path: str, provider: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """Central ASR dispatch supporting Faster-Whisper, NVIDIA Nemotron 3.5 ASR, and Sarvam AI."""
    with _local_asr_lock:
        provider = (provider or STT_PROVIDER or "whisper").lower().strip()
        try:
            if provider == "sarvam":
                return transcribe_sarvam(path)
            elif provider in ("parakeet", "nemotron", "nvidia", "nemo"):
                return transcribe_nemotron(path, model_name=model_name)
            else:
                return transcribe_whisper(path)
        except Exception as e:
            logger.error(f"[ASR Dispatch] Provider '{provider}' transcription failed: {e}. Falling back to Faster-Whisper...")
            return transcribe_whisper(path)


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Pack raw linear PCM bytes into a valid RIFF/WAVE container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def inspect_audio_chunk(chunk_bytes: bytes):
    """Read a PCM16 WAV, or return headerless PCM with unspecified metadata."""
    if not chunk_bytes.startswith(b"RIFF"):
        return chunk_bytes, None, None
    try:
        with wave.open(io.BytesIO(chunk_bytes), "rb") as w:
            if w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                raise ValueError("Only uncompressed 16-bit PCM WAV audio is supported")
            pcm = w.readframes(w.getnframes())
            expected_bytes = w.getnframes() * w.getnchannels() * w.getsampwidth()
            if len(pcm) != expected_bytes:
                raise ValueError("Truncated WAV audio payload")
            return pcm, w.getframerate(), w.getnchannels()
    except (wave.Error, EOFError) as e:
        raise ValueError(f"Invalid WAV audio: {e}") from e


def normalize_pcm_to_16k_mono(pcm_bytes: bytes, in_sample_rate: int = 16000, in_channels: int = 1) -> bytes:
    """Normalize raw 16-bit linear PCM audio to 16 kHz mono 16-bit linear PCM."""
    if in_sample_rate <= 0 or in_channels <= 0:
        raise ValueError("Audio sample rate and channel count must be positive")
    if len(pcm_bytes) % (2 * in_channels):
        raise ValueError("Audio payload must contain complete 16-bit PCM frames")
    if not pcm_bytes:
        return b""
    if in_sample_rate == 16000 and in_channels == 1:
        return pcm_bytes

    import math
    import numpy as np
    from scipy.signal import resample_poly

    # Resample in floating point: integer FIR coefficients can erase the signal
    # on older SciPy releases. Keep fractional downmix values until final encoding.
    arr = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    if in_channels > 1:
        arr = arr.reshape(-1, in_channels).mean(axis=1)
    if in_sample_rate != 16000:
        g = math.gcd(16000, in_sample_rate)
        arr = resample_poly(arr, 16000 // g, in_sample_rate // g)

    # FIR overshoot must saturate rather than wrap around the int16 boundaries.
    return np.clip(np.rint(arr), -32768, 32767).astype("<i2").tobytes()


def extract_pcm_from_wav_bytes(chunk_bytes: bytes) -> bytes:
    pcm, _, _ = inspect_audio_chunk(chunk_bytes)
    return pcm


def save_call_record(record: dict):
    with db_lock:
        calls = []
        if os.path.exists(CALLS_FILE):
            try:
                with open(CALLS_FILE, "r", encoding="utf-8") as f:
                    calls = json.load(f)
            except Exception:
                calls = []
        existing_idx = next((i for i, c in enumerate(calls) if c["call_id"] == record["call_id"]), None)
        if existing_idx is not None:
            calls[existing_idx] = record
        else:
            calls.insert(0, record)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(calls, f, indent=2)
        os.replace(tmp_path, CALLS_FILE)


def load_saved_sms() -> List[dict]:
    with db_lock:
        if not os.path.exists(SMS_FILE):
            return []
        try:
            with open(SMS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load SMS DB: {e}")
            return []


def save_sms_record(record: dict) -> dict:
    with db_lock:
        records = []
        if os.path.exists(SMS_FILE):
            try:
                with open(SMS_FILE, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []
        existing_idx = next((i for i, r in enumerate(records) if r.get("id") == record.get("id")), None)
        if existing_idx is not None:
            records[existing_idx] = record
        else:
            records.insert(0, record)
        if len(records) > 500:
            records = records[:500]
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp_path, SMS_FILE)
        except Exception as e:
            logger.error(f"Failed to save SMS record: {e}")
    return record


def delete_saved_sms(sms_id: str) -> bool:
    with db_lock:
        if not os.path.exists(SMS_FILE):
            return False
        try:
            with open(SMS_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
            new_records = [r for r in records if r.get("id") != sms_id]
            if len(new_records) == len(records):
                return False
            tmp_fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(new_records, f, indent=2)
            os.replace(tmp_path, SMS_FILE)
            return True
        except Exception as e:
            logger.error(f"Failed to delete SMS {sms_id}: {e}")
            return False


def clear_all_saved_sms() -> bool:
    with db_lock:
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
            os.replace(tmp_path, SMS_FILE)
            return True
        except Exception as e:
            logger.error(f"Failed to clear SMS DB: {e}")
            return False


# ---------------------------------------------------------------------------
# FastAPI Application & Startup
# ---------------------------------------------------------------------------
app = FastAPI(title="ScamShield AI Server", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_stt_state_lock = threading.RLock()
_stt_warmup_lock = threading.Lock()
_stt_revision = 0


def _stt_target():
    if STT_PROVIDER == "whisper":
        return STT_PROVIDER, WHISPER_MODEL_SIZE
    if STT_PROVIDER in ("parakeet", "nemotron", "nvidia", "nemo"):
        return STT_PROVIDER, NVIDIA_ASR_MODEL
    return STT_PROVIDER, (SARVAM_MODEL, SARVAM_MODE, SARVAM_LANGUAGE, os.getenv("SARVAM_API_KEY", ""))


def _reset_stt_readiness():
    """Called under the state lock; revision also rejects switch-away-and-back results."""
    global _stt_revision
    _stt_revision += 1
    MODEL_STATE.update(status="loading", stt="loading", device=None,
                       inference_ready=False, last_warmed_at=None)
    return _stt_target(), _stt_revision


def _warm_selected_stt(target, revision):
    """Serialize local warmups; running downloads cannot safely be cancelled."""
    global _whisper_model
    with _local_asr_lock, _stt_warmup_lock:
        with _stt_state_lock:
            if revision != _stt_revision or target != _stt_target():
                return
        provider, model_name = target
        device = None
        stt = "hot"
        try:
            if provider == "sarvam":
                if not model_name[-1].strip():
                    raise RuntimeError("SARVAM_API_KEY is not configured")
                # Configuration is not proof of cloud inference readiness; do not bill a probe.
                stt = "configured"
            elif provider in ("whisper", "parakeet", "nemotron", "nvidia", "nemo"):
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"\x00\x00" * 16000)
                buf.seek(0)
                if provider == "whisper":
                    with model_lock:
                        # A previous, superseded load may have populated the shared cache.
                        if globals().get("ACTIVE_CONNECTIONS", 0) == 0 and globals().get("_nemotron_model") is not None:
                            _release_local_model_memory()
                        _whisper_model = None
                        model = get_model()
                else:
                    model = get_nemotron_model(model_name)
                with _stt_state_lock:
                    if revision != _stt_revision or target != _stt_target():
                        MODEL_STATE["device"] = None
                        return
                if provider == "whisper":
                    segments, _ = model.transcribe(buf, beam_size=1, vad_filter=False)
                    list(segments)  # Faster-Whisper executes inference lazily.
                    device = _device_used
                else:
                    # Close before NeMo reopens the file (also required on Windows).
                    fd, path = tempfile.mkstemp(suffix=".wav")
                    try:
                        with os.fdopen(fd, "wb") as audio:
                            audio.write(buf.getvalue())
                        model.transcribe([path], verbose=False)
                    finally:
                        os.unlink(path)
                    device = str(model.device)
            else:
                raise ValueError(f"Unsupported STT provider: {provider}")
        except Exception as e:
            stt = f"error: {e}"
            logger.error(f"[Warmup] STT {provider} failed: {e}")

        with _stt_state_lock:
            if revision != _stt_revision or target != _stt_target():
                # get_model itself writes device; do not leave a stale Whisper device behind.
                MODEL_STATE["device"] = None
                return
            MODEL_STATE.update(
                stt=stt, status="hot" if stt in ("hot", "configured") else "error",
                device=device, inference_ready=stt == "hot",
                last_warmed_at=datetime.now().isoformat(),
            )
            logger.info(f"[Warmup] STT {provider}: {stt} (device={device}).")


async def _warm_stt_background(target, revision):
    await anyio.to_thread.run_sync(_warm_selected_stt, target, revision)


async def warmup_all_models():
    """Warm only the selected STT provider, and independently warm enabled LLM detection."""
    with _stt_state_lock:
        target, revision = _reset_stt_readiness()
    MODEL_STATE["llm"] = "loading" if DETECTION_MODE in ("all", "llm_only") else "offline"

    async def _warm_ollama():
        if DETECTION_MODE not in ("all", "llm_only"):
            return
        await _warm_llm_background()

    await asyncio.gather(_warm_stt_background(target, revision), _warm_ollama())


async def _warm_llm_background(target=None, revision=None):
    with _stt_state_lock:
        if target is None:
            target = (OLLAMA_HOST, OLLAMA_MODEL)
        if revision is None:
            revision = MODEL_STATE.get("llm_revision", 0)
        if (DETECTION_MODE not in ("all", "llm_only")
                or target != (OLLAMA_HOST, OLLAMA_MODEL)
                or revision != MODEL_STATE.get("llm_revision", 0)):
            return
    host, model = target
    logger.info(f"[Warmup] Warming up Ollama LLM '{model}' at {host}...")
    warm_error = None
    try:
        def _warm_llm():
            base = str(host).rstrip("/") if str(host).lower().startswith(("http://", "https://")) else f"http://{host}:11434"
            url = f"{base}/api/generate"
            r = requests.post(
                url,
                json={"model": model, "prompt": "ready", "stream": False, "keep_alive": "2h"},
                timeout=35,
            )
            if r.status_code == 404:
                available = _ollama_tags(host)
                names = ", ".join(available) if available else "none reported"
                raise RuntimeError(f"model '{model}' not found; available models: {names}")
            if r.status_code != 200:
                raise RuntimeError(f"Ollama warmup returned HTTP {r.status_code}")
            return True

        llm_hot = await anyio.to_thread.run_sync(_warm_llm)
    except Exception as e:
        warm_error = str(e)
        logger.warning(f"[Warmup] Ollama LLM '{model}' at {host} not reached: {e}")
        llm_hot = False

    with _stt_state_lock:
        if (DETECTION_MODE not in ("all", "llm_only")
                or target != (OLLAMA_HOST, OLLAMA_MODEL)
                or revision != MODEL_STATE.get("llm_revision", 0)):
            return
        MODEL_STATE["llm"] = "hot" if llm_hot else "offline"
        MODEL_STATE["llm_error"] = warm_error
        if llm_hot:
            logger.info(f"[Warmup] Ollama LLM '{model}' at {host} is HOT and resident in memory.")


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(warmup_all_models())


# ---------------------------------------------------------------------------
# WebSocket Endpoint: /audio-stream (100% Backwards Compatible)
# ---------------------------------------------------------------------------
@app.websocket("/audio-stream")
async def audio_stream(ws: WebSocket):
    """Server v3 Dual-Track Low-Latency Streaming Pipeline.
    Track A: Real-Time Zero-Copy Inference (Sarvam WS / NeMo Parakeet / Faster-Whisper).
    Track B: Forensic In-Memory Buffer -> Post-call SHA-256 Tamper-Proof Audit.
    """
    global ACTIVE_CONNECTIONS, LAST_CALL_DATA
    await ws.accept()
    ACTIVE_CONNECTIONS += 1

    call_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(2).hex()}"
    call_start_time = time.time()
    call_wav_path = os.path.join(RECORDINGS_DIR, f"{call_id}.wav")

    history: List[str] = []
    score_history: List[int] = []
    peak_risk = 0
    all_pcm_chunks: List[bytes] = []
    segments_log: List[dict] = []
    ws_lock = asyncio.Lock()

    active_llm_task: Optional[asyncio.Task] = None
    latest_llm_result: Optional[dict] = None
    stt_transport = "local"
    sentence_buffer = ""

    def _split_sentences(txt: str) -> List[str]:
        if not txt or not txt.strip():
            return []
        import re
        cleaned = clean_and_deduplicate_transcript(txt.strip())
        if not cleaned:
            return []
        abbrevs = [
            (r"\bMr\.", "Mr<DOT>"),
            (r"\bMrs\.", "Mrs<DOT>"),
            (r"\bMs\.", "Ms<DOT>"),
            (r"\bDr\.", "Dr<DOT>"),
            (r"\bProf\.", "Prof<DOT>"),
            (r"\bRs\.", "Rs<DOT>"),
            (r"\bNo\.", "No<DOT>"),
            (r"\bvs\.", "vs<DOT>"),
        ]
        protected = cleaned
        for pat, rep in abbrevs:
            protected = re.sub(pat, rep, protected, flags=re.IGNORECASE)
        parts = re.split(r"(?<=[.?!।\n])\s+", protected)
        sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
        return sentences if sentences else [cleaned]

    def _is_min_sentence(txt: str) -> bool:
        if not txt or not txt.strip():
            return False
        import re
        cleaned = clean_and_deduplicate_transcript(txt.strip())
        words = [w for w in re.findall(r"\b\w+\b", cleaned) if w]
        if not words:
            return False
        has_terminal = bool(re.search(r"[.?!।]\s*$", cleaned))
        if has_terminal and len(words) >= 3:
            return True
        if has_scam_indicator(cleaned) and len(words) >= 3:
            return True
        fillers = {"hello", "hi", "hey", "yes", "no", "ok", "okay", "yeah", "uh", "um", "ah", "sir", "madam"}
        non_fillers = [w for w in words if w.lower() not in fillers]
        if len(words) >= 6 and len(non_fillers) >= 4:
            return True
        return False

    LAST_CALL_DATA = {
        "call_id": call_id,
        "status": "in_progress",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": 0.0,
        "current_risk_score": 0,
        "peak_risk_score": 0,
        "avg_risk_score": 0.0,
        "raw_risk_score": 0,
        "classification": "MONITORING",
        "reason": "Server v3 Realtime Stream initialized. Awaiting audio packets...",
        "flags": [],
        "chunks_count": 0,
        "latest_transcript": "",
        "rolling_window": [],
        "engine_breakdown": {},
        "is_partial": False,
        "latency_ms": 0,
        "stt_provider": STT_PROVIDER,
    }

    last_processed_transcript = ""
    last_processed_is_partial = False

    async def _handle_transcript_event(text: str, is_partial: bool = False, latency_ms: Optional[int] = None):
        nonlocal peak_risk, active_llm_task, latest_llm_result, sentence_buffer, last_processed_transcript, last_processed_is_partial
        if not text or not text.strip():
            return
        cleaned = clean_and_deduplicate_transcript(text.strip())
        if not cleaned:
            return

        # Deduplication check: discard identical repeated partials/finals
        if cleaned == last_processed_transcript:
            if is_partial and last_processed_is_partial:
                return  # Skip identical in-progress partial repeatedly emitted during pauses
            if not is_partial and not last_processed_is_partial:
                return  # Skip identical finalized sentence already processed

        last_processed_transcript = cleaned
        last_processed_is_partial = is_partial

        elapsed = round(time.time() - call_start_time, 1)

        # Tier 1: Real-time fast heuristic scan (<1ms)
        heuristic_triggered = has_scam_indicator(cleaned)
        context = clean_and_deduplicate_transcript("\n".join(history[-10:] + [cleaned]))

        # Scoring can initialize native ML runtimes; never run it on the socket loop.
        result = await anyio.to_thread.run_sync(
            lambda: _score_with_mode(cleaned, "unknown", AUDIO_SYSTEM, context, skip_llm=True)
        )
        result["transcript"] = cleaned
        result["is_partial"] = is_partial
        result["stt_provider"] = STT_PROVIDER
        result["stt_transport"] = stt_transport
        if latency_ms is not None:
            result["latency_ms"] = latency_ms

        # Incorporate active / latest LLM detection into engine breakdown so UI displays live LLM status
        if latest_llm_result:
            result["engine_breakdown"]["llm"] = latest_llm_result
            llm_score = latest_llm_result.get("risk_score", 0)
            if llm_score > result.get("risk_score", 0):
                result["risk_score"] = llm_score
                if latest_llm_result.get("reason"):
                    result["reason"] = latest_llm_result["reason"]
            for f in latest_llm_result.get("flags", []):
                flag_str = f"[LLM] {f}"
                if flag_str not in result.get("flags", []):
                    result.setdefault("flags", []).append(flag_str)

        raw_score = result.get("risk_score", 0)
        if isinstance(raw_score, (int, float)):
            score_history.append(int(raw_score))
            if raw_score > peak_risk:
                peak_risk = int(raw_score)

        if score_history:
            avg_score = sum(score_history) / len(score_history)
            risky_scores = [s for s in score_history if s >= 40]
            if risky_scores:
                active_avg = sum(risky_scores) / len(risky_scores)
                smoothed_score = int(max(raw_score, active_avg, peak_risk * 0.85))
            else:
                smoothed_score = int(raw_score)

            result["raw_risk_score"] = raw_score
            result["avg_risk_score"] = round(avg_score, 1)
            result["peak_risk_score"] = peak_risk
            result["risk_score"] = min(100, max(0, smoothed_score))

        if not is_partial:
            # Parse incoming text into clean sentences (splits paragraphs into separate chunks)
            new_sentences = _split_sentences(cleaned)
            for s in new_sentences:
                history.append(s)
                segments_log.append({
                    "chunk_idx": len(history),
                    "elapsed": elapsed,
                    "text": s,
                    "risk_score": result.get("risk_score", 0),
                    "flags": result.get("flags", []),
                })

            # Sentence Accumulator for LLM: buffers 100ms fragments until a minimum sentence is formed
            if sentence_buffer:
                combined_llm = f"{sentence_buffer} {cleaned}".strip()
            else:
                combined_llm = cleaned

            llm_sentences = _split_sentences(combined_llm)
            llm_ready = False
            if len(llm_sentences) > 1:
                llm_ready = True
                sentence_buffer = llm_sentences[-1] if not _is_min_sentence(llm_sentences[-1]) else ""
            elif _is_min_sentence(combined_llm):
                llm_ready = True
                sentence_buffer = ""
            else:
                sentence_buffer = combined_llm

            # Tier 3: Trigger Non-Blocking Asynchronous LLM Deep Intent Reasoning in background
            # ONLY when a minimum complete sentence is reached!
            if llm_ready and DETECTION_MODE in ("all", "llm_only") and MODEL_STATE.get("llm") != "offline":
                if active_llm_task is None or active_llm_task.done():
                    async def _run_async_llm_turn(ctx_text: str):
                        nonlocal latest_llm_result, peak_risk
                        try:
                            llm_out = await anyio.to_thread.run_sync(ask_ollama, AUDIO_SYSTEM, f"Transcript so far:\n{ctx_text}")
                            if llm_out and isinstance(llm_out, dict):
                                latest_llm_result = llm_out
                                if "engine_breakdown" not in LAST_CALL_DATA:
                                    LAST_CALL_DATA["engine_breakdown"] = {}
                                LAST_CALL_DATA["engine_breakdown"]["llm"] = llm_out
                                llm_score = llm_out.get("risk_score", 0)
                                if llm_score > peak_risk:
                                    peak_risk = int(llm_score)
                                    LAST_CALL_DATA["peak_risk_score"] = peak_risk
                                if llm_score >= 40:
                                    if llm_out.get("reason"):
                                        LAST_CALL_DATA["reason"] = llm_out["reason"]
                                    for f in llm_out.get("flags", []):
                                        flag_str = f"[LLM] {f}"
                                        if flag_str not in LAST_CALL_DATA.get("flags", []):
                                            LAST_CALL_DATA.setdefault("flags", []).append(flag_str)
                                await anyio.to_thread.run_sync(save_last_call_data)
                        except Exception as le:
                            logger.warning(f"[Async LLM] Turn failed: {le}")

                    active_llm_task = asyncio.create_task(_run_async_llm_turn(context))

        LAST_CALL_DATA["duration_seconds"] = elapsed
        LAST_CALL_DATA["chunks_count"] = len(history) if history else 1
        LAST_CALL_DATA["latest_transcript"] = cleaned
        LAST_CALL_DATA["rolling_window"] = history[-10:] if history else [cleaned]
        LAST_CALL_DATA["current_risk_score"] = result.get("risk_score", 0)
        LAST_CALL_DATA["peak_risk_score"] = peak_risk
        LAST_CALL_DATA["raw_risk_score"] = result.get("raw_risk_score", 0)
        LAST_CALL_DATA["avg_risk_score"] = result.get("avg_risk_score", 0.0)
        LAST_CALL_DATA["reason"] = result.get("reason", "")
        LAST_CALL_DATA["flags"] = result.get("flags", [])
        LAST_CALL_DATA["engine_breakdown"] = result.get("engine_breakdown", {})
        LAST_CALL_DATA["is_partial"] = is_partial
        LAST_CALL_DATA["stt_provider"] = STT_PROVIDER
        LAST_CALL_DATA["stt_transport"] = stt_transport
        if latency_ms is not None:
            LAST_CALL_DATA["latency_ms"] = latency_ms
        await anyio.to_thread.run_sync(save_last_call_data)

        async with ws_lock:
            try:
                await ws.send_json(result)
            except Exception:
                pass

    sarvam_client: Optional[AsyncSarvamStreamingClient] = None
    sarvam_stream_started = False
    sarvam_stream_failed = False
    # Pin the provider/model for this call so a settings change does not
    # redirect an active call mid-stream.
    if STT_PROVIDER == "whisper":
        stream_stt_target = (STT_PROVIDER, globals().get("WHISPER_MODEL_SIZE", "medium"))
    elif STT_PROVIDER in ("parakeet", "nemotron", "nvidia", "nemo"):
        stream_stt_target = (STT_PROVIDER, globals().get("NVIDIA_ASR_MODEL", ""))
    else:
        stream_stt_target = (STT_PROVIDER, (globals().get("SARVAM_MODEL", "saaras:v4"), globals().get("SARVAM_MODE", "transcribe"), globals().get("SARVAM_LANGUAGE", "auto"), os.getenv("SARVAM_API_KEY", "")))
    provider = (stream_stt_target[0] or "whisper").lower().strip()
    stream_model_name = stream_stt_target[1] if provider != "sarvam" else None
    stream_sarvam_settings = stream_stt_target[1] if provider == "sarvam" else None

    def _transcribe_pinned(path: str):
        # Keep compatibility with injected/test one-argument transcribers.
        transcriber_code = getattr(transcribe_wav, "__code__", None)
        if transcriber_code is not None and transcriber_code.co_argcount <= 1:
            return transcribe_wav(path)
        return transcribe_wav(path, provider, stream_model_name)
    sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
    last_chunk_recv_time = time.time()

    if provider == "sarvam":
        stt_transport = "sarvam_realtime" if sarvam_key else "batch_fallback"
    if provider == "sarvam" and sarvam_key:
        last_sarvam_partial = ""

        async def on_sarvam_partial(text: str, idx: int, lang: Optional[str]):
            nonlocal last_sarvam_partial
            cleaned_partial = clean_and_deduplicate_transcript(text.strip()) if text else ""
            if cleaned_partial and cleaned_partial == last_sarvam_partial:
                return  # Skip identical streaming partial emitted across consecutive frames
            last_sarvam_partial = cleaned_partial
            lat = max(10, int((time.time() - last_chunk_recv_time) * 1000))
            await _handle_transcript_event(text, is_partial=True, latency_ms=lat)

        async def on_sarvam_final(text: str, idx: int, lang: Optional[str]):
            nonlocal last_sarvam_partial
            last_sarvam_partial = ""  # Reset partial tracking for next utterance
            lat = max(10, int((time.time() - last_chunk_recv_time) * 1000))
            await _handle_transcript_event(text, is_partial=False, latency_ms=lat)

        sarvam_client = AsyncSarvamStreamingClient(
            api_key=sarvam_key,
            model=stream_sarvam_settings[0],
            mode=stream_sarvam_settings[1],
            language_code=stream_sarvam_settings[2],
            sample_rate=16000,
            stream_type="fast",
            endpointing="vad",
            on_partial=on_sarvam_partial,
            on_final=on_sarvam_final,
        )

    client_sample_rate = 16000
    client_channels = 1
    client_encoding = "pcm_s16le"
    local_stt_buffer = bytearray()
    MIN_CHUNK_BYTES = 32000  # 1.0 second of 16kHz 16-bit mono PCM
    # The Android client sends approximately six seconds of audio per message.
    CALL_INACTIVITY_TIMEOUT = float(os.getenv("CALL_INACTIVITY_TIMEOUT", "20.0"))
    INITIAL_AUDIO_WAIT_TIMEOUT = float(os.getenv("INITIAL_AUDIO_WAIT_TIMEOUT", "120.0"))
    incoming = asyncio.Queue(maxsize=256)
    queued_bytes = 0
    max_queued_bytes = 16 * 1024 * 1024
    backlog_exceeded = False

    async def _receive_messages():
        nonlocal queued_bytes, backlog_exceeded
        try:
            while True:
                message = await ws.receive()
                size = len(message.get("bytes") or b"") + len((message.get("text") or "").encode("utf-8"))
                if incoming.full() or queued_bytes + size > max_queued_bytes:
                    backlog_exceeded = True
                    logger.warning(f"[Audio Stream] Inference backlog limit reached for {call_id}")
                    async with ws_lock:
                        await ws.send_json({
                            "type": "error",
                            "error": "Audio processing backlog exceeded; pending audio discarded and call incomplete. Wait for STT readiness before reconnecting",
                        })
                        await ws.close(code=1013, reason="Audio processing backlog exceeded")
                    break
                queued_bytes += size
                incoming.put_nowait((message, size))
                if message.get("type") == "websocket.disconnect" or (message.get("text") or "").strip().upper() == "END":
                    return
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"[Audio Stream] Receive failed for {call_id}: {e}")
        await incoming.put(({"type": "websocket.disconnect"}, 0))

    # Drain protocol messages even while inference is loading/running. Otherwise
    # Uvicorn's bounded receive queue fills and stops reading client ping frames.
    receiver_task = asyncio.create_task(_receive_messages())
    try:
        await anyio.to_thread.run_sync(save_last_call_data)
        if sarvam_client:
            if await sarvam_client.connect():
                sarvam_stream_started = True
            else:
                sarvam_stream_failed = True
                stt_transport = "batch_fallback"
                logger.warning("[Audio Stream] Sarvam realtime setup failed; using per-chunk batch fallback, not realtime.")
                sarvam_client = None
        while True:
            if backlog_exceeded:
                # Do not spend GPU time on an abandoned queue after overload closure.
                break
            # Allow several Android packet intervals before finalizing an idle call.
            recv_timeout = CALL_INACTIVITY_TIMEOUT if (all_pcm_chunks or local_stt_buffer) else INITIAL_AUDIO_WAIT_TIMEOUT
            try:
                msg, message_size = await asyncio.wait_for(incoming.get(), timeout=recv_timeout)
                queued_bytes -= message_size
            except asyncio.TimeoutError:
                if all_pcm_chunks or local_stt_buffer:
                    logger.info(f"[Audio Stream] Inactivity timeout ({CALL_INACTIVITY_TIMEOUT}s without audio). Call {call_id} ended.")
                else:
                    logger.info(f"[Audio Stream] Initial wait timeout ({INITIAL_AUDIO_WAIT_TIMEOUT}s) without audio. Closing call {call_id}.")
                break
            except WebSocketDisconnect:
                break
            except Exception:
                break

            if msg.get("type") == "websocket.disconnect":
                break

            # Handle text messages (Format negotiation, "END", or ping/metadata)
            if "text" in msg:
                txt = (msg["text"] or "").strip()
                if txt.upper() == "END":
                    logger.info(f"[Audio Stream] Client signaled END for call {call_id}")
                    break
                try:
                    meta = json.loads(txt)
                    if isinstance(meta, dict):
                        mtype = str(meta.get("type", "")).lower()
                        if mtype in ("audio_format", "format", "config"):
                            try:
                                sample_rate = int(meta.get("sampleRate", meta.get("sample_rate", client_sample_rate)))
                                channels = int(meta.get("channels", meta.get("channel_count", client_channels)))
                                encoding = str(meta.get("encoding", client_encoding)).lower().strip()
                                if not 8000 <= sample_rate <= 192000 or not 1 <= channels <= 8:
                                    raise ValueError("Supported sample rates: 8000-192000 Hz; channels: 1-8")
                                if encoding != "pcm_s16le":
                                    raise ValueError("Unsupported encoding; send pcm_s16le or PCM16 WAV audio")
                            except (TypeError, ValueError) as e:
                                async with ws_lock:
                                    await ws.send_json({"type": "error", "error": str(e)})
                                break
                            client_sample_rate = sample_rate
                            client_channels = channels
                            client_encoding = encoding
                            logger.info(f"[Audio Stream] Negotiated format for {call_id}: {client_sample_rate}Hz, {client_channels}ch, {client_encoding}")
                            async with ws_lock:
                                try:
                                    await ws.send_json({
                                        "type": "audio_format_ack",
                                        "status": "ready",
                                        "sample_rate": 16000,
                                        "channels": 1,
                                        "encoding": "pcm_s16le"
                                    })
                                except Exception:
                                    pass
                            continue
                except Exception as e:
                    logger.debug(f"[Audio Stream] Text frame ignored: {e}")
                continue

            if "bytes" not in msg:
                continue

            chunk_t0 = time.time()
            last_chunk_recv_time = chunk_t0
            raw_chunk = msg["bytes"]
            if not raw_chunk:
                continue

            try:
                pcm_frames, wav_sr, wav_ch = inspect_audio_chunk(raw_chunk)
                sr = wav_sr or client_sample_rate
                ch = wav_ch or client_channels
                if not 8000 <= sr <= 192000 or not 1 <= ch <= 8:
                    raise ValueError("Unsupported audio sample rate or channel count")
                norm_pcm = normalize_pcm_to_16k_mono(pcm_frames, in_sample_rate=sr, in_channels=ch)
            except ValueError as e:
                logger.warning(f"[Audio Stream] Rejected audio for {call_id}: {e}")
                async with ws_lock:
                    await ws.send_json({"type": "error", "error": str(e)})
                break
            if not norm_pcm:
                continue

            # Track B: Accumulate normalized raw PCM frames in RAM for tamper-proof recording
            all_pcm_chunks.append(norm_pcm)

            # Track A: Hot Inference
            if sarvam_client:
                if await sarvam_client.send_audio_chunk(norm_pcm):
                    continue
                sarvam_stream_failed = True
                stt_transport = "batch_fallback"
                logger.warning("[Audio Stream] Sarvam realtime failed; using per-chunk batch fallback and recovering full audio at call end.")
                await sarvam_client.close()
                sarvam_client = None
            if not sarvam_client:
                # Local zero-copy in-process transcription (NeMo Parakeet / Faster-Whisper)
                # Buffer small telephony packets (20ms-100ms) until target ~1.0s window is reached,
                # or process immediately if client sent large chunk (>= 32000 bytes)
                local_stt_buffer.extend(norm_pcm)
                if len(local_stt_buffer) < MIN_CHUNK_BYTES:
                    continue

                chunk_to_transcribe = bytes(local_stt_buffer)
                local_stt_buffer.clear()

                wav_payload = pcm_to_wav_bytes(chunk_to_transcribe, sample_rate=16000, channels=1, sampwidth=2)
                fd, path = tempfile.mkstemp(suffix=".wav")
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(wav_payload)
                    text = await anyio.to_thread.run_sync(_transcribe_pinned, path)
                except Exception as e:
                    logger.error(f"[Audio Stream] STT failed on audio chunk: {e}. Fallback to Whisper...")
                    try:
                        text = await anyio.to_thread.run_sync(transcribe_whisper, path)
                    except Exception as we:
                        logger.error(f"[Audio Stream] Fallback Whisper failed: {we}")
                        text = ""
                finally:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                        except Exception:
                            pass

                chunk_lat = max(5, int((time.time() - chunk_t0) * 1000))
                if text and text.strip():
                    await _handle_transcript_event(text, is_partial=False, latency_ms=chunk_lat)
                else:
                    LAST_CALL_DATA["latency_ms"] = chunk_lat
                    LAST_CALL_DATA["is_partial"] = False
                    LAST_CALL_DATA["stt_provider"] = STT_PROVIDER
                    async with ws_lock:
                        try:
                            await ws.send_json({
                                "risk_score": peak_risk,
                                "raw_risk_score": 0,
                                "avg_risk_score": round(sum(score_history) / len(score_history), 1) if score_history else 0.0,
                                "peak_risk_score": peak_risk,
                                "transcript": "",
                                "reason": "No speech detected in audio packet",
                                "flags": [],
                                "is_partial": False,
                                "mode": DETECTION_MODE,
                                "engine_breakdown": {},
                                "latency_ms": chunk_lat,
                                "stt_provider": STT_PROVIDER,
                            })
                        except Exception:
                            pass

    except asyncio.CancelledError:
        try:
            await ws.close()
        except Exception:
            pass
        raise
    finally:
        receiver_task.cancel()
        await asyncio.gather(receiver_task, return_exceptions=True)
        ACTIVE_CONNECTIONS = max(0, ACTIVE_CONNECTIONS - 1)
        if sarvam_client:
            try:
                await sarvam_client.flush()
                if not await sarvam_client.close():
                    sarvam_stream_failed = True
            except Exception as e:
                sarvam_stream_failed = True
                logger.warning(f"[Audio Stream] Sarvam finalization failed: {e}")

    # Flush any remaining audio in local_stt_buffer if present
    if not sarvam_client and local_stt_buffer:
        tmp_path = None
        try:
            wav_payload = pcm_to_wav_bytes(bytes(local_stt_buffer), sample_rate=16000, channels=1, sampwidth=2)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(wav_payload)
            leftover_text = await anyio.to_thread.run_sync(_transcribe_pinned, tmp_path)
            if leftover_text and leftover_text.strip():
                await _handle_transcript_event(leftover_text, is_partial=False)
        except Exception as e:
            logger.warning(f"[Audio Stream] Final audio transcription failed for {call_id}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            local_stt_buffer.clear()

    if history or all_pcm_chunks:
        full_transcript = clean_and_deduplicate_transcript("\n".join(history)) if history else ""
        recovery_failed = False
        needs_recovery = (sarvam_stream_started and sarvam_stream_failed) or (not full_transcript and not sarvam_stream_started)
        if needs_recovery and all_pcm_chunks:
            # Replace, do not append to, partial history after a failed stream.
            # A successfully completed silent Sarvam session needs no REST upload.
            logger.warning(f"[Audio Stream] Running full-call batch recovery for {call_id}; realtime_complete={not sarvam_stream_failed}")
            tmp_full = None
            stt_transport = "batch_recovery"
            try:
                tmp_fd, tmp_full = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(tmp_fd, "wb") as wf:
                    with wave.open(wf, "wb") as out_wav:
                        out_wav.setnchannels(1)
                        out_wav.setsampwidth(2)
                        out_wav.setframerate(16000)
                        out_wav.writeframes(b"".join(all_pcm_chunks))
                full_transcript = await anyio.to_thread.run_sync(_transcribe_pinned, tmp_full)
                if full_transcript:
                    recovered_sentences = split_into_sentences(full_transcript)
                    history = recovered_sentences
                    segments_log = [
                        {
                            "chunk_idx": idx + 1,
                            "elapsed": round((idx + 1) * 2.5, 1),
                            "text": s,
                            "risk_score": 0,
                            "flags": [],
                        }
                        for idx, s in enumerate(recovered_sentences)
                    ]
                    LAST_CALL_DATA["rolling_window"] = history[-10:]
                    LAST_CALL_DATA["chunks_count"] = len(history)
                    LAST_CALL_DATA["latest_transcript"] = history[-1] if history else ""
            except Exception as e:
                recovery_failed = True
                logger.warning(f"[Audio Stream] Full-call batch recovery failed for {call_id}: {e}")
            finally:
                if tmp_full and os.path.exists(tmp_full):
                    os.unlink(tmp_full)

        # Wait up to 3.5s for any in-flight async LLM task
        if active_llm_task and not active_llm_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(active_llm_task), timeout=3.5)
            except Exception:
                pass

        try:
            result = await anyio.to_thread.run_sync(_score_with_mode, full_transcript, "unknown", AUDIO_SYSTEM)
            result["transcript"] = full_transcript
            result["audio_incomplete"] = backlog_exceeded or recovery_failed
            result["realtime_failed"] = sarvam_stream_failed
            result["stt_transport"] = stt_transport

            # If final LLM timed out but async LLM previously succeeded, restore the valid LLM breakdown
            if latest_llm_result and (not result.get("engine_breakdown", {}).get("llm") or result["engine_breakdown"]["llm"].get("risk_score", 0) == 0):
                result.setdefault("engine_breakdown", {})["llm"] = latest_llm_result
                if latest_llm_result.get("risk_score", 0) > result.get("risk_score", 0):
                    result["risk_score"] = latest_llm_result["risk_score"]
                    result["reason"] = latest_llm_result.get("reason", result.get("reason", ""))

            if score_history:
                final_raw = result.get("risk_score", 0)
                score_history.append(final_raw)
                result["raw_risk_score"] = final_raw
                result["avg_risk_score"] = round(sum(score_history) / len(score_history), 1)
                result["peak_risk_score"] = max(peak_risk, final_raw)
                if peak_risk >= 40:
                    result["risk_score"] = max(final_raw, peak_risk)

            async with ws_lock:
                try:
                    await ws.send_json(result)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Final analysis error: {e}")

        duration_sec = round(time.time() - call_start_time, 1)
        audio_sha256 = None
        if all_pcm_chunks:
            try:
                with wave.open(call_wav_path, "wb") as out_wav:
                    out_wav.setnchannels(1)
                    out_wav.setsampwidth(2)
                    out_wav.setframerate(16000)
                    out_wav.writeframes(b"".join(all_pcm_chunks))
                audio_sha256 = compute_file_sha256(call_wav_path)
                logger.info(f"[Integrity] Post-call audio SHA-256 computed: {audio_sha256[:16]}... for call {call_id}")
            except Exception as e:
                logger.error(f"Failed to write call audio WAV: {e}")

        final_score = result.get("risk_score", 0)
        classification = "SCAM" if final_score >= 70 or peak_risk >= 70 else "LEGITIMATE"

        call_record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "formatted_time": datetime.now().strftime("%b %d, %Y %H:%M:%S"),
            "duration_seconds": duration_sec,
            "peak_risk_score": max(peak_risk, final_score),
            "final_risk_score": final_score,
            "raw_risk_score": result.get("raw_risk_score", final_score),
            "avg_risk_score": result.get("avg_risk_score", 0.0),
            "classification": classification,
            "reason": result.get("reason", ""),
            "flags": result.get("flags", []),
            "transcript": full_transcript,
            "chunks_count": len(history),
            "audio_file": f"{call_id}.wav" if os.path.exists(call_wav_path) else None,
            "audio_sha256": audio_sha256,
            "hash_algorithm": "SHA-256",
            "tamper_status": "verified" if audio_sha256 else "none",
            "tamper_verified": True if audio_sha256 else False,
            "segments": segments_log,
            "audio_incomplete": backlog_exceeded or recovery_failed,
            "realtime_failed": sarvam_stream_failed,
            "stt_transport": stt_transport,
            "engine_breakdown": result.get("engine_breakdown", {}),
        }
        save_call_record(call_record)

        LAST_CALL_DATA["status"] = "completed"
        LAST_CALL_DATA["audio_incomplete"] = backlog_exceeded or recovery_failed
        LAST_CALL_DATA["realtime_failed"] = sarvam_stream_failed
        LAST_CALL_DATA["stt_transport"] = stt_transport
        LAST_CALL_DATA["classification"] = classification
        LAST_CALL_DATA["current_risk_score"] = final_score
        LAST_CALL_DATA["peak_risk_score"] = max(peak_risk, final_score)
        LAST_CALL_DATA["raw_risk_score"] = result.get("raw_risk_score", final_score)
        LAST_CALL_DATA["avg_risk_score"] = result.get("avg_risk_score", 0.0)
        LAST_CALL_DATA["reason"] = result.get("reason", "")
        LAST_CALL_DATA["flags"] = result.get("flags", [])
        LAST_CALL_DATA["engine_breakdown"] = result.get("engine_breakdown", {})
        LAST_CALL_DATA["duration_seconds"] = duration_sec
        LAST_CALL_DATA["audio_sha256"] = audio_sha256
        LAST_CALL_DATA["hash_algorithm"] = "SHA-256"
        LAST_CALL_DATA["tamper_status"] = "verified" if audio_sha256 else "none"
        LAST_CALL_DATA["tamper_verified"] = True if audio_sha256 else False
        save_last_call_data()

    try:
        await ws.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# WebSocket Endpoint: /sms-check (100% Backwards Compatible)
# ---------------------------------------------------------------------------
@app.websocket("/sms-check")
async def sms_check(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            msg = await ws.receive()
        except Exception:
            break
        if msg.get("type") == "websocket.disconnect":
            break
        if "text" not in msg:
            continue
        sender_type = "unknown"
        body = msg["text"]
        sender = ""
        try:
            payload = json.loads(msg["text"])
            if isinstance(payload, dict):
                # Supports both ScamShield v3 {"sender_type": ..., "body": ...}
                # and Android BCR {"type": "sms", "body": ..., "timestamp": ...}
                body = str(payload.get("body") or payload.get("message") or payload.get("text") or "")
                raw_sender_type = payload.get("sender_type")
                sender = str(payload.get("sender") or payload.get("originating_address") or payload.get("from") or "").strip()

                if raw_sender_type:
                    sender_type = re.sub(r"[\s-]+", "_", str(raw_sender_type).strip().lower())
                    if sender_type in {"dlt", "registered"}:
                        sender_type = "dlt_registered"
                elif sender:
                    # Automatic DLT / sender classification
                    clean_s = re.sub(r"[^a-zA-Z0-9]", "", sender)
                    if re.match(r"^[a-zA-Z]{2}[a-zA-Z0-9]{6}$", clean_s) or (len(clean_s) in (6, 8, 9) and any(c.isalpha() for c in clean_s) and not clean_s.isdigit()):
                        sender_type = "dlt_registered"
                    elif re.match(r"^\+?[0-9]{10,13}$", sender):
                        sender_type = "plain_number"
                    else:
                        sender_type = "unknown"
                else:
                    # Heuristic inference from body
                    if re.search(r"\b(SBI|HDFC|ICICI|AXIS|PNB|BOB|PAYTM|AIRTEL|JIO)\b", body, re.IGNORECASE) and re.search(r"\b(OTP|credited|debited|acct|a/c)\b", body, re.IGNORECASE):
                        sender_type = "dlt_registered"
                    elif re.search(r"https?://|bit\.ly|t\.co|wa\.me|t\.me|\.apk\b", body, re.IGNORECASE):
                        sender_type = "unknown"
                    else:
                        sender_type = "unknown"
        except json.JSONDecodeError:
            pass
        result = await anyio.to_thread.run_sync(_score_with_mode, body, sender_type, SMS_SYSTEM)
        result["sender_type"] = sender_type
        risk_score = result.get("risk_score", 0)
        classification = "CRITICAL SMISHING" if risk_score >= 70 else ("SUSPICIOUS" if risk_score >= 40 else "SAFE")
        sms_id = f"sms_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(2).hex()}"
        sms_record = {
            "id": sms_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sender": sender or "Unknown",
            "sender_type": sender_type,
            "body": body,
            "risk_score": risk_score,
            "classification": classification,
            "reason": result.get("reason", ""),
            "flags": result.get("flags", []),
            "engine_breakdown": result.get("engine_breakdown", {}),
        }
        await anyio.to_thread.run_sync(save_sms_record, sms_record)

        result["id"] = sms_id
        result["sender"] = sender or "Unknown"
        result["classification"] = classification
        try:
            await ws.send_json(result)
        except Exception:
            break
    try:
        await ws.close()
    except Exception:
        pass


@app.get("/api/sms")
async def get_received_sms():
    records = await anyio.to_thread.run_sync(load_saved_sms)
    scams_count = sum(1 for r in records if r.get("risk_score", 0) >= 50)
    suspicious_count = sum(1 for r in records if 30 <= r.get("risk_score", 0) < 50)
    safe_count = sum(1 for r in records if r.get("risk_score", 0) < 30)
    return {
        "status": "ok",
        "total": len(records),
        "scams": scams_count,
        "suspicious": suspicious_count,
        "safe": safe_count,
        "records": records,
    }


@app.post("/api/sms")
async def analyze_and_store_sms(request: Request):
    data = await request.json()
    body = str(data.get("body") or data.get("text") or "").strip()
    sender = str(data.get("sender") or "").strip()
    sender_type = re.sub(r"[\s-]+", "_", str(data.get("sender_type") or "unknown").strip().lower())
    if sender_type in {"dlt", "registered"}:
        sender_type = "dlt_registered"
    if not sender_type or sender_type == "unknown":
        if sender:
            clean_s = re.sub(r"[^a-zA-Z0-9]", "", sender)
            if re.match(r"^[a-zA-Z]{2}[a-zA-Z0-9]{6}$", clean_s) or (len(clean_s) in (6, 8, 9) and any(c.isalpha() for c in clean_s) and not clean_s.isdigit()):
                sender_type = "dlt_registered"
            elif re.match(r"^\+?[0-9]{10,13}$", sender):
                sender_type = "plain_number"
            else:
                sender_type = "unknown"

    result = await anyio.to_thread.run_sync(_score_with_mode, body, sender_type, SMS_SYSTEM)
    risk_score = result.get("risk_score", 0)
    classification = "CRITICAL SMISHING" if risk_score >= 70 else ("SUSPICIOUS" if risk_score >= 40 else "SAFE")
    sms_id = f"sms_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(2).hex()}"
    sms_record = {
        "id": sms_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sender": sender or "Unknown",
        "sender_type": sender_type,
        "body": body,
        "risk_score": risk_score,
        "classification": classification,
        "reason": result.get("reason", ""),
        "flags": result.get("flags", []),
        "engine_breakdown": result.get("engine_breakdown", {}),
    }
    await anyio.to_thread.run_sync(save_sms_record, sms_record)
    return {"status": "ok", "record": sms_record}


@app.delete("/api/sms/{sms_id}")
async def delete_sms_endpoint(sms_id: str):
    success = await anyio.to_thread.run_sync(delete_saved_sms, sms_id)
    if not success:
        raise HTTPException(status_code=404, detail="SMS record not found")
    return {"status": "ok", "message": f"Deleted SMS {sms_id}"}


@app.delete("/api/sms")
async def clear_all_sms_endpoint():
    await anyio.to_thread.run_sync(clear_all_saved_sms)
    return {"status": "ok", "message": "All SMS records cleared"}


# ---------------------------------------------------------------------------
# REST API: Status, Configuration, Calls & Transcripts Management
# ---------------------------------------------------------------------------
@app.get("/api/status")
def get_status():
    ollama_ok = False
    ollama_models = []
    try:
        r = requests.get(f"{_ollama_base_url(OLLAMA_HOST)}/api/tags", timeout=1.5)
        ollama_ok = r.status_code == 200
        if ollama_ok:
            ollama_models = [str(item.get("name", "")) for item in r.json().get("models", []) if item.get("name")]
    except Exception:
        ollama_ok = False

    saved_calls = load_saved_calls()
    scam_count = sum(1 for c in saved_calls if c.get("classification") == "SCAM")

    sarvam_key = os.getenv("SARVAM_API_KEY", "")
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")

    return {
        "status": "online",
        "model_status": MODEL_STATE.get("status", "hot"),
        "stt_status": MODEL_STATE.get("stt", "hot"),
        "llm_status": MODEL_STATE.get("llm", "offline"),
        "model_error": MODEL_STATE.get("stt") if str(MODEL_STATE.get("stt", "")).startswith("error:") else None,
        "detection_mode": DETECTION_MODE,
        "stt_provider": STT_PROVIDER,
        "whisper_model": WHISPER_MODEL_SIZE,
        "whisper_language": WHISPER_LANGUAGE,
        "sarvam_model": SARVAM_MODEL,
        "sarvam_mode": SARVAM_MODE,
        "sarvam_language": SARVAM_LANGUAGE,
        "sarvam_configured": bool(sarvam_key),
        "sarvam_key_hint": f"sk_...{sarvam_key[-4:]}" if len(sarvam_key) >= 8 else ("Configured" if sarvam_key else "Not set"),
        "nvidia_asr_model": NVIDIA_ASR_MODEL,
        "nvidia_asr_endpoint": "local_in_process",
        "nvidia_configured": True,
        "nvidia_is_local": True,
        "device": MODEL_STATE.get("device"),
        "stt_inference_ready": MODEL_STATE.get("inference_ready", False),
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "ollama_connected": ollama_ok,
        "ollama_models": ollama_models,
        "ollama_model_available": OLLAMA_MODEL in ollama_models or f"{OLLAMA_MODEL}:latest" in ollama_models,
        "llm_error": MODEL_STATE.get("llm_error"),
        "active_connections": ACTIVE_CONNECTIONS,
        "total_saved_calls": len(saved_calls),
        "total_scams_detected": scam_count,
        "engines_available": {
            "llm": ollama_ok,
            "xgboost": _xgboost_score is not None,
            "rulebased": _rulebased_score is not None,
        },
    }


class ConfigUpdate(BaseModel):
    mode: Optional[str] = None
    stt_provider: Optional[str] = None
    whisper_model: Optional[str] = None
    sarvam_model: Optional[str] = None
    sarvam_mode: Optional[str] = None
    sarvam_language: Optional[str] = None
    sarvam_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    nvidia_asr_endpoint: Optional[str] = None
    nvidia_asr_model: Optional[str] = None
    ollama_host: Optional[str] = None
    ollama_model: Optional[str] = None
    audio_system_prompt: Optional[str] = None
    sms_system_prompt: Optional[str] = None


def _dashboard_config_updates(cfg: ConfigUpdate) -> dict[str, str]:
    fields = {
        "mode": "DETECTION_MODE", "stt_provider": "STT_PROVIDER",
        "whisper_model": "WHISPER_MODEL", "sarvam_model": "SARVAM_MODEL",
        "sarvam_mode": "SARVAM_MODE", "sarvam_language": "SARVAM_LANGUAGE",
        "sarvam_api_key": "SARVAM_API_KEY", "nvidia_api_key": "NVIDIA_API_KEY",
        "nvidia_asr_endpoint": "NVIDIA_ASR_ENDPOINT", "nvidia_asr_model": "NVIDIA_ASR_MODEL",
        "ollama_host": "OLLAMA_HOST", "ollama_model": "OLLAMA_MODEL",
        "audio_system_prompt": "AUDIO_SYSTEM_PROMPT", "sms_system_prompt": "SMS_SYSTEM_PROMPT",
    }
    updates = {}
    for field, key in fields.items():
        value = getattr(cfg, field, None)
        if value is None:
            continue
        if not isinstance(value, str) or any(char in value for char in ("\n", "\r", "\x00")):
            raise HTTPException(status_code=422, detail=f"Invalid value for {field}")
        value = value.strip()
        if not value:
            if field in ("sarvam_api_key", "nvidia_api_key"):
                continue  # Empty password inputs retain the saved key.
            raise HTTPException(status_code=422, detail=f"{field} must not be empty")
        if field in ("audio_system_prompt", "sms_system_prompt") and len(value) > 16000:
            raise HTTPException(status_code=422, detail=f"{field} must be at most 16000 characters")
        if field in ("mode", "stt_provider", "whisper_model"):
            value = value.lower()
        updates[key] = value

    if "DETECTION_MODE" in updates and updates["DETECTION_MODE"] not in ("all", "llm_only", "xgboost_only", "rulebased_only"):
        raise HTTPException(status_code=422, detail="Unsupported detection mode")
    provider = updates.get("STT_PROVIDER")
    if provider is not None and provider not in ("whisper", "sarvam", "parakeet", "nemotron", "nvidia", "nemo"):
        raise HTTPException(status_code=422, detail="Unsupported STT provider")
    if provider == "parakeet":
        updates.setdefault("NVIDIA_ASR_MODEL", "nvidia/parakeet-ctc-0.6b")
    elif provider == "nemotron":
        updates.setdefault("NVIDIA_ASR_MODEL", "nvidia/nemotron-speech-streaming-en-0.6b")
    return updates


@app.post("/api/config")
def update_config(cfg: ConfigUpdate, background_tasks: BackgroundTasks):
    global DETECTION_MODE, STT_PROVIDER, WHISPER_MODEL_SIZE
    global SARVAM_MODEL, SARVAM_MODE, SARVAM_LANGUAGE
    global NVIDIA_ASR_MODEL, NVIDIA_ASR_ENDPOINT
    global OLLAMA_HOST, OLLAMA_MODEL
    global AUDIO_SYSTEM, SMS_SYSTEM

    updates = _dashboard_config_updates(cfg)
    with config_lock:
        try:
            RUNTIME_CONFIG_STORE.save(updates)
        except RuntimeConfigError:
            logger.error("[Config] Could not persist dashboard settings; runtime settings unchanged.")
            raise HTTPException(status_code=503, detail="Could not save dashboard configuration; no changes were applied.") from None

        with _stt_state_lock:
            previous_stt_target = _stt_target()
            previous_llm_enabled = DETECTION_MODE in ("all", "llm_only")
            previous_llm_target = (OLLAMA_HOST, OLLAMA_MODEL)
            os.environ.update(updates)
            DETECTION_MODE = updates.get("DETECTION_MODE", DETECTION_MODE)
            STT_PROVIDER = updates.get("STT_PROVIDER", STT_PROVIDER)
            WHISPER_MODEL_SIZE = updates.get("WHISPER_MODEL", WHISPER_MODEL_SIZE)
            SARVAM_MODEL = updates.get("SARVAM_MODEL", SARVAM_MODEL)
            SARVAM_MODE = updates.get("SARVAM_MODE", SARVAM_MODE)
            SARVAM_LANGUAGE = updates.get("SARVAM_LANGUAGE", SARVAM_LANGUAGE)
            NVIDIA_ASR_MODEL = updates.get("NVIDIA_ASR_MODEL", NVIDIA_ASR_MODEL)
            NVIDIA_ASR_ENDPOINT = updates.get("NVIDIA_ASR_ENDPOINT", NVIDIA_ASR_ENDPOINT)
            OLLAMA_HOST = updates.get("OLLAMA_HOST", OLLAMA_HOST)
            OLLAMA_MODEL = updates.get("OLLAMA_MODEL", OLLAMA_MODEL)
            if "AUDIO_SYSTEM_PROMPT" in updates:
                AUDIO_SYSTEM = updates["AUDIO_SYSTEM_PROMPT"]
            if "SMS_SYSTEM_PROMPT" in updates:
                SMS_SYSTEM = updates["SMS_SYSTEM_PROMPT"]

            llm_enabled = DETECTION_MODE in ("all", "llm_only")
            llm_target = (OLLAMA_HOST, OLLAMA_MODEL)
            llm_changed = (llm_enabled != previous_llm_enabled or llm_target != previous_llm_target)
            if llm_changed:
                MODEL_STATE["llm_revision"] = MODEL_STATE.get("llm_revision", 0) + 1
            if not llm_enabled:
                MODEL_STATE["llm"] = "offline"
            elif llm_changed:
                MODEL_STATE["llm"] = "loading"
                background_tasks.add_task(_warm_llm_background, llm_target, MODEL_STATE["llm_revision"])

            if _stt_target() != previous_stt_target:
                target, revision = _reset_stt_readiness()
                background_tasks.add_task(_warm_stt_background, target, revision)

        sarvam_key = os.getenv("SARVAM_API_KEY", "")
        logger.info(f"[Config] Saved and applied dashboard configuration: STT={STT_PROVIDER}, Mode={DETECTION_MODE}")
        return {
            "success": True,
            "persisted": True,
            "detection_mode": DETECTION_MODE,
            "stt_provider": STT_PROVIDER,
            "whisper_model": WHISPER_MODEL_SIZE,
            "sarvam_model": SARVAM_MODEL,
            "sarvam_mode": SARVAM_MODE,
            "sarvam_language": SARVAM_LANGUAGE,
            "sarvam_configured": bool(sarvam_key),
            "sarvam_key_hint": f"sk_...{sarvam_key[-4:]}" if len(sarvam_key) >= 8 else ("Configured" if sarvam_key else "Not set"),
            "nvidia_asr_model": NVIDIA_ASR_MODEL,
            "nvidia_asr_endpoint": "local_in_process",
            "nvidia_configured": True,
            "nvidia_is_local": True,
            "ollama_host": OLLAMA_HOST,
            "ollama_model": OLLAMA_MODEL,
            "audio_system_prompt": globals().get("AUDIO_SYSTEM", ""),
            "sms_system_prompt": globals().get("SMS_SYSTEM", ""),
        }


@app.get("/api/last-call")
def get_last_call():
    return LAST_CALL_DATA


@app.post("/api/last-call/clear")
def clear_last_call():
    """Clear dashboard monitor to idle state without deleting historical saved calls."""
    global LAST_CALL_DATA
    LAST_CALL_DATA = {
        "call_id": None,
        "status": "idle",
        "start_time": None,
        "duration_seconds": 0.0,
        "current_risk_score": 0,
        "peak_risk_score": 0,
        "avg_risk_score": 0.0,
        "raw_risk_score": 0,
        "classification": "IDLE",
        "reason": "Dashboard cleared. Ready for next call.",
        "flags": [],
        "chunks_count": 0,
        "latest_transcript": "",
        "rolling_window": [],
        "engine_breakdown": {},
        "audio_sha256": None,
        "hash_algorithm": "SHA-256",
        "tamper_status": "none",
        "tamper_verified": False,
        "is_partial": False,
        "latency_ms": 0,
        "stt_provider": STT_PROVIDER,
    }
    save_last_call_data()
    return {"success": True, "message": "Dashboard cleared"}


@app.get("/api/calls")
def get_calls():
    return load_saved_calls()


@app.get("/api/calls/{call_id}")
def get_call_details(call_id: str):
    calls = load_saved_calls()
    for c in calls:
        if c["call_id"] == call_id:
            return c
    raise HTTPException(status_code=404, detail="Call record not found")


@app.get("/api/calls/{call_id}/verify-hash")
def verify_call_hash(call_id: str):
    """Cryptographically verify that the call recording audio file has not been altered since recording."""
    return verify_call_audio_integrity(call_id)


@app.get("/api/calls/{call_id}/audio")
def get_call_audio(call_id: str):
    wav_path = os.path.join(RECORDINGS_DIR, f"{call_id}.wav")
    if not os.path.exists(wav_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(wav_path, media_type="audio/wav", filename=f"{call_id}.wav")


@app.delete("/api/calls/{call_id}")
def delete_call(call_id: str):
    wav_path = os.path.join(RECORDINGS_DIR, f"{call_id}.wav")
    if os.path.exists(wav_path):
        try:
            os.unlink(wav_path)
        except Exception:
            pass
    with db_lock:
        calls = load_saved_calls()
        calls = [c for c in calls if c["call_id"] != call_id]
        with open(CALLS_FILE, "w", encoding="utf-8") as f:
            json.dump(calls, f, indent=2)
    return {"success": True, "call_id": call_id}


class SmishingRequest(BaseModel):
    body: str
    sender_type: Optional[str] = "unknown"


@app.post("/api/test/sms")
async def test_sms(req: SmishingRequest):
    logger.info(f"[Simulator] Running SMS test: '{req.body[:50]}...' (sender_type={req.sender_type})")
    result = await anyio.to_thread.run_sync(_score_with_mode, req.body, req.sender_type or "unknown", SMS_SYSTEM)
    result["body"] = req.body
    result["sender_type"] = req.sender_type or "unknown"
    return result


@app.post("/api/test/sample/{sample_name}")
async def test_sample_audio(sample_name: str, update_dashboard: bool = True):
    """Run real STT and multi-engine detection on a bundled sample call."""
    sample_file = os.path.join(TEST_SAMPLES_DIR, f"{sample_name}.wav")
    if not os.path.exists(sample_file):
        alias_map = {'scam_call': 'scam_sample.wav', 'normal_call': 'normal_sample.wav'}
        if sample_name in alias_map:
            sample_file = os.path.join(TEST_SAMPLES_DIR, alias_map[sample_name])
    if not os.path.exists(sample_file):
        raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found.")

    logger.info(f"[Simulator] Running bundled test sample: '{sample_name}.wav'")
    t0 = time.time()
    text = await anyio.to_thread.run_sync(transcribe_wav, sample_file)
    transcribe_time = round(time.time() - t0, 2)
    logger.info(f"[Simulator] Transcription ({transcribe_time}s): \"{text[:120]}...\"")

    t1 = time.time()
    result = await anyio.to_thread.run_sync(_score_with_mode, text, "unknown", AUDIO_SYSTEM)
    score_time = round(time.time() - t1, 2)

    result["transcript"] = text
    result["filename"] = f"{sample_name}.wav"
    result["transcribe_time_sec"] = transcribe_time
    result["score_time_sec"] = score_time

    if update_dashboard:
        global LAST_CALL_DATA
        call_id = f"sim_{sample_name}_{os.urandom(2).hex()}"
        final_score = result.get("risk_score", 0)
        classification = "SCAM" if final_score >= 70 else "LEGITIMATE"

        sim_wav_path = os.path.join(RECORDINGS_DIR, f"{call_id}.wav")
        try:
            with open(sample_file, "rb") as sf, open(sim_wav_path, "wb") as df:
                df.write(sf.read())
        except Exception:
            pass

        duration_sec = 35.0 if "scam" in sample_name else 45.0
        audio_sha256 = compute_file_sha256(sim_wav_path) if os.path.exists(sim_wav_path) else None
        logger.info(f"[Integrity] Sample test {call_id} audio hash: {audio_sha256[:16] if audio_sha256 else 'none'}... (SHA-256)")
        call_record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "formatted_time": datetime.now().strftime("%b %d, %Y %H:%M:%S") + f" (Sample: {sample_name})",
            "duration_seconds": duration_sec,
            "peak_risk_score": final_score,
            "final_risk_score": final_score,
            "raw_risk_score": final_score,
            "avg_risk_score": float(final_score),
            "classification": classification,
            "reason": result.get("reason", ""),
            "flags": result.get("flags", []),
            "transcript": text,
            "chunks_count": 1,
            "audio_file": f"{call_id}.wav" if os.path.exists(sim_wav_path) else None,
            "audio_sha256": audio_sha256,
            "hash_algorithm": "SHA-256",
            "tamper_status": "verified" if audio_sha256 else "none",
            "tamper_verified": True if audio_sha256 else False,
            "segments": [{"chunk_idx": 1, "elapsed": 0.0, "text": text, "risk_score": final_score, "flags": result.get("flags", [])}],
            "engine_breakdown": result.get("engine_breakdown", {}),
        }
        save_call_record(call_record)

        LAST_CALL_DATA.update({
            "call_id": call_id,
            "status": "completed",
            "start_time": call_record["formatted_time"],
            "duration_seconds": duration_sec,
            "current_risk_score": final_score,
            "peak_risk_score": final_score,
            "avg_risk_score": float(final_score),
            "raw_risk_score": final_score,
            "classification": classification,
            "reason": result.get("reason", ""),
            "flags": result.get("flags", []),
            "chunks_count": 1,
            "latest_transcript": text,
            "rolling_window": [text],
            "engine_breakdown": result.get("engine_breakdown", {}),
            "audio_sha256": audio_sha256,
            "hash_algorithm": "SHA-256",
            "tamper_status": "verified" if audio_sha256 else "none",
            "tamper_verified": True if audio_sha256 else False,
        })
        save_last_call_data()
        result["audio_sha256"] = audio_sha256
        result["saved_call_id"] = call_id

    return result


@app.post("/api/test/audio")
async def test_audio_upload(
    file: UploadFile = File(...),
    save_to_history: bool = Form(True)
):
    logger.info(f"[Simulator] Received audio file '{file.filename}' for threat simulation.")
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        t0 = time.time()
        text = await anyio.to_thread.run_sync(transcribe_wav, path)
        transcribe_time = round(time.time() - t0, 2)
        logger.info(f"[Simulator] Transcription ({transcribe_time}s): \"{text}\"")

        t1 = time.time()
        result = await anyio.to_thread.run_sync(_score_with_mode, text, "unknown", AUDIO_SYSTEM)
        score_time = round(time.time() - t1, 2)
        logger.info(f"[Simulator] Scoring ({score_time}s): score={result.get('risk_score', 0)}")

        result["transcript"] = text
        result["filename"] = file.filename
        result["transcribe_time_sec"] = transcribe_time
        result["score_time_sec"] = score_time

        if save_to_history:
            global LAST_CALL_DATA
            call_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(2).hex()}"
            sim_wav_path = os.path.join(RECORDINGS_DIR, f"{call_id}.wav")
            with open(sim_wav_path, "wb") as out_f:
                out_f.write(content)

            final_score = result.get("risk_score", 0)
            classification = "SCAM" if final_score >= 70 else "LEGITIMATE"
            duration_sec = round(len(content) / 32000.0, 1) if len(content) > 32000 else 6.0
            call_record = {
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
                "formatted_time": datetime.now().strftime("%b %d, %Y %H:%M:%S") + " (Simulation)",
                "duration_seconds": duration_sec,
                "peak_risk_score": final_score,
                "final_risk_score": final_score,
                "raw_risk_score": final_score,
                "avg_risk_score": float(final_score),
                "classification": classification,
                "reason": result.get("reason", ""),
                "flags": result.get("flags", []),
                "transcript": text,
                "chunks_count": 1,
                "audio_file": f"{call_id}.wav",
                "segments": [{"chunk_idx": 1, "elapsed": 0.0, "text": text, "risk_score": final_score, "flags": result.get("flags", [])}],
                "engine_breakdown": result.get("engine_breakdown", {}),
            }
            save_call_record(call_record)

            LAST_CALL_DATA.update({
                "call_id": call_id,
                "status": "completed",
                "start_time": call_record["formatted_time"],
                "duration_seconds": duration_sec,
                "current_risk_score": final_score,
                "peak_risk_score": final_score,
                "avg_risk_score": float(final_score),
                "raw_risk_score": final_score,
                "classification": classification,
                "reason": result.get("reason", ""),
                "flags": result.get("flags", []),
                "chunks_count": 1,
                "latest_transcript": text,
                "rolling_window": [text],
                "engine_breakdown": result.get("engine_breakdown", {}),
                "audio_sha256": audio_sha256,
                "hash_algorithm": "SHA-256",
                "tamper_status": "verified" if audio_sha256 else "none",
                "tamper_verified": True if audio_sha256 else False,
            })
            save_last_call_data()
            result["saved_call_id"] = call_id
            result["audio_sha256"] = audio_sha256

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Simulator] Audio simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# UI Dashboard (Accessible at http://localhost:8000/)
# ---------------------------------------------------------------------------
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ScamShield Dashboard Error: templates/index.html not found</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
