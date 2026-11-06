# Echoguard Server

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-CUDA%2012-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

High-performance, real-time backend engine for automated scam call and smishing detection. **Echoguard Server** processes live streaming audio and SMS text via WebSockets, executing a low-latency multi-engine detection ensemble (LLM + XGBoost + Heuristic Rule Engine) with pluggable Speech-to-Text (STT) providers.

> 📱 **Client Application:**  
> The companion Android client that captures on-device calls and SMS in real-time is hosted in a separate repository: **[Echoguard-companion-app](https://github.com/nithinp1/Echoguard-companion-app)**.

---

## ⚡ Key Highlights & Architecture

- **Multi-Engine Hybrid Scoring Ensemble:**
  - **LLM Contextual Reasoning:** Local inference via Ollama (`phi3.5`) analyzing semantic deceit, psychological urgency, and impersonation cues.
  - **Machine Learning (XGBoost):** Lightweight classification model (`xgboost_model/`) evaluating engineered linguistic and behavioral feature vectors.
  - **Deterministic Rule Engine:** Sub-millisecond regex and pattern matching (`rulebased/`) for critical triggers (OTP requests, bank threats, malicious URLs).
  - **Configurable Modes:** Run `all` (ensemble max score & union flags), `llm_only`, `xgboost_only`, or `rulebased_only`.
- **Pluggable Speech-to-Text (STT) Layer:**
  - **Local Whisper (`faster-whisper`):** CUDA float16 GPU acceleration with CPU fallback.
  - **Sarvam AI Streaming:** Real-time bilingual and Indic speech transcription via low-latency WebSocket frames.
  - **NVIDIA NeMo / Parakeet:** Streaming CTC ASR model support.
- **Real-Time Streaming WebSocket Pipeline:**
  - Ingests streaming PCM16 audio chunks over `/audio-stream` with Voice Activity Detection (VAD) and silence filtering.
  - Non-blocking background worker queues prevent audio packet dropping during downstream inference.
- **Interactive Security Dashboard:**
  - Embedded web interface (`/`) with live call telemetry, risk gauges, real-time audio playback simulation, and atomic runtime configuration persistence (`data/runtime_config.json`).
- **Production-Ready Docker Deployment:**
  - Multi-stage Docker image built on `nvidia/cuda:12.2.2-runtime-ubuntu22.04` with automated GPU reservation and volume-backed state.

---

## 🏗️ System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │     Mobile Client (Echoguard-companion-app) / WebSocket   │
       └───────────────────────────┬────────────────────────────┘
                                   │ Streaming Audio / SMS
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           Echoguard Server                                 │
│                                                                           │
│   ┌────────────────────────┐         ┌────────────────────────────────┐   │
│   │ Audio Stream Ingestion │ ───────►│    Pluggable STT Pipeline      │   │
│   │  (VAD + 16kHz PCM16)   │         │   (Whisper / Sarvam / NeMo)    │   │
│   └────────────────────────┘         └───────────────┬────────────────┘   │
│                                                      │ Transcripts         │
│                                                      ▼                    │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                 Multi-Engine Detection Ensemble                   │   │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │   │
│   │  │   LLM (Ollama)   │  │  XGBoost Model   │  │   Rule Engine   │  │   │
│   │  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘  │   │
│   └───────────┼─────────────────────┼─────────────────────┼───────────┘   │
│               └─────────────────────┼─────────────────────┘               │
│                                     ▼                                     │
│                     Ensemble Risk Aggregator & Scorer                     │
│                                     │                                     │
│                                     ▼                                     │
│                  Live WebSocket Alerts & Dashboard Feed                   │
└───────────────────────────────────────────────────────────────────────────┘
```

Detailed architectural diagrams, sequence flows, and model benchmarks are available in [`docs/`](docs/).

---

## 🚀 Quickstart

### Published Docker Image

Every push to `main` publishes `ghcr.io/nithinp1/echoguard-server:latest` and a commit-specific `sha-*` tag. Pushing a version tag such as `v1.2.0` also publishes version tags. The image is linked to this repository through its OCI source label.

To update a server already running with Docker Compose, pull the new image and recreate the container:

```bash
docker compose pull
docker compose up -d --no-build
```

The Compose file retains its build configuration for local builds. To build locally instead of using GHCR, run `docker compose up --build -d`.

### Prerequisites
- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
- *(Optional for local GPU)* NVIDIA Container Toolkit & CUDA Drivers
- [Ollama](https://ollama.ai/) running on host (with `ollama pull phi3.5`)

### 1. Clone & Configure
```bash
git clone https://github.com/nithinp1/Ecoguard-server.git
cd Ecoguard-server

# Copy environment template
cp .env.example .env
```
*Edit `.env` to configure your preferred STT provider and credentials (e.g., `SARVAM_API_KEY`).*

### 2. Run with Docker Compose
```bash
docker compose up --build -d
```
The server will start on `http://localhost:8000`.

### 3. Access Dashboard & API
- **Web Dashboard:** [http://localhost:8000/](http://localhost:8000/)
- **Health / Readiness:** [http://localhost:8000/api/status](http://localhost:8000/api/status)
- **Interactive OpenAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API & WebSocket Interfaces

| Endpoint | Protocol | Description |
| :--- | :--- | :--- |
| `/audio-stream` | **WebSocket** | Streams binary PCM16 audio chunks; receives real-time `{risk_score, reason, flags, transcript}`. |
| `/sms-check` | **WebSocket** | Analyzes SMS content payload; returns risk classification. |
| `/api/sms-check` | **POST** | REST endpoint for SMS fraud analysis and record logging. |
| `/api/test/sample/{sample_name}` | **POST** | Triggers end-to-end evaluation on bundled audio samples (`scam_call`, `normal_call`). |
| `/api/config` | **GET / POST** | Reads and updates persistent runtime configuration (STT provider, model, detection mode). |
| `/api/status` | **GET** | Health check reporting STT warmup status, inference readiness, and hardware state. |

---

## 🧪 Testing

Run automated unit and integration tests:

```bash
# Run unit & configuration tests
python -m pytest tests/ -v

# Stream a test audio sample through the WebSocket client
python tests/test_client.py test_samples/scam_sample.wav
```

---

## 📁 Repository Structure

```text
.
├── Dockerfile                  # CUDA-accelerated Ubuntu 22.04 runtime container
├── docker-compose.yml          # Container configuration & GPU passthrough
├── requirements.txt            # Core Python dependencies
├── server.py                   # Main FastAPI server & WebSocket handlers
├── runtime_config.py           # Thread-safe persistent configuration store
├── templates/
│   └── index.html              # Real-time monitoring and analytics dashboard
├── rulebased/                  # Low-latency regex and heuristic rule engine
├── xgboost_model/              # Feature extraction pipeline and trained XGBoost model
├── test_samples/               # Bundled sample audio calls for simulation
├── tests/                      # Pytest test suite & test WebSocket client
└── docs/                       # In-depth system specifications and evaluation reports
    ├── ARCHITECTURE.md
    ├── SERVER_DOCUMENTATION.md
    ├── SERVER_V3_SPECIFICATION.md
    ├── AUDIO_DEBUGGING.md
    └── TEST_RESULTS.md
```

---

## 🔗 Related Repositories

- **Android Companion App:** [nithinp1/Echoguard-companion-app](https://github.com/nithinp1/Echoguard-companion-app) — Mobile client for on-device call interception and audio streaming.
