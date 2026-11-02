# Server v3 Architecture & Technical Specification Proposal
**Next-Generation Low-Latency Real-Time Voice Scam Shield & Forensic Engine**

---

## 1. Executive Summary & Motivation

### Current Limitations in Server v2
1. **Audio Chunking Latency**: Server v2 processes audio in discrete 2–3 second chunks saved temporarily to disk, causing a 3.5s – 5.0s feedback loop between caller speech and threat score output.
2. **Whisper Hallucination on Short Segments**: Faster-Whisper utilizes a rigid 30-second window. On short 2–5s clips containing pauses, silence, or line static, the autoregressive decoder frequently hallucinates or enters repetitive phrasing loops.
3. **Suboptimal Cloud STT Ingestion**: Sarvam AI was previously invoked via REST batch multipart uploads with file chunking instead of its native WebSocket streaming endpoint.
4. **Synchronous Execution on Audio Loop**: Running feature extraction, scoring, and disk operations in the active receive loop creates jitter for continuous incoming streams.

### Server v3 Vision
Server v3 re-engineers the audio pipeline into an **event-driven, dual-track streaming architecture** capable of ingesting **20 ms – 100 ms raw PCM packets** directly from telephony/WebRTC interfaces. It provides:
- **Instantaneous Transcription**: Sub-200ms turnaround via non-autoregressive local **NVIDIA NeMo (Parakeet-CTC 0.6B)** or cloud **Sarvam AI Realtime Streaming WebSocket (`saaras:v4`)**.
- **Dual-Track Stream Split ("Tee")**: Full-speed forwarding to the STT engine while in parallel accumulating an in-memory forensic buffer for SHA-256 tamper-proof call recording.
- **Hierarchical Multi-Tier Threat Arbitration**: Real-time keyword filtering (<1 ms) on partials, XGBoost evaluation (<5 ms) on sentence fragments, and non-blocking asynchronous LLM semantic scoring on turn boundaries.

---

## 2. System Architecture Diagram

```
                              [Incoming Telephony / Mobile Client]
                                               │
                                               │ (20ms - 100ms PCM Packets @ 16kHz)
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │          Server v3 Gateway: /audio-stream      │
                       └───────────────────────┬───────────────────────┘
                                               │
                                      Memory "Tee" Splitter
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    ▼                                                     ▼
     ┌─────────────────────────────┐                       ┌─────────────────────────────┐
     │   Track A: Hot Inference    │                       │  Track B: Forensic Archive  │
     │      (Zero-Copy Stream)     │                       │     (In-Memory Ring Buffer) │
     └──────────────┬──────────────┘                       └──────────────┬──────────────┘
                    │                                                     │
     ┌──────────────┴──────────────┐                                      │ (On Call Disconnect)
     │ Dynamic STT Engine Selector │                                      ▼
     └──────┬───────────────┬──────┘                       ┌─────────────────────────────┐
            │               │                              │ Writes data/recordings/.wav │
   (Local In-Process)  (Cloud Realtime WS)                 │ Computes SHA-256 Hash       │
            │               │                              │ Stores Audit Manifest       │
            ▼               ▼                              └─────────────────────────────┘
  ┌──────────────────┐ ┌──────────────────┐
  │ NVIDIA NeMo      │ │ Sarvam AI        │
  │ Parakeet-CTC 0.6B│ │ Realtime WS      │
  │ FastConformer    │ │ saaras:v4        │
  │ (Non-autoreg,    │ │ (Indic + English,│
  │  no loops, <50ms)│ │  partials <200ms)│
  └─────────┬────────┘ └────────┬─────────┘
            │                   │
            └─────────┬─────────┘
                      │ Emits Partials & Finals
                      ▼
   ┌──────────────────────────────────────────────────────────┐
   │         Hierarchical 3-Tier Threat Arbitrator            │
   ├──────────────────────────────────────────────────────────┤
   │ Tier 1 (Partials): Regex & Indicator Matrix (<1 ms)      │
   │ Tier 2 (Phrases) : XGBoost Probability Scorer (<5 ms)    │
   │ Tier 3 (Turns)   : Async LLM Generative Reasoner (Ollama)│
   └──────────────────────────┬───────────────────────────────┘
                              │
                              ▼
        [WebSocket Push to Live Dashboard & Client Alert]
```

---

## 3. Core Component Specifications

### 3.1. Audio Ingress & Dual-Track "Tee" Router
- **Input Specifications**:
  - Sample Rate: `16,000 Hz` (or `8,000 Hz` upsampled to 16 kHz).
  - Format: `Linear PCM 16-bit Mono Little-Endian` (`linear16`).
  - Packet Cadence: `20 ms` (640 bytes), `50 ms` (1,600 bytes), or `100 ms` (3,200 bytes).
- **In-Memory Streaming Tee**:
  - **Track A (Live Inference)**: Forwards raw PCM bytes immediately to the active ASR connection. No disk writes, no WAV header generation on the hot path.
  - **Track B (Evidence Accumulator)**: Appends raw chunks into a `bytearray` buffer in RAM. Upon call termination, the complete byte stream is written to `data/recordings/<call_id>.wav` and its SHA-256 checksum is calculated.

### 3.2. Speech-to-Text (ASR) Engine Matrix

| Feature | NVIDIA NeMo Parakeet-CTC 0.6B | Sarvam AI Realtime WebSocket | Faster-Whisper (Safety Fallback) |
| :--- | :--- | :--- | :--- |
| **Model Type** | FastConformer-CTC (Non-autoregressive) | Saaras:v4 / Saaras:v3-realtime | Encoder-Decoder Transformer |
| **Execution** | Local In-Process (CUDA GPU) | Cloud WebSocket Bridge | Local In-Process (CUDA/CPU) |
| **Hallucination Risk** | **Zero** (no autoregressive decoder) | Low (Acoustic CTC/Conformer) | **High** on <5s audio clips |
| **Input Chunk Size** | 20 ms – 100 ms arbitrary chunks | 20 ms – 100 ms (recommends 100 ms) | Requires buffer window ($\ge 2.0\text{s}$) |
| **Latency** | **< 50 ms** on RTX 3060 | **~150 ms – 250 ms** over WAN | 400 ms – 800 ms per chunk |
| **Language Support** | High-accuracy English | 22 Indic Languages + English | Multilingual |
| **Privacy / Offline** | 100% Offline & Private | Cloud TLS 1.3 | 100% Offline & Private |

### 3.3. Asynchronous Multi-Tier Threat Arbitration

To prevent LLM inference from blocking streaming audio, detection is split into three priority tiers:

1. **Tier 1 — Real-Time Heuristic Filter (<1 ms, Active on Partials)**:
   - Scans incoming `transcript.partial` streams against high-risk regex clusters (OTP requests, police/CBI arrest threats, remote desktop access tools like AnyDesk/TeamViewer).
   - Can fire immediate emergency alerts before the speaker even finishes their sentence.
2. **Tier 2 — Machine Learning Classification (<5 ms, Active on Fragments)**:
   - Evaluates TF-IDF vector embeddings with the pre-trained gradient-boosted decision tree (`xgboost_model`).
   - Produces continuous calibrated risk probabilities ($0-100$).
3. **Tier 3 — Generative Reasoning (Async, Active on VAD Turns)**:
   - Triggered only on `transcript.final` or VAD turn completion (`vad.speech_end`).
   - Dispatched to Ollama (`phi3.5` / `phi4-mini`) in a non-blocking background task.
   - Provides structured forensic explanations and detects sophisticated social engineering tactics.

### 3.4. Cryptographic Forensic Audit & Evidence Vault
- **SHA-256 Integrity Verification**: Computed on the exact binary bytes of the saved call audio upon completion.
- **Tamper Detection**: Endpoint `GET /api/calls/{call_id}/verify-hash` reads the physical `.wav` file, computes a fresh SHA-256 digest, and compares it against the recorded database hash.
- **Evidence Vault UI**: Built-in audio playback, download, interactive timeline inspect drawer, and visual integrity verification badges.

---

## 4. REST & WebSocket API Specification

### 4.1. WebSocket Endpoints
* **`WS /audio-stream`**:
  * Ingests binary PCM chunks (`linear16`, 16 kHz).
  * Streams back real-time detection telemetry:
    ```json
    {
      "event": "partial_score",
      "text": "I am calling from police department please share your otp",
      "risk_score": 100,
      "tier": "CRITICAL",
      "flags": ["[HEURISTIC] Police impersonation", "[HEURISTIC] OTP demand"],
      "latency_ms": 68
    }
    ```
* **`WS /sms-check`**:
  * Real-time bidirectional socket for SMS payload analysis and sender verification.

### 4.2. REST Configuration & Telemetry Endpoints
* **`GET /api/status`**: System health, active STT engine, hardware accelerator (`cuda`), and model residency.
* **`POST /api/config`**: Dynamic runtime reconfiguration (switches STT between `parakeet`, `nemotron`, `sarvam`, or `whisper`; adjusts VAD sensitivity; updates keys).
* **`GET /api/last-call`**: Live threat monitor polling for active call sessions.
* **`POST /api/last-call/clear`**: Resets monitor to `IDLE` state.
* **`GET /api/calls`**: Retrieves archived call manifest.
* **`GET /api/calls/{call_id}/verify-hash`**: Re-computes and audits cryptographic audio hash.
* **`GET /api/calls/{call_id}/audio`**: Streams WAV recording.
* **`POST /api/test/sms`**: Direct API for smishing test payloads.
* **`POST /api/test/sample/{sample_name}`**: Runs automated end-to-end test against bundled scam or normal call recordings.

---

## 5. Migration Roadmap from Server v2 to Server v3

| Phase | Milestone | Scope |
| :--- | :--- | :--- |
| **Phase 1** | **Parakeet-CTC 0.6B Integration** | Add `nvidia/parakeet-ctc-0.6b` to the local NeMo loader. Benchmark inference latency on 50ms–100ms packets versus Nemotron. |
| **Phase 2** | **Sarvam Realtime WebSocket Client** | Implement `AsyncSarvamStreamingClient` using `websockets` / `sarvamai` SDK for bidirectional streaming over `wss://api.sarvam.ai/speech-to-text-realtime/ws`. |
| **Phase 3** | **Dual-Track Stream Splitter** | Replace temporary disk-based chunk files with an in-memory `LinearPCMBuffer` that splits live audio to the active ASR while retaining complete call audio for the Evidence Vault. |
| **Phase 4** | **Async Tiered Arbitrator** | Decouple LLM inference from the primary audio loop; evaluate Tier 1 (Regex) on partials and Tier 2/3 on turn completions. |
| **Phase 5** | **UI/UX Final Polish** | Align web frontend with Stitch design specifications, displaying real-time partial waveform meters and sub-second latency monitors. |
