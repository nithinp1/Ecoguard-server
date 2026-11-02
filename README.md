# Scam-call Detection Server & Model Benchmarking Framework

FastAPI + faster-whisper + Ollama backend for real-time scam call and smishing detection, accompanied by a reproducible LLM evaluation framework and fine-tuned classification model.

---

## Server Setup & Usage

### Installation

```bash
pip install -r requirements.txt
ollama pull phi3.5
```

### Start Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --ws-ping-interval 20
```

### WebSocket Endpoints

- `/audio-stream`: Binary PCM16 WAV messages or raw signed 16-bit little-endian PCM (defaults to 16 kHz mono), returns `{risk_score, reason, flags, transcript}`.
  For raw PCM at another rate/channel count, first send
  `{"type":"audio_format","sample_rate":48000,"channels":2,"encoding":"pcm_s16le"}`.
  Each message must contain complete PCM frames; WAV messages must be self-contained.
  Accepted input: 8–192 kHz, 1–8 channels, normalized to 16 kHz mono.
  Unsupported encodings and malformed WAVs produce an error and end the stream.
  Send text `END` after the final audio message to flush and finalize the call.
- `/sms-check`: Text/JSON SMS evaluation, returns `{risk_score, reason, flags}`.

---

## Persistent dashboard settings

Use **Save & Apply Configuration** or the settings dropdowns; changes are saved to
`data/runtime_config.json` and survive server restarts and container recreation
through the existing `./data:/app/data` volume. No host `.env` edits are needed
for dashboard changes. Provider/model changes still warm the selected model in
background; wait for its readiness before sending audio.

Precedence: built-in defaults → `.env`/Compose environment → saved dashboard settings.
Saved overrides load before model selection and warmup. Settings not saved through
the dashboard continue using deployment defaults. Ollama settings and API keys are
included; blank key inputs retain existing keys.

A save failure returns HTTP 503 without changing running settings. The UI shows
success only after the API confirms `persisted=true`. Invalid settings return 422.
Malformed saved JSON is not overwritten: startup logs a warning and uses deployment
defaults; repair the file or remove it and restart to restore deployment defaults.

The JSON contains plaintext API keys if entered through the dashboard. Files use
owner-only permissions on POSIX; protect host filesystem ACLs and backups as well.
`data/` is excluded from Git and Docker builds, and secrets are not returned in API
responses or logs. Keep this unauthenticated administration dashboard restricted to
trusted networks. Use one Uvicorn worker/replica; configuration and active call state
are not coordinated across processes. Saves are atomic and file-synced, not a
power-loss durability guarantee for the host filesystem.

## Audio troubleshooting

The active Docker application is `server.py`; `old/` and `Advanced Versions/`
are historical implementations. See [AUDIO_DEBUGGING.md](AUDIO_DEBUGGING.md)
for the investigation, reproduced regressions, mobile capture risks, and validation limits.

- `WHISPER_LANGUAGE=auto` detects the spoken language; set `en`, `hi`, etc.
  for a known language. This also applies when another provider falls back to Whisper.
- `WHISPER_MODEL` defaults to `medium`; Compose now respects an explicit override.
- `CALL_INACTIVITY_TIMEOUT` defaults to 20 seconds, allowing jitter between
  multi-second mobile WAV messages. Set it above the maximum expected packet
  interval. `END` finalizes without waiting for timeout.
- Keep Uvicorn's WebSocket ping interval at `20`, not `0`: zero sends pings
  continuously with the installed protocol implementation; it does not disable them.
- Socket receiving runs independently of inference, with a maximum backlog of
  256 messages / 16 MiB per connection. Overload closes with code `1013`, discards
  pending audio, and marks the saved call `audio_incomplete=true`.
- Startup warms only the configured STT provider. After selecting or changing a
  local model, wait for `stt_status=hot` and `stt_inference_ready=true` in
  `/api/status` before streaming. Sarvam reports `configured` without a cloud probe.
- Sarvam realtime uses `saaras:v4` with `stream_type=fast` and VAD endpointing.
  The server splits normalized PCM into 100 ms frames; multi-second mobile messages
  must not be forwarded whole because fast mode caps each frame at 16,000 bytes.
  Responses/call records include `stt_transport` (`sarvam_realtime`, `batch_fallback`,
  or `batch_recovery`) so degraded batch processing is not mistaken for realtime.
  Smaller mobile capture packets are still necessary for subsecond end-to-end latency.
- Whisper VAD is enabled to reduce decoding on silence; per-window text conditioning
  is disabled to reduce repetitive decoding. These settings still need evaluation
  against representative device recordings.

Run model-free audio regressions (NumPy and SciPy required):

```bash
python -B -m unittest discover -s tests -v
```

## Model Benchmarking System

A reproducible benchmarking system evaluating local LLMs on a fixed 48-case scam dataset using identical prompts, classification rules, and resource tracking.

> **Audio test dataset source:** [github.com/CyberSorceress/Scam-Call-Detection](https://github.com/CyberSorceress/Scam-Call-Detection) (WAV files extracted into `evaluation_48.csv`)

### Benchmark Setup

```bash
pip install -r benchmark/requirements.txt

# Extract 48 audio test cases into CSV dataset (runs once)
python -m benchmark.setup_dataset
```

### Running Benchmarks

Ensure Ollama is active with target models pulled (`ollama pull phi3.5`, `ollama pull phi4`).

```bash
# Benchmark Phi-3.5 Mini
python -m benchmark.run_benchmark --model phi35

# Benchmark Phi-4 Mini
python -m benchmark.run_benchmark --model phi4

# Generate comparison table, confusion matrices, and plots
python -m benchmark.compare
```

### Output Directory Structure

```text
benchmark/
├── datasets/
│   └── evaluation_48.csv       # Fixed 48-case evaluation dataset
├── prompts/
│   └── ca_scam_system_prompt.txt # Fixed system prompt
├── runs/
│   ├── phi35/                   # Raw results, outputs, metadata for Phi-3.5
│   └── phi4/                    # Raw results, outputs, metadata for Phi-4
└── results/
    ├── model_comparison.csv     # Metric matrix comparison
    ├── confusion_matrices/      # JSON confusion matrices
    └── plots/                   # Bar charts, latency boxplots, matrices
```

### Measured Metrics

- **Classification:** Accuracy, Precision, Recall, F1, TP, TN, FP, FN, FP-Rate, FN-Rate.
- **Latency:** Mean, Median, Min, Max, P95 latency (ms).
- **Resource Usage:** VRAM peak (MB), RAM (MB), CPU usage (%).
- **Reliability & Consistency:** Success/Error counts, Throughput (RPS/RPM), Output consistency rate.

---

## Fine-Tuned Model (`phi35-scam`)

A QLoRA-fine-tuned Phi-3.5 Mini achieving **100% accuracy** (48/48) on the holdout benchmark, up from 93.75% baseline.

### Quick Start

```bash
# Import into Ollama
ollama create phi35-scam -f finetune/gguf/Modelfile

# Use via API
curl http://localhost:11434/api/chat -d '{
  "model": "phi35-scam",
  "messages": [
    {"role": "system", "content": "You detect phone scams. Given a transcribed phone call, classify whether it is SCAM or LEGITIMATE. Reply with only SCAM or LEGITIMATE."},
    {"role": "user", "content": "<transcript>"}
  ],
  "stream": false
}'
```

### Key Results

| Metric | Phi-3.5 Baseline | **phi35-scam** |
|:-------|:-----------------:|:--------------:|
| Accuracy | 93.75% | **100.0%** |
| False Positives | 3 | **0** |
| Mean Latency | 9,468 ms | ~3,300 ms |

Full report: [`finetune/FINE_TUNED_MODEL_REPORT.md`](finetune/FINE_TUNED_MODEL_REPORT.md)

---

## Environment & Reproducibility

- **OS:** Windows 11 / Linux
- **Python:** 3.10+
- **Inference Engine:** Ollama (`http://localhost:11434`)
- **Transcription Model:** faster-whisper `medium` by default (float16/CUDA, int8/CPU fallback at model initialization)
- **Scam Threshold:** `risk_score >= 70`
