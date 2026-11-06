# Audio transcription investigation

## Active code path

Docker starts `uvicorn server:app` using the root `server.py`.
The Android source reviewed in the initial investigation, then under
`app/src/main/java/com/echoguard/companion/ScamDetectionService.kt`,
captured 16 kHz mono PCM16 and sent approximately six-second WAV messages.
That source is no longer at this workspace path, and the installed APK version
has not been identified. The subsequent live logs show a 3.008-second input;
do not assume every mobile variant uses the original six-second cadence.

The endpoint inspects WAV metadata (or uses negotiated raw PCM metadata), normalizes
to 16 kHz mono PCM16, saves normalized audio in memory, and either forwards it to
Sarvam's WebSocket or transcribes local buffered windows. Transcripts feed the
rule/XGBoost/LLM scoring layer. At call end, recordings and call metadata are saved
under `data/`. The scoring engines do not decode audio.

`test_client.py` sends five-second WAV messages. The sample upload endpoint
transcribes whole files and therefore does not validate mobile capture or streaming
boundaries. Some historical architecture/testing documents describe older behavior.

## Findings addressed

| Finding | Change | Verification |
| --- | --- | --- |
| Android sends roughly every six seconds, equal to the old six-second inactivity timeout | Default increased to 20 seconds; explicit environment override remains available | Async endpoint test records the default receive timeout |
| Whisper forced `language="en"`, including provider fallbacks | `WHISPER_LANGUAGE=auto` by default; explicit language codes supported; visible in `/api/status` | Mock model verifies language forwarding |
| Remaining local audio below 8,000 bytes / 250 ms was skipped once previous segments existed | Submit every nonempty final buffer; clean up temporary files on failure; log tail errors | Async endpoint tests with a 100 ms tail and failed tail transcription |
| Negotiated encodings were acknowledged but never decoded | Accept only `pcm_s16le`; reject invalid rate/channel metadata explicitly | Async format negotiation tests |
| Invalid WAV headers fell through as raw PCM, and WAV sample width was ignored | Reject malformed/truncated WAV and non-PCM16 WAV instead of interpreting their bytes as PCM16 | Helper and endpoint tests |
| Resampling used integer input, truncated fractional stereo mixing, and could wrap FIR overshoot on conversion | Float32 downmix/resampling; round and saturate into explicit little-endian int16; declare NumPy/SciPy directly | Sine RMS/duration tests at 8/48 kHz, float-input spy, saturation and stereo tests |
| Normalization could silently return original-rate audio labeled as 16 kHz after failure | Remove the silent normalization fallback | Invalid PCM-frame/format tests |
| Model-size reload held a non-reentrant lock and called the loader, which acquired the same lock again | Use a reentrant model lock | Bounded-thread test with a fake model loader |

Whisper also enables VAD and disables previous-text conditioning to reduce decoding
on silence and repetitive text. The option forwarding is tested, but recognition
quality needs representative real audio evaluation. Short or quiet speech can still
be missed; these settings are not a measured accuracy claim.

The integer-resampling concern is version-dependent: baseline sine amplitudes were
preserved on the locally available SciPy 1.18.1. Silence corruption was **not**
reproduced in that version. Explicit floating-point input avoids older integer-FIR
behavior, and clipping prevents the independently reproducible conversion wraparound.
Subsequent live-container inspection found SciPy 1.15.3. The complete current
regression suite also passes with that container's dependencies.

Compose now honors `WHISPER_MODEL` overrides instead of unconditionally selecting
`medium`. The existing Dockerfile default remains `medium`.

## Important unresolved mobile and provider risks

1. **The Android client discards its unsent final chunk.** `recordLoop()` only sends
   after collecting roughly six seconds. `cleanupSession()` stops capture and closes
   the socket without flushing that buffer or sending `END`. Up to approximately six
   seconds, or an entire short session, may never reach the server. Server tail
   flushing cannot recover unsent audio. The mobile lifecycle needs a coordinated
   stop/drain/END handshake; it was not changed during this server-focused pass.
2. **Remote-party capture is not established.** The actual Android capture source
   preference is `VOICE_COMMUNICATION`, `VOICE_RECOGNITION`, `MIC`, then `DEFAULT`.
   Successful `AudioRecord` initialization does not prove the far-end caller is
   present. Android/device restrictions may provide only microphone audio. Verify
   audibility on the target device before changing ASR models; a server cannot
   reconstruct speech missing from the captured signal. The Android README's direct
   call-capture description does not match the implementation.
3. **Local acoustic context resets at window boundaries.** A one-second minimum
   buffer is used for small-packet clients; this does not mean the checked-in mobile
   client's six-second messages are split into one-second windows. There is no
   overlapping acoustic context, and whole-call transcription only runs if all
   segment text is empty. Missing boundary words remain possible. Compare segmented
   and whole-recording recognition before redesigning buffering.
4. **Sarvam latency still depends on client packet cadence.** The subsequent
   investigation below fixed oversized provider frames, ignored provider errors,
   and premature session closure. Splitting a three-second mobile packet after
   arrival cannot remove the three-second client-side capture delay. True low-latency
   partial updates require smaller packets from the mobile app.
5. **CUDA recovery remains limited.** CPU fallback occurs at model construction,
   not necessarily at lazy inference. Provider-aware warmup now checks actual local
   inference and does not publish `hot` after an inference error, but runtime CUDA
   failures still require investigation. The supplied live logs show successful
   Whisper CUDA loading, not successful Parakeet inference.

## Safe cleanup

Removed generated `__pycache__` directories from the root, benchmark, root detection
engines, and historical engine/server directories. Kept recordings, trained models,
Android build outputs, backups, archived source, and UI assets: their disposability
could not be established from version control because this directory has no Git
repository metadata.

Extended `.dockerignore` to keep host `.env` files, backup/session files, Android
artifacts, historical implementations, and the unused UI export/archive out of the
image build context. This does not delete those files. Compose still supplies runtime
configuration through its existing `env_file` and read-only `.env` mount. Root
`rulebased/` and `xgboost_model/` remain in the image.

## Validation and limits

Run from the project root with NumPy and SciPy installed:

```sh
python -B -m unittest discover -s tests -v
```

The tests AST-extract selected server definitions instead of importing the full
application, so they do not load ASR models, start services, contact providers, or
modify real call history. Async endpoint tests use fake sockets/scoring and temporary
recording directories. They verify waveform handling, control flow, and model option
forwarding—not word error rate or production concurrency.

Baseline: 13 test methods produced 23 failing assertion/subtest cases against the
original implementation. After fixes and expanded coverage: 19 test methods passed.
Validation used a disposable Windows Python 3.14 virtual environment with NumPy/SciPy,
not the Docker image's Python 3.10 dependency set. Both edited Python files
also passed Python 3.10 grammar parsing. `docker compose config --quiet
--no-env-resolution --no-interpolate` passed; this checks configuration structure,
not the presence of runtime secrets, model readiness, or a successful image build.

Editor diagnostics still report unresolved runtime imports in its selected Python
environment, existing style/type warnings, a potentially unbound final scoring
`result`, and an undefined `audio_sha256` in the separate audio-upload endpoint.
These were not introduced by the audio changes and were not swept into this patch.
The isolated tests do not exercise the full upload endpoint or all scoring failures.

During the initial investigation, Docker inspection found no running containers;
two attempts to run cached images timed out (120 seconds and 30 seconds). The later
reconnect investigation successfully tested the user's running container as described
below. No production server was restarted or replaced by the agent, and no models
were downloaded by the tests. End-to-end mobile accuracy, actual model inference
with these latest changes, and Sarvam API sessions remain unvalidated.

## Reconnect-loop investigation and fixes

The user's subsequent logs showed Whisper loading from 06:03:30 to 06:11:08,
a runtime switch to Parakeet at 06:06:13, and Parakeet beginning a cold load at
06:07:06. Readiness nevertheless became `hot` after Whisper warmed, although
Parakeet had not logged successful loading. Ollama separately timed out.

Live inspection found Uvicorn 0.54.0 and websockets 16.1.1. The installed protocol
only disables keepalive when the interval is `None`; `0` sleeps zero seconds
between ping/pong cycles. The Dockerfile's `--ws-ping-interval 0` therefore did
not disable heartbeat traffic. It is now `20`, with the default pong timeout
unchanged. Increasing Android's timeout was not used as a workaround.

The old application stopped calling `ws.receive()` while awaiting each model
load/transcription. That can fill the protocol's bounded message queue, which
then stops reading subsequent control frames, including client pings. This can
happen even when model work is correctly offloaded and the event loop is idle.

Changes in `server.py`:

- A dedicated receiver drains transport messages while inference is running.
  Its per-connection application queue is capped at 256 messages and 16 MiB.
  Exceeding either cap emits an explicit incomplete-call error and closes with
  code 1013. Pending queued audio is discarded rather than consuming GPU time
  after overload closure. Saved calls/final state carry `audio_incomplete=true`.
  Already in-flight work is allowed to finish; model threads are not forcibly killed.
- Receiver tasks are cancelled and joined during cleanup, including cancellation
  during initial state persistence. Rule/XGBoost scoring and live-state persistence
  now run off the event loop. Other post-call synchronous I/O is not redesigned.
- Startup warms only the selected provider. Provider/model changes reset readiness
  and schedule background validation. Superseded warmups cannot mark another model
  ready. Local warmup and inference share a reentrant lock to avoid overlapping
  GPU model operations. In-flight downloads cannot safely be cancelled.
- Local `hot` requires a successful inference pass. `/api/status.device` comes
  from selected-provider readiness, not always Whisper. `stt_inference_ready`
  distinguishes actual local readiness from Sarvam's unprobed `configured` state.
- Ollama warms independently when enabled, including when detection mode is
  switched on later; stale warmup responses cannot overwrite newer readiness.

**Reproduction:** A real loopback websockets server with `max_queue=1` received
multiple audio messages while mocked transcription was deliberately blocked.
A subsequent client ping timed out with the deployed pre-fix `audio_stream`.
The same test passed against the changed source without releasing transcription
before the pong. This establishes the backlog mechanism, not every possible
cause of an Android timeout.

**Current validation:** 56 tests passed inside the existing container (Python 3.10),
including the protocol test, waveform regressions, overflow/cancellation cleanup,
provider selection, readiness transitions, and warmup/inference locking. Tests ran
from `/tmp/scamshield-heartbeat-tests`, not by replacing `/app/server.py`.
Real GPU/ASR services are mocked. Legacy-websockets deprecation warnings remain.

### Applying these changes

Stop mobile monitoring during deployment. After deploying, select Parakeet and
`nvidia/parakeet-ctc-0.6b` in the dashboard and save. Dashboard changes now persist
in the data volume; manually editing the host `.env` is no longer required.

```sh
docker compose up -d --build
docker compose logs -f scam-detection-server
```

Wait for the selected provider's `stt_status=hot` and `stt_inference_ready=true`
in `/api/status` before starting mobile audio. Existing package-install build
layers should be reused if unchanged; do not use `--no-cache` for this patch.

The historical `.env` warning came from attempting atomic replacement of a
read-only bind-mounted file. The persistent-dashboard change below removes that
write path entirely; `.env` now supplies deployment defaults only.

## Sarvam realtime-versus-bulk investigation

The next live log review established this sequence:

- 08:26:15: WebSocket handshake logged as connected to `saaras:v4`.
- 08:27:29: call inactivity timeout, with no streaming transcript history.
- 08:27:31–32: two REST `/speech-to-text` requests produced the full transcript.

The realtime endpoint, event schema, model, language, and VAD query settings were
confirmed against the installed official Sarvam SDK 0.1.41. A live connection
received `session.begin` with the expected `fast`/`vad`/16 kHz/`linear16` settings.
The model name was not the fault.

**Reproduced root cause:** sending one 3.008-second / 96,256-byte synthetic-silence
PCM frame caused the provider to respond:

```json
{"event":"error","code":"chunk_too_large","is_fatal":false,"received_bytes":96256,"max_bytes":16000}
```

Fast mode rejected that audio; the old client silently ignored the error event.
It recorded no live transcript, then the existing empty-history fallback uploaded
the whole call through REST. An HTTP/WebSocket handshake alone was not evidence
that audio had been accepted.

Changes in `AsyncSarvamStreamingClient` and `/audio-stream`:

- Split outgoing normalized PCM into at most 100 ms frames (3,200 bytes at 16 kHz),
  preserving every byte and respecting the 16,000-byte provider cap.
- Wait for `session.begin` before declaring the stream ready; log provider errors
  and unexpected receiver termination. Sends return explicit success/failure.
- `flush` is only used with manual endpointing, as defined by the SDK. Normal VAD
  shutdown sends `end`, drains finals until `session.end`, then closes and joins
  the receiver. The five-second completion deadline includes the end-send itself.
- Failed live sends switch explicitly to per-chunk batch fallback. Failed started
  streams recover the full call even if earlier finals exist, replacing rather
  than appending transcript text. A setup failure does not redundantly re-upload
  already-successful batch chunks at finalization.
- A successful silent stream no longer causes a needless whole-call REST upload.
- Results and saved call records expose `stt_transport` and `realtime_failed` to
  distinguish realtime, batch fallback, and batch recovery. Failed full-call recovery
  is logged and marks audio incomplete. Provider-side automatic Whisper fallback
  within the existing batch wrapper remains possible.
- Session summaries log provider-reported processed audio, utterance count, and
  received partial/final counts without logging audio content.

**Validation:** 74 tests passed inside the existing container, including 18 Sarvam
mock-protocol/endpoint regressions and the existing real loopback heartbeat test.
A second live probe using the changed client sent the same 3.008 seconds of synthetic
silence split into small frames. Sarvam reported `audio_duration_s=3.01`, clean
session completion, and no error. Silence intentionally produced zero utterances;
this proves accepted framing/finalization, not speech accuracy or actual partial
latency. No private recording was sent. Brief live probes may incur small API usage.

The running service was not restarted or overwritten. Rebuild/recreate it to apply
these changes, then select Sarvam, `saaras:v4`, `transcribe`, and `auto` in the
dashboard and save. Existing deployment API keys remain available; dashboard-entered
keys are now saved securely with the other overrides as described below.
Do not put keys in source code or logs.

## Persistent dashboard configuration

Dashboard settings no longer attempt to overwrite `/app/.env`. `runtime_config.py`
provides a whitelisted, atomic JSON store at `DATA_DIR/runtime_config.json`, inside
the existing persistent data volume. All exposed provider/model, detection mode,
Sarvam parameters, Ollama host/model, and API-key updates are saved before they are
applied in memory. Blank key inputs keep the previously configured key. Partial
updates merge with existing overrides rather than overwriting unrelated settings.

Startup loads `.env`/Compose defaults first, then saved dashboard overrides before
initializing runtime settings or warming models. Saved choices therefore win over
Compose's defaults across restart/recreation. Failed saves return HTTP 503 and do
not mutate runtime settings or queue model warmup. The dashboard checks HTTP status,
`success`, and `persisted=true` before displaying saved feedback or clearing key
inputs. Invalid input returns 422. Corrupt files produce a sanitized startup warning
and deployment-default fallback; saving does not silently overwrite a corrupt file.

`runtime_config.json` may contain plaintext API keys. Creation/replacement uses
POSIX owner-only permissions; protect Windows host ACLs, data-volume access, and
backups. Complete `data/` is excluded from Docker build contexts and Git so runtime
secrets are not baked into an image. API responses contain only existing key hints,
not complete keys. The administration API remains unauthenticated and must be kept
on trusted networks.

This implementation is single-process: the Docker command explicitly uses one
Uvicorn worker, and deployment examples were corrected accordingly. It does not
coordinate runtime state or configuration writes across workers/replicas. Writes
are atomic and file-synced, but are not a host-power-loss durability guarantee.
Removing the saved JSON and restarting restores deployment defaults.

Validation includes actual temporary-file saves, fresh-store/process reload,
startup precedence, blank-key preservation, failed-save rollback, NUL/invalid-input
rejection, concurrent same-process updates, and dashboard JavaScript feedback tests.
The existing model/stream regressions remain part of the suite. The container run
passed 103 tests, with four Node-based UI tests skipped because Node is absent in
the image; those four UI tests passed separately with host Node. Actual FastAPI
requests verified a persisted save and HTTP-503 rollback, then a fresh server
process restored the saved provider, mode, Ollama model, and fake test API key
over conflicting deployment defaults. No production container settings were
changed by these tests.

## Next debugging sequence

1. Rebuild/deploy the server in the intended Docker environment with the existing
   `.env` configuration. Inspect `/api/status` for provider, language, device, and
   model state; inspect startup logs for actual inference errors, not just `hot`.
2. On the target phone, record a known sentence from each call participant. Listen
   to the server-saved WAV locally and verify both voices, normal speed, no clipping,
   and no missing end of call. Do not upload private recordings to external services
   without consent. Fix mobile capture/stop delivery if this check fails.
3. For a recording both participants can hear, compare whole-file transcription
   against the live segmented output and manually verified text. Measure word error
   rate separately from scam-classification accuracy.
4. Repeat with the actual languages, background noise, Bluetooth/speaker routes,
   short calls, quiet speech, jitter, stop/reconnect, and selected STT provider.
5. If Sarvam is used, verify final-event delivery and disconnected-provider recovery
   with deterministic protocol tests before relying on final call transcripts.
