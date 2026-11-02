"""Audio regressions without importing server or loading models/persistent state.

Run from the project root: python -B -m unittest discover -s tests -v
Only explicitly selected definitions (and the Whisper language setting) execute.
"""

import ast
import asyncio
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
import time
from datetime import datetime
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
from scipy.signal import resample_poly


SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"


def extract_functions(*names, namespace=None, settings=()):
    """Compile selected top-level nodes, never server imports/startup code."""
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), str(SERVER_PATH))
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = set(names) - functions.keys()
    if missing:
        raise AssertionError(f"Missing server functions: {sorted(missing)}")
    nodes = []
    for name in settings:
        assignments = [
            node for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        ]
        if len(assignments) != 1:
            raise AssertionError(f"Expected one top-level {name} setting")
        nodes.extend(assignments)
    nodes.extend(functions[name] for name in names)
    module = ast.Module(
        body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )] + nodes,
        type_ignores=[],
    )
    scope = {
        "io": io, "wave": wave, "os": os, "math": math,
        "np": np, "resample_poly": resample_poly,
        "logger": logging.getLogger("audio-pipeline-tests"),
        "_local_asr_lock": threading.RLock(),
    }
    scope.update(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(SERVER_PATH), "exec"), scope)
    return scope


def pcm(samples):
    return np.asarray(samples, dtype="<i2").tobytes()


def wav_bytes(payload, rate=16000, channels=1, width=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(payload)
    return buffer.getvalue()


class NormalizePCMTests(unittest.TestCase):
    def setUp(self):
        self.scope = extract_functions("normalize_pcm_to_16k_mono")
        self.normalize = self.scope["normalize_pcm_to_16k_mono"]

    def test_same_format_is_byte_identical(self):
        payload = pcm([-32768, -123, 0, 123, 32767])
        self.assertEqual(self.normalize(payload, 16000, 1), payload)
        self.assertEqual(self.normalize(b"", 16000, 1), b"")

    def test_stereo_is_averaged_without_integer_overflow(self):
        stereo = [[32767, 32767], [-32768, -32768], [12000, -4000], [-9000, 3000]]
        self.assertEqual(
            self.normalize(pcm(stereo), 16000, 2),
            pcm([32767, -32768, 4000, -3000]),
        )

    def test_resampler_receives_float32_for_mono_and_stereo(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                captured = []

                def observe(samples, *args, **kwargs):
                    captured.append(samples.copy())
                    return resample_poly(samples, *args, **kwargs)

                spy = Mock(side_effect=observe)
                payload = pcm([[1000, 3001], [-1000, -3001]]) if channels == 2 else pcm([2000, -2000])
                expected = [2000.5, -2000.5] if channels == 2 else [2000, -2000]
                with patch("scipy.signal.resample_poly", spy), patch.dict(
                    self.scope, {"resample_poly": spy}
                ):
                    self.normalize(payload, 8000, channels)
                self.assertEqual(len(captured), 1, "Expected the scipy resampling path")
                self.assertEqual(captured[0].dtype, np.dtype("float32"))
                np.testing.assert_array_equal(captured[0], expected)

    def test_resampled_output_is_rounded_clipped_little_endian_int16(self):
        result = np.array(
            [-40000, -32768.6, -2.5, -0.6, 0.6, 2.5, 32767.6, 40000],
            dtype=np.float32,
        )
        fake_resample = Mock(return_value=result)
        with patch("scipy.signal.resample_poly", fake_resample), patch.dict(
            self.scope, {"resample_poly": fake_resample}
        ):
            output = self.normalize(pcm([1000] * 4), 8000, 1)
        fake_resample.assert_called_once()
        self.assertEqual(output, struct.pack("<8h", -32768, -32768, -2, -1, 1, 2, 32767, 32767))

    def test_resampling_preserves_sine_rms_and_duration(self):
        for rate in (8000, 48000):
            with self.subTest(rate=rate):
                # Exact half-second duration, safely below either Nyquist frequency.
                samples = np.rint(12000 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate))
                output = np.frombuffer(self.normalize(pcm(samples), rate, 1), dtype="<i2")
                self.assertEqual(len(output), 8000)
                rms = np.sqrt(np.mean(output.astype(np.float64) ** 2))
                self.assertAlmostEqual(rms, 12000 / math.sqrt(2), delta=100)

    def test_invalid_rates_channels_and_partial_frames_raise(self):
        cases = [
            (pcm([1, 2]), 0, 1), (pcm([1, 2]), -8000, 1),
            (pcm([1, 2]), 16000, 0), (pcm([1, 2]), 16000, -1),
            (b"\x00", 16000, 1), (b"\x00\x01\x02", 8000, 1),
            (pcm([1, 2, 3]), 16000, 2), (pcm([1, 2, 3]), 48000, 2),
        ]
        for payload, rate, channels in cases:
            with self.subTest(rate=rate, channels=channels, size=len(payload)):
                with self.assertRaises(ValueError):
                    self.normalize(payload, rate, channels)


class InspectAudioTests(unittest.TestCase):
    def setUp(self):
        self.inspect = extract_functions("inspect_audio_chunk")["inspect_audio_chunk"]

    def test_valid_16_bit_wav_preserves_pcm_and_metadata(self):
        payload = pcm([[1000, -1000], [2000, -2000]])
        self.assertEqual(self.inspect(wav_bytes(payload, 48000, 2)), (payload, 48000, 2))

    def test_raw_pcm_falls_back_without_fabricated_metadata(self):
        for payload in (b"", pcm([0, 1, -1, 32767, -32768])):
            with self.subTest(payload=payload):
                self.assertEqual(self.inspect(payload), (payload, None, None))

    def test_recognized_but_malformed_riff_wav_is_not_raw_pcm(self):
        for payload in (b"RIFF", b"RIFF\x00\x00\x00\x00WAVE", b"RIFF\x04\x00\x00\x00WAVExxxx"):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.inspect(payload)

    def test_non_16_bit_wav_is_rejected(self):
        for width in (1, 3, 4):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    self.inspect(wav_bytes(b"\x00" * width * 8, width=width))


class WhisperTests(unittest.TestCase):
    def test_reload_reenters_model_lock_and_loads_new_model(self):
        old_model = object()
        new_model = object()
        constructor = Mock(return_value=new_model)
        scope = extract_functions(
            "get_model", "reload_whisper_model", settings=("model_lock",),
            namespace={
                "threading": threading, "WhisperModel": constructor,
                "_whisper_model": old_model, "_device_used": "cpu",
                "WHISPER_MODEL_SIZE": "tiny",
                "MODEL_STATE": {"status": "hot", "stt": "hot", "device": "cpu"},
            },
        )
        errors = []

        def reload_model():
            try:
                scope["reload_whisper_model"]("SMALL")
            except Exception as error:
                errors.append(error)

        # A plain Lock deadlocks on get_model's nested acquire. Keep that failure bounded.
        worker = threading.Thread(target=reload_model, daemon=True)
        worker.start()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive(), "Model reload deadlocked acquiring model_lock twice")
        self.assertEqual(errors, [], f"Model reload raised: {errors!r}")
        constructor.assert_called_once_with("small", device="cuda", compute_type="float16")
        self.assertEqual(scope["WHISPER_MODEL_SIZE"], "small")
        self.assertIs(scope["_whisper_model"], new_model)
        self.assertIsNot(scope["_whisper_model"], old_model)
        self.assertEqual(scope["MODEL_STATE"]["status"], "hot")
        self.assertEqual(scope["MODEL_STATE"]["stt"], "hot")
        self.assertEqual(scope["MODEL_STATE"]["device"], scope["_device_used"])

    def test_language_configuration_and_transcription_options(self):
        for configured, expected in ((None, None), ("auto", None), ("en", "en"), ("hi", "hi")):
            with self.subTest(language=configured):
                model = Mock()
                model.transcribe.return_value = (
                    iter([SimpleNamespace(text=" hello "), SimpleNamespace(text=" world ")]),
                    SimpleNamespace(),
                )
                get_model = Mock(return_value=model)
                clean = Mock(side_effect=lambda text: text)
                environment = {} if configured is None else {"WHISPER_LANGUAGE": configured}
                with patch.dict(os.environ, environment, clear=True):
                    scope = extract_functions(
                        "transcribe_whisper",
                        settings=("WHISPER_LANGUAGE",),
                        namespace={"get_model": get_model, "clean_and_deduplicate_transcript": clean},
                    )
                    self.assertEqual(scope["transcribe_whisper"]("unused.wav"), "hello world")
                get_model.assert_called_once_with()
                args, kwargs = model.transcribe.call_args
                self.assertEqual(args, ("unused.wav",))
                self.assertIn("language", kwargs)
                self.assertEqual(kwargs["language"], expected)
                self.assertIs(kwargs.get("vad_filter"), True)
                self.assertIs(kwargs.get("condition_on_previous_text"), False)
                clean.assert_called_once_with("hello world")


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return next(self.messages, {"type": "websocket.disconnect"})

    async def send_json(self, message):
        self.sent.append(message.copy())

    async def close(self):
        self.closed = True


class AudioStreamTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.transcribed = []
        self.logger = Mock()

        async def run_sync(function, *args, **kwargs):
            return function(*args, **kwargs)

        def transcribe(path):
            self.assertTrue(Path(path).resolve().is_relative_to(self.root))
            with wave.open(str(path), "rb") as source:
                self.assertEqual((source.getframerate(), source.getnchannels(), source.getsampwidth()), (16000, 1, 2))
                self.transcribed.append(source.readframes(source.getnframes()))
            return f"segment {len(self.transcribed)}"

        def score(*args, **kwargs):
            return {"risk_score": 0, "reason": "test", "flags": [], "engine_breakdown": {}}

        # Both permanent recordings and transient ASR WAVs stay inside this sandbox.
        # Persistence and external services are stubs, not extracted server functions.
        self.scope = extract_functions(
            "inspect_audio_chunk", "normalize_pcm_to_16k_mono", "pcm_to_wav_bytes", "audio_stream",
            namespace={
                "app": SimpleNamespace(websocket=lambda route: lambda function: function),
                "asyncio": asyncio, "datetime": datetime, "time": time, "json": json,
                "tempfile": SimpleNamespace(
                    mkstemp=lambda **kwargs: tempfile.mkstemp(dir=self.root, **kwargs)
                ),
                "anyio": SimpleNamespace(to_thread=SimpleNamespace(run_sync=run_sync)),
                "WebSocketDisconnect": type("WebSocketDisconnect", (Exception,), {}),
                "logger": self.logger,
                "RECORDINGS_DIR": str(self.root), "ACTIVE_CONNECTIONS": 0,
                "LAST_CALL_DATA": {}, "STT_PROVIDER": "whisper",
                "DETECTION_MODE": "rules", "MODEL_STATE": {"llm": "offline"},
                "AUDIO_SYSTEM": "test", "has_scam_indicator": lambda text: False,
                "clean_and_deduplicate_transcript": lambda text: text.strip(),
                "transcribe_wav": transcribe, "transcribe_whisper": transcribe,
                "_score_with_mode": score, "save_last_call_data": Mock(),
                "save_call_record": Mock(),
                "compute_file_sha256": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            },
        )

    async def run_stream(self, messages):
        ws = FakeWebSocket(messages)
        await asyncio.wait_for(self.scope["audio_stream"](ws), timeout=5)
        self.assertTrue(ws.accepted)
        self.assertTrue(ws.closed)
        self.assertEqual(self.scope["ACTIVE_CONNECTIONS"], 0)
        self.logger.error.assert_not_called()
        return ws

    async def test_end_flushes_short_tail_after_a_full_chunk(self):
        full_chunk = pcm([1000] * 16000)
        tail = pcm([2000] * 1600)  # 100 ms / 3200 bytes, below the old 8000-byte gate.
        ws = await self.run_stream([
            {"bytes": full_chunk}, {"bytes": tail}, {"text": "END"},
        ])
        self.assertEqual(
            [len(chunk) for chunk in self.transcribed], [len(full_chunk), len(tail)],
            "END must transcribe the short tail separately from the full chunk",
        )
        self.assertTrue(self.transcribed[0] == full_chunk, "Full chunk PCM changed")
        self.assertTrue(self.transcribed[1] == tail, "Tail PCM changed")
        self.assertTrue(any(message.get("transcript") == "segment 2" for message in ws.sent))
        record = self.scope["save_call_record"].call_args.args[0]
        self.assertIn("segment 2", record["transcript"])
        with wave.open(str(self.root / record["audio_file"]), "rb") as recording:
            self.assertTrue(
                recording.readframes(recording.getnframes()) == full_chunk + tail,
                "Recording must preserve the full chunk and short tail",
            )

    async def test_unsupported_negotiated_encoding_returns_error_not_ready(self):
        ws = await self.run_stream([
            {"text": json.dumps({
                "type": "audio_format", "sample_rate": 16000,
                "channels": 1, "encoding": "mulaw",
            })},
            {"text": "END"},
        ])
        self.assertTrue(any(
            message.get("type") == "error" or message.get("status") == "error" or "error" in message
            for message in ws.sent
        ), f"Expected an explicit format error, received: {ws.sent!r}")
        self.assertFalse(any(message.get("status") == "ready" for message in ws.sent))
        self.assertEqual(self.transcribed, [])

    async def test_truncated_wav_returns_error_without_transcription(self):
        payload = wav_bytes(pcm([1000] * 16000))[:-2]
        ws = await self.run_stream([{"bytes": payload}, {"text": "END"}])
        self.assertTrue(any(message.get("type") == "error" for message in ws.sent))
        self.assertEqual(self.transcribed, [])
        self.scope["save_call_record"].assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_invalid_rate_or_channel_negotiation_returns_error_not_ready(self):
        cases = [
            {"sample_rate": rate} for rate in (0, -1, 7999, 192001, "invalid", None)
        ] + [
            {"channels": channels} for channels in (0, -1, 9, "invalid", None)
        ]
        for invalid in cases:
            with self.subTest(format=invalid):
                config = {
                    "type": "audio_format", "sample_rate": 16000,
                    "channels": 1, "encoding": "pcm_s16le",
                }
                config.update(invalid)
                ws = await self.run_stream([
                    {"text": json.dumps(config)}, {"text": "END"},
                ])
                self.assertTrue(any(message.get("type") == "error" for message in ws.sent))
                self.assertFalse(any(message.get("status") == "ready" for message in ws.sent))
                self.assertEqual(self.transcribed, [])
                self.scope["save_call_record"].assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_48k_stereo_wav_is_normalized_before_transcription_and_recording(self):
        sine = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
        stereo = np.column_stack((np.rint(12000 * sine), np.rint(4000 * sine)))
        # No format negotiation: the WAV header must override the raw PCM defaults.
        ws = await self.run_stream([
            {"bytes": wav_bytes(pcm(stereo), rate=48000, channels=2)}, {"text": "END"},
        ])
        self.assertFalse(any(message.get("type") == "error" for message in ws.sent))
        self.assertEqual(len(self.transcribed), 1)
        output = np.frombuffer(self.transcribed[0], dtype="<i2")
        self.assertEqual(len(output), 16000)
        rms = np.sqrt(np.mean(output.astype(np.float64) ** 2))
        self.assertAlmostEqual(rms, 8000 / math.sqrt(2), delta=100)
        expected = np.clip(np.rint(resample_poly(
            stereo.astype(np.float32).mean(axis=1), 1, 3
        )), -32768, 32767).astype("<i2")
        np.testing.assert_array_equal(output, expected)
        record = self.scope["save_call_record"].call_args.args[0]
        with wave.open(str(self.root / record["audio_file"]), "rb") as recording:
            self.assertEqual(
                (recording.getframerate(), recording.getnchannels(), recording.getsampwidth()),
                (16000, 1, 2),
            )
            self.assertTrue(
                recording.readframes(recording.getnframes()) == self.transcribed[0],
                "Recording must use the same normalized PCM as transcription",
            )

    async def test_default_inactivity_timeout_exceeds_six_second_packet_interval(self):
        original_wait_for = asyncio.wait_for
        ws = FakeWebSocket([{"bytes": pcm([1000] * 1600)}, {"text": "END"}])
        with patch.dict(os.environ):
            for name in ("CALL_INACTIVITY_TIMEOUT", "INITIAL_AUDIO_WAIT_TIMEOUT"):
                os.environ.pop(name, None)
            with patch.object(asyncio, "wait_for", wraps=original_wait_for) as wait_for:
                # Use the original for the test watchdog, so only server waits are recorded.
                await original_wait_for(self.scope["audio_stream"](ws), timeout=5)
        timeouts = [call.kwargs["timeout"] for call in wait_for.await_args_list]
        self.assertEqual(len(timeouts), 2, "Expected an initial wait and a post-audio wait")
        self.assertGreater(timeouts[0], 0)
        self.assertGreater(timeouts[1], 6.0, "Inactivity must allow six-second audio packets")
        self.assertTrue(ws.closed)
        self.assertEqual(self.scope["ACTIVE_CONNECTIONS"], 0)
        self.logger.error.assert_not_called()

    async def test_tail_temp_file_is_removed_when_transcription_fails(self):
        transcribe = self.scope["transcribe_wav"]
        attempted_paths = []

        def fail_on_tail(path):
            attempted_paths.append(Path(path))
            self.assertTrue(Path(path).resolve().is_relative_to(self.root))
            self.assertTrue(Path(path).is_file())
            if len(attempted_paths) == 2:
                raise RuntimeError("simulated tail STT failure")
            return transcribe(path)

        self.scope["transcribe_wav"] = fail_on_tail
        await self.run_stream([
            {"bytes": pcm([1000] * 16000)},
            {"bytes": pcm([2000] * 1600)},
            {"text": "END"},
        ])
        self.assertEqual(len(attempted_paths), 2, "Expected full-chunk and tail STT attempts")
        for path in attempted_paths:
            self.assertFalse(path.exists(), f"Temporary ASR WAV was not removed: {path}")
        self.logger.warning.assert_called()
        record = self.scope["save_call_record"].call_args.args[0]
        self.assertEqual(record["transcript"], "segment 1")
        self.assertEqual(
            set(self.root.iterdir()), {self.root / record["audio_file"]},
            "Only the persistent recording should remain in the sandbox",
        )


if __name__ == "__main__":
    unittest.main()
