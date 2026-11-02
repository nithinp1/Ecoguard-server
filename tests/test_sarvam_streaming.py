"""Mocked Sarvam protocol regressions; never import server or contact a provider.

Run: python -B -m unittest discover -s tests -p test_sarvam_streaming.py -v
"""

import ast
import asyncio
import base64
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
import unittest
from unittest.mock import AsyncMock, Mock, patch
import urllib.parse


SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
WATCHDOG = 2.0
SESSION_BEGIN = {
    "event": "session.begin",
    "config": {
        "sample_rate": 16000, "encoding": "linear16",
        "stream_type": "fast", "endpointing": "vad",
    },
}
CHUNK_ERROR = {
    "event": "error", "code": "chunk_too_large",
    "message": "Audio exceeds per-frame cap 16000 bytes for stream_type fast",
    "is_fatal": False, "received_bytes": 96256, "max_bytes": 16000,
}


def extract_client(logger):
    """Execute only the class, with explicit standard-library dependencies."""
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), str(SERVER_PATH))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AsyncSarvamStreamingClient"
    ]
    if len(classes) != 1:
        raise AssertionError("Expected one AsyncSarvamStreamingClient class")
    module = ast.Module(body=classes, type_ignores=[])
    scope = {
        "asyncio": asyncio, "json": json, "base64": base64,
        "logging": logging, "logger": logger, "time": time,
        "urllib": urllib, "os": os, "Any": Any, "Callable": Callable,
        "Dict": Dict, "List": List, "Optional": Optional,
    }
    exec(compile(ast.fix_missing_locations(module), str(SERVER_PATH), "exec"), scope)
    return scope["AsyncSarvamStreamingClient"]


class FakeProviderSocket:
    """Keep the transport open after session.end, like the real provider can."""

    def __init__(self):
        self.incoming = asyncio.Queue()
        self.reading = asyncio.Event()
        self.end_sent = asyncio.Event()
        self.closed = False
        self.sent = []
        self.fail_event = None
        self.fail_audio_at = None
        self.audio_attempts = 0

    def feed(self, message):
        self.incoming.put_nowait(json.dumps(message))

    async def barrier(self):
        # Reaching the next iterator item means the previous callback completed.
        reached = asyncio.Event()
        self.incoming.put_nowait(reached)
        await asyncio.wait_for(reached.wait(), WATCHDOG)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            self.reading.set()
            item = await self.incoming.get()
            if item is None:
                raise StopAsyncIteration
            if isinstance(item, Exception):
                raise item
            if isinstance(item, asyncio.Event):
                item.set()
                continue
            return item

    async def send(self, raw):
        message = json.loads(raw)
        event = message.get("event")
        if event == "audio_input":
            self.audio_attempts += 1
        if event == self.fail_event or (
            event == "audio_input" and self.audio_attempts == self.fail_audio_at
        ):
            raise OSError("mock provider send failure")
        self.sent.append(message)
        if event == "end":
            self.end_sent.set()

    async def close(self):
        self.closed = True
        self.incoming.put_nowait(None)


class SarvamStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logger = Mock()
        self.client_type = extract_client(self.logger)
        self.socket = FakeProviderSocket()
        self.handshake = asyncio.Event()
        self.connect_calls = []
        self.clients = []
        self.tasks = []

        async def connect(url, **kwargs):
            self.connect_calls.append((url, kwargs))
            self.handshake.set()
            return self.socket

        module_patch = patch.dict(sys.modules, {"websockets": SimpleNamespace(connect=connect)})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.addAsyncCleanup(self.cleanup_transport)

    async def cleanup_transport(self):
        await self.socket.close()
        tasks = self.tasks + [
            client._rx_task for client in self.clients if client._rx_task is not None
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def client(self, **kwargs):
        client = self.client_type(api_key="mock-only-not-a-real-key", **kwargs)
        self.clients.append(client)
        return client

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def connected(self, **kwargs):
        client = self.client(**kwargs)
        self.socket.feed(SESSION_BEGIN)
        self.assertIs(await asyncio.wait_for(client.connect(), WATCHDOG), True)
        await self.socket.barrier()
        return client

    def assert_failed(self, client):
        self.assertIs(client.failed, True)
        self.assertFalse(client._is_connected)
        self.assertTrue(client.last_error, "Failure must retain actionable diagnostics")

    def assert_joined(self, client):
        self.assertTrue(self.socket.closed, "close must always close the transport")
        self.assertTrue(client._rx_task.done(), "close must join, not just cancel, its receiver")

    async def test_connect_waits_for_session_begin_not_raw_websocket_handshake(self):
        client = self.client()
        connecting = self.start(client.connect())
        await asyncio.wait_for(self.handshake.wait(), WATCHDOG)
        await asyncio.wait_for(self.socket.reading.wait(), WATCHDOG)
        self.assertFalse(connecting.done(), "Raw handshake is not provider readiness")
        self.assertIsInstance(client._ready, asyncio.Event)
        self.assertFalse(client._ready.is_set())
        self.socket.feed(SESSION_BEGIN)
        self.assertIs(await asyncio.wait_for(connecting, WATCHDOG), True)
        self.assertTrue(client._ready.is_set())
        self.assertFalse(client.failed)

    async def test_connect_error_wakes_readiness_wait_and_returns_false(self):
        client = self.client()
        connecting = self.start(client.connect())
        await asyncio.wait_for(self.socket.reading.wait(), WATCHDOG)
        self.socket.feed(CHUNK_ERROR)
        self.assertIs(await asyncio.wait_for(connecting, WATCHDOG), False)
        self.assert_failed(client)

    async def test_live_96256_byte_packet_is_split_without_protocol_or_pcm_mutation(self):
        client = await self.connected()
        payload = bytes(range(256)) * 376
        self.assertEqual(len(payload), 96256)
        self.assertIs(await client.send_audio_chunk(payload), True)
        messages = self.socket.sent
        self.assertTrue(all(set(message) == {"event", "audio"} for message in messages))
        self.assertTrue(all(message["event"] == "audio_input" for message in messages))
        frames = [base64.b64decode(message["audio"], validate=True) for message in messages]
        self.assertEqual([len(frame) for frame in frames], [3200] * 30 + [256])
        self.assertTrue(all(len(frame) <= 16000 for frame in frames))
        self.assertEqual(b"".join(frames), payload)
        url, kwargs = self.connect_calls[0]
        self.assertEqual(urllib.parse.urlsplit(url).path, "/speech-to-text-realtime/ws")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(query["encoding"], ["linear16"])
        self.assertEqual(query["sample_rate"], ["16000"])
        self.assertEqual(query["endpointing"], ["vad"])
        headers = kwargs.get("additional_headers", kwargs.get("extra_headers"))
        self.assertEqual(headers, {"Api-Subscription-Key": "mock-only-not-a-real-key"})

    async def test_frame_duration_tracks_configured_sample_rate(self):
        client = await self.connected(sample_rate=8000)
        payload = b"\x01\x02" * 1700
        self.assertIs(await client.send_audio_chunk(payload), True)
        frames = [base64.b64decode(message["audio"]) for message in self.socket.sent]
        self.assertEqual([len(frame) for frame in frames], [1600, 1600, 200])
        self.assertEqual(b"".join(frames), payload)

    async def test_disconnected_send_returns_false_without_sending(self):
        client = self.client()
        self.assertIs(await client.send_audio_chunk(b"\x00\x00"), False)
        self.assertEqual(self.socket.sent, [])

    async def test_mid_chunk_send_failure_stops_remaining_frames(self):
        client = await self.connected()
        self.socket.fail_audio_at = 2
        self.assertIs(await client.send_audio_chunk(b"\x01\x02" * 4800), False)
        self.assertEqual(self.socket.audio_attempts, 2)
        self.assertEqual(len(self.socket.sent), 1)
        self.assert_failed(client)
        self.assertIs(await client.send_audio_chunk(b"\x00\x00"), False)
        self.assertEqual(self.socket.audio_attempts, 2)

    async def test_nonfatal_chunk_too_large_is_not_silently_ignored(self):
        client = await self.connected()
        self.socket.feed(CHUNK_ERROR)
        # Both continuing to read and terminating on a provider error are valid.
        barrier = self.start(self.socket.barrier())
        done, _ = await asyncio.wait(
            (barrier, client._rx_task), timeout=WATCHDOG,
            return_when=asyncio.FIRST_COMPLETED,
        )
        self.assertTrue(done, "Provider error was not consumed")
        if barrier in done:
            await barrier
        else:
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)
        self.assert_failed(client)
        self.assertIn("chunk_too_large", str(client.last_error))
        self.assertTrue(self.logger.warning.called or self.logger.error.called)
        self.assertIs(await client.send_audio_chunk(b"\x00\x00"), False)

    async def test_partial_and_final_use_official_text_index_language_fields(self):
        partial = Mock()
        final_calls = []

        async def final(text, idx, lang):
            await asyncio.sleep(0)
            final_calls.append((text, idx, lang))

        await self.connected(on_partial=partial, on_final=final)
        self.socket.feed({
            "event": "transcript.partial", "text": "namaste", "utterance_idx": 7,
            "language": "hi-IN",
        })
        self.socket.feed({
            "event": "transcript.final", "text": "namaste duniya", "utterance_idx": 7,
            "language": "hi-IN",
        })
        await self.socket.barrier()
        partial.assert_called_once_with("namaste", 7, "hi-IN")
        self.assertEqual(final_calls, [("namaste duniya", 7, "hi-IN")])

    async def test_close_drains_delayed_final_then_session_end_even_if_socket_stays_open(self):
        calls = []

        async def final(text, idx, lang):
            await asyncio.sleep(0)
            calls.append((text, idx, lang))

        client = await self.connected(on_final=final)
        closing = self.start(client.close())
        await asyncio.wait_for(self.socket.end_sent.wait(), WATCHDOG)
        self.assertFalse(closing.done())
        self.assertFalse(self.socket.closed, "Transport closed before provider finalization")
        self.socket.feed({
            "event": "transcript.final", "text": "delayed final", "utterance_idx": 0,
            "language": "en-IN",
        })
        await self.socket.barrier()
        self.assertEqual(calls, [("delayed final", 0, "en-IN")])
        self.assertFalse(closing.done(), "A final transcript alone is not session completion")
        self.socket.feed({"event": "session.end"})
        self.assertIs(await asyncio.wait_for(closing, WATCHDOG), True)
        self.assertFalse(client.failed)
        self.assert_joined(client)
        self.assertEqual(self.socket.sent, [{"event": "end"}])

    async def test_close_end_send_failure_still_closes_and_joins(self):
        client = await self.connected()
        self.socket.fail_event = "end"
        self.assertIs(await asyncio.wait_for(client.close(), WATCHDOG), False)
        self.assert_failed(client)
        self.assert_joined(client)

    async def test_close_missing_session_end_times_out_and_reports_failure(self):
        client = await self.connected()
        real_wait_for = asyncio.wait_for
        waits = []

        async def short_wait(awaitable, timeout):
            waits.append(timeout)
            return await real_wait_for(awaitable, min(timeout, 0.03))

        # Exercise the real timeout path without a five-second unit-test delay.
        with patch.object(asyncio, "wait_for", side_effect=short_wait):
            result = await real_wait_for(client.close(), WATCHDOG)
        self.assertIs(result, False)
        self.assertTrue(waits, "Finalization must use a bounded wait")
        self.assertTrue(all(0 < timeout <= 5 for timeout in waits), waits)
        self.assert_failed(client)
        self.assert_joined(client)

    async def test_close_stalled_end_send_times_out_and_still_closes_and_joins(self):
        client = await self.connected()
        real_wait_for = asyncio.wait_for
        waits = []
        send_started = asyncio.Event()
        send_cancelled = asyncio.Event()
        never_sent = asyncio.Event()

        async def stalled_send(raw):
            self.assertEqual(json.loads(raw), {"event": "end"})
            send_started.set()
            try:
                await never_sent.wait()
            except asyncio.CancelledError:
                send_cancelled.set()
                raise

        async def short_wait(awaitable, timeout):
            waits.append(timeout)
            return await real_wait_for(awaitable, min(timeout, 0.03))

        # The deadline must include sending end, not just receiving session.end.
        with patch.object(self.socket, "send", side_effect=stalled_send), patch.object(
            asyncio, "wait_for", side_effect=short_wait
        ):
            result = await real_wait_for(client.close(), WATCHDOG)
        self.assertIs(result, False)
        self.assertTrue(send_started.is_set())
        self.assertTrue(send_cancelled.is_set(), "Timed-out end send must not remain pending")
        self.assertTrue(waits, "Finalization must bound a backpressured end send")
        self.assertTrue(all(0 < timeout <= 5 for timeout in waits), waits)
        self.assert_failed(client)
        self.assert_joined(client)

    async def test_receiver_termination_without_session_end_is_failure(self):
        for termination in (None, OSError("mock receive failure")):
            with self.subTest(termination=termination):
                # Each subcase needs its own live provider queue and receiver.
                self.socket = FakeProviderSocket()
                client = await self.connected()
                self.socket.incoming.put_nowait(termination)
                await asyncio.wait_for(client._rx_task, WATCHDOG)
                self.assert_failed(client)
                self.assertIs(await asyncio.wait_for(client.close(), WATCHDOG), False)
                self.assert_joined(client)

    async def test_flush_is_vad_noop_and_manual_only(self):
        client = await self.connected()
        await client.flush()
        self.assertEqual(self.socket.sent, [], "VAD endpointing must not send manual flush")
        client.endpointing = "manual"
        await client.flush()
        self.assertEqual(self.socket.sent, [{"event": "flush"}])


# Reuse the existing endpoint sandbox without inheriting/collecting its tests here.
_pipeline_spec = importlib.util.spec_from_file_location(
    "_sarvam_audio_pipeline_sandbox", Path(__file__).with_name("test_audio_pipeline.py")
)
pipeline = importlib.util.module_from_spec(_pipeline_spec)
_pipeline_spec.loader.exec_module(pipeline)


class SarvamEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pipeline.AudioStreamTests.setUp(self)
        self.scope.update({
            "STT_PROVIDER": "sarvam", "SARVAM_MODEL": "saaras:v4",
            "SARVAM_MODE": "transcribe", "SARVAM_LANGUAGE": "auto",
        })
        environment = patch.dict(os.environ, {"SARVAM_API_KEY": "mock-only-not-a-real-key"})
        environment.start()
        self.addCleanup(environment.stop)

    async def exercise(self, *, text=None, failed=False):
        instances = []

        class FakeSarvam:
            def __init__(inner, **kwargs):
                inner.on_final = kwargs["on_final"]
                inner.failed = False
                inner.last_error = None
                inner.chunks = []
                inner.closed = False
                instances.append(inner)

            async def connect(inner):
                return True

            async def send_audio_chunk(inner, payload):
                inner.chunks.append(payload)
                if text and len(inner.chunks) == 1:
                    await inner.on_final(text, 0, "en-IN")
                return True

            async def flush(inner):
                pass

            async def close(inner):
                inner.closed = True
                # Fail at finalization: send succeeded and history may already exist.
                inner.failed = failed
                inner.last_error = "mock finalization failure" if failed else None
                return not failed

        self.scope["AsyncSarvamStreamingClient"] = FakeSarvam
        payload = (
            b"\xe8\x03" * 16000 + b"\xd0\x07" * 1600
            if text else b"\x00\x00" * 17600
        )
        ws = await pipeline.AudioStreamTests.run_stream(self, [
            {"bytes": payload[:32000]}, {"bytes": payload[32000:]}, {"text": "END"},
        ])
        self.assertEqual(len(instances), 1)
        self.assertTrue(instances[0].closed)
        self.assertEqual(b"".join(instances[0].chunks), payload)
        return ws, payload

    async def test_valid_stream_uses_streamed_transcript_without_bulk_rest(self):
        ws, _ = await self.exercise(text="streamed words")
        self.assertEqual(self.transcribed, [], "Valid realtime STT must not call bulk REST")
        record = self.scope["save_call_record"].call_args.args[0]
        self.assertEqual(record["transcript"], "streamed words")
        self.assertEqual(ws.sent[-1]["transcript"], "streamed words")
        for metadata in (ws.sent[-1], record):
            self.assertEqual(metadata["stt_transport"], "sarvam_realtime")
            self.assertIs(metadata["realtime_failed"], False)

    async def test_failed_stream_with_history_recovers_full_call_without_duplicate_text(self):
        ws, payload = await self.exercise(text="stale streamed prefix", failed=True)
        self.assertEqual(len(self.transcribed), 1, "Failure must recover the complete call once")
        self.assertTrue(self.transcribed[0] == payload, "Recovery PCM must match the complete call")
        record = self.scope["save_call_record"].call_args.args[0]
        self.assertEqual(record["transcript"], "segment 1")
        self.assertEqual(ws.sent[-1]["transcript"], "segment 1")
        self.assertNotIn("stale streamed prefix", record["transcript"])
        for metadata in (ws.sent[-1], record):
            self.assertEqual(metadata["stt_transport"], "batch_recovery")
            self.assertIs(metadata["realtime_failed"], True)

    async def test_setup_failure_uses_batch_chunks_without_redundant_full_call_recovery(self):
        client = SimpleNamespace(
            connect=AsyncMock(return_value=False),
            send_audio_chunk=AsyncMock(),
            failed=True,
            last_error="mock session setup failure",
        )
        factory = Mock(return_value=client)
        self.scope["AsyncSarvamStreamingClient"] = factory
        full_chunk = b"\xe8\x03" * 16000
        tail = b"\xd0\x07" * 1600
        ws = await pipeline.AudioStreamTests.run_stream(self, [
            {"bytes": full_chunk}, {"bytes": tail}, {"text": "END"},
        ])
        factory.assert_called_once()
        client.connect.assert_awaited_once_with()
        client.send_audio_chunk.assert_not_called()
        self.assertEqual(
            [len(chunk) for chunk in self.transcribed], [len(full_chunk), len(tail)],
            "Setup failure must batch each chunk once, without full-call recovery",
        )
        self.assertTrue(self.transcribed[0] == full_chunk, "Batch fallback changed full-chunk PCM")
        self.assertTrue(self.transcribed[1] == tail, "Batch fallback changed tail PCM")
        record = self.scope["save_call_record"].call_args.args[0]
        self.assertEqual(record["transcript"].split(), ["segment", "1", "segment", "2"])
        self.assertEqual(ws.sent[-1]["transcript"], record["transcript"])
        for metadata in (ws.sent[-1], record):
            self.assertEqual(metadata["stt_transport"], "batch_fallback")
            self.assertIs(metadata["realtime_failed"], True)

    async def test_successful_silent_stream_does_not_trigger_bulk_rest(self):
        ws, _ = await self.exercise()
        self.assertEqual(len(self.transcribed), 0, "Empty successful stream is not provider failure")
        record = self.scope["save_call_record"].call_args.args[0]
        self.assertEqual(record["transcript"], "")
        self.assertEqual(ws.sent[-1]["transcript"], "")
        for metadata in (ws.sent[-1], record):
            self.assertEqual(metadata["stt_transport"], "sarvam_realtime")
            self.assertIs(metadata["realtime_failed"], False)


if __name__ == "__main__":
    unittest.main()
