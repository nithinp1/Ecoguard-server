"""Bounded websocket regressions; no live app imports or persistent state."""

import asyncio
import json
from pathlib import Path
import threading
import unittest
import wave

import test_audio_pipeline as pipeline

try:
    import websockets
except ImportError:
    websockets = None


class QueuedWebSocket(pipeline.FakeWebSocket):
    """Only return supplied frames; an empty queue really blocks receive()."""

    def __init__(self):
        super().__init__([])
        self.incoming = asyncio.Queue()
        self.receive_count = 0
        self.changed = asyncio.Event()
        self.blocked = asyncio.Event()
        self.closed_event = asyncio.Event()
        self.close_calls = []
        self.receiver = None
        self.receive_cancelled = False

    def feed(self, *messages):
        for message in messages:
            self.incoming.put_nowait(message)

    async def receive(self):
        self.receiver = asyncio.current_task()
        if self.incoming.empty():
            self.blocked.set()
        try:
            message = await self.incoming.get()
        except asyncio.CancelledError:
            self.receive_cancelled = True
            raise
        self.receive_count += 1
        self.changed.set()
        return message

    async def received(self, count):
        while self.receive_count < count:
            self.changed.clear()
            await asyncio.wait_for(self.changed.wait(), 2)

    async def close(self, code=1000, reason=None):
        self.closed = True
        self.close_calls.append((code, reason))
        self.closed_event.set()


class WebSocketLivenessTests(unittest.IsolatedAsyncioTestCase):
    # Reuse only the sandbox/model setup, not the existing test methods.
    setUp = pipeline.AudioStreamTests.setUp

    def start_stream(self, ws):
        task = asyncio.create_task(self.scope["audio_stream"](ws))

        async def cleanup():
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.addAsyncCleanup(cleanup)
        return task

    async def finish(self, task):
        await asyncio.wait_for(task, 5)
        self.assertEqual(self.scope["ACTIVE_CONNECTIONS"], 0)
        self.logger.error.assert_not_called()

    def gate_stt(self):
        entered, release = asyncio.Event(), asyncio.Event()
        transcribe = self.scope["transcribe_wav"]

        async def run_sync(function, *args, **kwargs):
            if function is transcribe and not entered.is_set():
                entered.set()
                await release.wait()
            return function(*args, **kwargs)

        self.scope["anyio"].to_thread.run_sync = run_sync
        self.addCleanup(release.set)
        return entered, release

    def assert_recording(self, chunks):
        self.assertEqual(self.transcribed, chunks)
        record = self.scope["save_call_record"].call_args.args[0]
        self.assertEqual(record["transcript"], "\n".join(
            f"segment {index}" for index in range(1, len(chunks) + 1)
        ))
        with wave.open(str(self.root / record["audio_file"]), "rb") as recording:
            self.assertEqual(recording.readframes(recording.getnframes()), b"".join(chunks))
        self.assertEqual(set(self.root.iterdir()), {self.root / record["audio_file"]})

    async def test_receiver_drains_during_stt_and_preserves_order_and_tail(self):
        entered, release = self.gate_stt()
        chunks = [pipeline.pcm([value] * 16000) for value in (100, 200, 300)]
        chunks.append(pipeline.pcm([400] * 1600))
        ws = QueuedWebSocket()
        ws.feed({"bytes": chunks[0]})
        task = self.start_stream(ws)
        try:
            await asyncio.wait_for(entered.wait(), 2)
            ws.feed(*({"bytes": chunk} for chunk in chunks[1:]), {"text": "END"})
            await ws.received(5)
            self.assertFalse(task.done())
            self.assertEqual(self.transcribed, [])
        finally:
            release.set()
        await self.finish(task)
        self.assertTrue(ws.receiver.done())
        self.assertTrue(ws.closed)
        self.assert_recording(chunks)
        self.assertEqual(
            [message["transcript"] for message in ws.sent if "latency_ms" in message],
            ["segment 1", "segment 2", "segment 3"],
        )
        self.assertTrue(any(message.get("transcript") == "segment 4" for message in ws.sent))

    async def test_rejected_format_cancels_and_joins_blocked_receiver(self):
        ws = QueuedWebSocket()
        ws.feed({"text": json.dumps({"type": "audio_format", "encoding": "mulaw"})})
        await self.finish(self.start_stream(ws))
        self.assertTrue(ws.blocked.is_set())
        self.assertTrue(ws.receive_cancelled)
        self.assertTrue(ws.receiver.done())
        self.assertTrue(ws.receiver.cancelled())
        self.assertTrue(ws.closed)
        self.assertTrue(any(message.get("type") == "error" for message in ws.sent))
        self.assertFalse(any(message.get("status") == "ready" for message in ws.sent))
        self.assertEqual(self.transcribed, [])
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_stream_cancellation_joins_receiver_blocked_in_receive(self):
        ws = QueuedWebSocket()
        task = self.start_stream(ws)
        await asyncio.wait_for(ws.blocked.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(ws.receive_cancelled)
        self.assertTrue(ws.receiver.done())
        self.assertTrue(ws.receiver.cancelled())
        self.assertEqual(self.scope["ACTIVE_CONNECTIONS"], 0)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_cancellation_during_initial_persistence_closes_and_joins_receiver(self):
        entered, gate, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        persist = self.scope["save_last_call_data"]

        async def run_sync(function, *args, **kwargs):
            if function is persist:
                entered.set()
                try:
                    await gate.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return function(*args, **kwargs)

        self.scope["anyio"].to_thread.run_sync = run_sync
        ws = QueuedWebSocket()
        task = self.start_stream(ws)
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(ws.blocked.wait(), 2)
        self.assertEqual(self.scope["ACTIVE_CONNECTIONS"], 1)
        self.assertFalse(ws.closed)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(gate.is_set())
        self.assertEqual(self.scope["ACTIVE_CONNECTIONS"], 0)
        self.assertTrue(ws.closed)
        self.assertTrue(ws.receive_cancelled)
        self.assertTrue(ws.receiver.done())
        self.assertTrue(ws.receiver.cancelled())
        persist.assert_not_called()
        self.scope["save_call_record"].assert_not_called()
        self.assertEqual(self.transcribed, [])
        self.assertEqual(list(self.root.iterdir()), [])

    async def check_overflow(self, messages):
        entered, release = self.gate_stt()
        ws = QueuedWebSocket()
        chunk = pipeline.pcm([100] * 16000)
        ws.feed({"bytes": chunk})
        task = self.start_stream(ws)
        try:
            await asyncio.wait_for(entered.wait(), 2)
            ws.feed(*messages)
            await asyncio.wait_for(ws.closed_event.wait(), 2)
            self.assertEqual(ws.receive_count, 1 + len(messages))
            self.assertEqual(ws.close_calls, [(1013, "Audio processing backlog exceeded")])
            errors = [message for message in ws.sent if message.get("type") == "error"]
            self.assertEqual(len(errors), 1)
            self.assertIn("backlog", errors[0]["error"].lower())
            self.assertIn("incomplete", errors[0]["error"].lower())
            self.assertEqual(self.transcribed, [])
            self.assertFalse(task.done(), "Overflow must be reported before STT is released")
        finally:
            release.set()
        await self.finish(task)
        self.assertTrue(ws.receiver.done())
        self.assert_recording([chunk])

    async def test_message_count_overflow_reports_error_and_closes_1013(self):
        # Tiny frames hit only the 256-message cap, not the byte cap.
        await self.check_overflow([{"text": "{}"}] * 257)

    async def test_audio_overflow_discards_queued_chunks_after_current_stt(self):
        # 257 full audio frames exceed the count cap but stay below 16 MiB.
        queued_chunk = pipeline.pcm([200] * 16000)
        await self.check_overflow([{"bytes": queued_chunk}] * 257)

    async def test_byte_overflow_reports_error_and_closes_1013(self):
        # Exactly 16 MiB is allowed; the third frame exceeds bytes, not count.
        padding = " " * (8 * 1024 * 1024)
        await self.check_overflow([{"text": padding}, {"text": padding}, {"text": "{}"}])

    async def check_callback_offload(self, operation):
        self.scope["anyio"].to_thread.run_sync = asyncio.to_thread
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        observations = []
        original = self.scope[operation]

        def gated(*args, **kwargs):
            callback = (kwargs.get("skip_llm") if operation == "_score_with_mode" else
                        self.scope["LAST_CALL_DATA"].get("latest_transcript") and
                        self.scope["LAST_CALL_DATA"].get("status") == "in_progress")
            if callback and not observations:
                observations.append(threading.get_ident())
                loop.call_soon_threadsafe(entered.set)
                observations.append(release.wait(2))
            return original(*args, **kwargs)

        self.scope[operation] = gated
        ws = QueuedWebSocket()
        ws.feed({"bytes": pipeline.pcm([100] * 16000)})
        task = self.start_stream(ws)
        try:
            await asyncio.wait_for(entered.wait(), 3)
            self.assertNotEqual(observations[0], loop_thread)
            self.assertEqual(len(observations), 1, "Callback must still be blocked while loop runs")
            ws.feed({"text": "END"})
            await ws.received(2)
            self.assertFalse(task.done())
        finally:
            release.set()
        await self.finish(task)
        self.assertEqual(observations[1], True, "Thread gate timed out rather than being released")

    async def test_slow_scoring_does_not_block_event_loop(self):
        await self.check_callback_offload("_score_with_mode")

    async def test_slow_callback_persistence_does_not_block_event_loop(self):
        await self.check_callback_offload("save_last_call_data")

    @unittest.skipIf(websockets is None, "websockets is not installed locally")
    async def test_protocol_ping_pong_during_blocked_stt_with_tiny_receive_queue(self):
        from websockets.exceptions import ConnectionClosed
        from websockets.legacy.client import connect
        from websockets.legacy.server import serve

        self.scope["anyio"].to_thread.run_sync = asyncio.to_thread
        entered, handler_done = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        transcribe = self.scope["transcribe_wav"]
        gates = []
        failures = []

        def blocked_transcribe(path):
            if not gates:
                loop.call_soon_threadsafe(entered.set)
                gates.append(release.wait(3))
            return transcribe(path)

        self.scope["transcribe_wav"] = blocked_transcribe
        self.addCleanup(release.set)

        class Adapter:
            def __init__(self, socket):
                self.socket = socket

            async def accept(self):
                pass

            async def receive(self):
                try:
                    frame = await self.socket.recv()
                except ConnectionClosed:
                    return {"type": "websocket.disconnect"}
                return {"bytes" if isinstance(frame, bytes) else "text": frame}

            async def send_json(self, message):
                await self.socket.send(json.dumps(message))

            async def close(self, code=1000, reason=None):
                await self.socket.close(code=code, reason=reason or "")

        async def handler(socket, path):
            try:
                await asyncio.wait_for(self.scope["audio_stream"](Adapter(socket)), 6)
            except Exception as error:
                failures.append(error)
            finally:
                handler_done.set()

        chunks = [pipeline.pcm([value] * 16000) for value in range(100, 108)]
        chunks.append(pipeline.pcm([200] * 1600))
        async with serve(handler, "127.0.0.1", 0, max_queue=1, ping_interval=None,
                         close_timeout=0.2) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", ping_interval=None,
                               close_timeout=0.2, open_timeout=2) as client:
                try:
                    await client.send(chunks[0])
                    await asyncio.wait_for(entered.wait(), 2)
                    for chunk in chunks[1:]:
                        await client.send(chunk)
                    # Ping follows enough data frames to fill the legacy protocol queue.
                    pong = await client.ping(b"stt-is-blocked")
                    await asyncio.wait_for(pong, 1)
                    self.assertFalse(release.is_set())
                    self.assertEqual(gates, [], "STT must remain blocked until after pong")
                    await client.send("END")
                finally:
                    release.set()

                async def drain():
                    return [json.loads(message) async for message in client]

                replies = await asyncio.wait_for(drain(), 5)
                await asyncio.wait_for(handler_done.wait(), 2)
        self.assertEqual(failures, [])
        self.assertEqual(gates, [True])
        self.assertEqual(self.scope["ACTIVE_CONNECTIONS"], 0)
        self.logger.error.assert_not_called()
        self.assert_recording(chunks)
        self.assertTrue(any(message.get("transcript") == "segment 9" for message in replies))


class DockerWebSocketSettingsTests(unittest.TestCase):
    def test_exec_form_cmd_enables_twenty_second_protocol_pings(self):
        dockerfile = Path(pipeline.SERVER_PATH).with_name("Dockerfile")
        commands = [line[4:] for line in dockerfile.read_text(encoding="utf-8").splitlines()
                    if line.startswith("CMD ")]
        self.assertEqual(len(commands), 1)
        command = json.loads(commands[0])
        self.assertIsInstance(command, list)
        self.assertIn("uvicorn", command)
        self.assertEqual(command.count("--ws-ping-interval"), 1)
        self.assertEqual(command[command.index("--ws-ping-interval") + 1], "20")
        self.assertEqual(command[command.index("--workers") + 1], "1")


if __name__ == "__main__":
    unittest.main()
