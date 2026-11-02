"""Isolated readiness regressions: no server import, models, network or persisted state.

Run: python -B -m unittest discover -s tests -p test_model_readiness.py -v
"""

import ast
import asyncio
from datetime import datetime
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock
import wave

from runtime_config import RuntimeConfigError


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"


class BackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, function, *args):
        self.tasks.append((function, args))

    async def run(self):
        for function, args in self.tasks:
            await function(*args)


def config(**values):
    fields = (
        "mode stt_provider whisper_model sarvam_model sarvam_mode sarvam_language "
        "sarvam_api_key nvidia_api_key nvidia_asr_endpoint nvidia_asr_model "
        "ollama_host ollama_model"
    ).split()
    return SimpleNamespace(**{field: values.get(field) for field in fields})


def isolated_server():
    """Execute only the selected function bodies, with all services replaced."""
    names = {
        "_stt_target", "_reset_stt_readiness", "_warm_selected_stt",
        "_warm_stt_background", "warmup_all_models", "_warm_llm_background",
        "_dashboard_config_updates", "update_config",
    }
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), str(SERVER_PATH))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    if {node.name for node in nodes} != names:
        raise AssertionError("Missing readiness functions")
    for node in nodes:
        node.decorator_list = []
    environment = {}
    whisper = Mock()
    whisper.transcribe.return_value = (iter(()), None)
    nemo = Mock(device="cuda:0")
    nemo.transcribe.return_value = [""]
    fake_os = SimpleNamespace(
        getenv=lambda name, default="": environment.get(name, default),
        environ=environment,
        fdopen=MagicMock(), unlink=Mock(),
    )
    scope = {
        "asyncio": asyncio, "io": io, "wave": wave, "os": fake_os,
        "tempfile": SimpleNamespace(mkstemp=Mock(return_value=(42, "silence.wav"))),
        "datetime": datetime, "logger": Mock(),
        "anyio": SimpleNamespace(to_thread=SimpleNamespace(run_sync=asyncio.to_thread)),
        "requests": SimpleNamespace(post=Mock(return_value=SimpleNamespace(status_code=200))),
        "STT_PROVIDER": "whisper", "WHISPER_MODEL_SIZE": "medium",
        "SARVAM_MODEL": "saaras:v4", "SARVAM_MODE": "transcribe", "SARVAM_LANGUAGE": "auto",
        "NVIDIA_ASR_MODEL": "nvidia/parakeet-ctc-0.6b", "NVIDIA_ASR_ENDPOINT": "local_in_process",
        "DETECTION_MODE": "all", "OLLAMA_MODEL": "phi3.5", "OLLAMA_HOST": "localhost",
        "MODEL_STATE": {"status": "hot", "stt": "hot", "llm": "hot", "device": "cpu"},
        "_stt_state_lock": threading.RLock(), "_stt_warmup_lock": threading.Lock(),
        "_stt_revision": 0, "model_lock": threading.RLock(),
        "_local_asr_lock": threading.RLock(),
        "_whisper_model": None, "_device_used": "cpu",
        "get_model": Mock(return_value=whisper), "get_nemotron_model": Mock(return_value=nemo),
        "reload_whisper_model": Mock(side_effect=AssertionError("Blocking config reload")),
        "RUNTIME_CONFIG_STORE": SimpleNamespace(save=Mock()), "config_lock": threading.RLock(),
        "RuntimeConfigError": RuntimeConfigError, "HTTPException": HTTPException,
        "BackgroundTasks": BackgroundTasks, "ConfigUpdate": SimpleNamespace,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(SERVER_PATH), "exec"), scope)
    return scope


class ModelReadinessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.s = isolated_server()
        self.state = self.s["MODEL_STATE"]

    def apply(self, **values):
        tasks = BackgroundTasks()
        result = self.s["update_config"](config(**values), tasks)
        self.assertTrue(result["success"])
        return tasks

    def request(self):
        with self.s["_stt_state_lock"]:
            return self.s["_reset_stt_readiness"]()

    async def test_whisper_startup_executes_inference_and_consumes_segments(self):
        consumed = []

        def segments():
            consumed.append(True)
            yield ""

        model = self.s["get_model"].return_value
        model.transcribe.return_value = (segments(), None)
        await self.s["warmup_all_models"]()
        self.assertEqual(consumed, [True])
        self.assertEqual(self.state["stt"], "hot")
        self.assertTrue(self.state["inference_ready"])
        self.assertEqual(self.state["device"], "cpu")
        self.s["get_nemotron_model"].assert_not_called()
        buf = model.transcribe.call_args.args[0]
        buf.seek(0)
        with wave.open(buf) as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
            self.assertEqual(audio.readframes(16000), b"\x00\x00" * 16000)
        self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])

    async def test_whisper_lazy_inference_failure_is_not_hot(self):
        def broken_segments():
            raise RuntimeError("CUDA inference failed")
            yield

        self.s["get_model"].return_value.transcribe.return_value = (broken_segments(), None)
        await self.s["warmup_all_models"]()
        self.assertIn("CUDA inference failed", self.state["stt"])
        self.assertEqual(self.state["status"], "error")
        self.assertFalse(self.state["inference_ready"])
        self.assertEqual(self.state["llm"], "hot")

    async def test_nemo_startup_only_loads_selected_model_and_runs_inference(self):
        for provider in ("parakeet", "nemotron", "nvidia", "nemo"):
            with self.subTest(provider=provider):
                self.s = isolated_server()
                self.state = self.s["MODEL_STATE"]
                self.s.update(STT_PROVIDER=provider, NVIDIA_ASR_MODEL="selected/model")
                self.assertEqual(self.s["_stt_target"](), (provider, "selected/model"))
                await self.s["warmup_all_models"]()
                self.assertEqual(self.state["stt"], "hot")
                self.assertEqual(self.state["status"], "hot")
                self.s["get_model"].assert_not_called()
                self.s["get_nemotron_model"].assert_called_once_with("selected/model")
                self.s["get_nemotron_model"].return_value.transcribe.assert_called_once_with(
                    ["silence.wav"], verbose=False)
                self.s["os"].unlink.assert_called_once_with("silence.wav")
                payload = self.s["os"].fdopen.return_value.__enter__.return_value.write.call_args.args[0]
                with wave.open(io.BytesIO(payload)) as audio:
                    self.assertEqual(audio.getframerate(), 16000)
                    self.assertEqual(audio.getnframes(), 16000)
                self.assertEqual(self.state["device"], "cuda:0")
                self.assertEqual(self.s["_device_used"], "cpu", "NeMo must not change Whisper's device")
                self.assertTrue(self.state["inference_ready"])

    async def test_nemo_inference_failure_cleans_up_and_is_not_hot(self):
        self.s["STT_PROVIDER"] = "parakeet"
        self.s["get_nemotron_model"].return_value.transcribe.side_effect = RuntimeError("bad inference")
        await self.s["warmup_all_models"]()
        self.assertIn("bad inference", self.state["stt"])
        self.assertEqual(self.state["status"], "error")
        self.s["os"].unlink.assert_called_once_with("silence.wav")
        self.s["get_model"].assert_not_called()

    async def test_sarvam_configured_does_not_probe_cloud_or_claim_inference_ready(self):
        self.s.update(STT_PROVIDER="sarvam", DETECTION_MODE="rules_only")
        self.s["os"].environ["SARVAM_API_KEY"] = "test-secret"
        await self.s["warmup_all_models"]()
        self.assertEqual(self.state["stt"], "configured")
        self.assertEqual(self.state["status"], "hot")  # Dashboard compatibility only.
        self.assertFalse(self.state["inference_ready"])
        self.assertIsNone(self.state["device"])
        self.s["requests"].post.assert_not_called()
        self.s["get_model"].assert_not_called()
        self.s["get_nemotron_model"].assert_not_called()

    async def test_sarvam_missing_key_is_error(self):
        self.s["STT_PROVIDER"] = "sarvam"
        await self.s["warmup_all_models"]()
        self.assertIn("SARVAM_API_KEY", self.state["stt"])
        self.assertEqual(self.state["status"], "error")

    async def test_ollama_is_only_warmed_in_enabled_modes(self):
        for mode in ("all", "llm_only", "rulebased_only", "xgboost_only"):
            with self.subTest(mode=mode):
                self.s = isolated_server()
                self.s["DETECTION_MODE"] = mode
                await self.s["warmup_all_models"]()
                enabled = mode in ("all", "llm_only")
                self.assertEqual(self.s["requests"].post.call_count, int(enabled))
                self.assertEqual(self.s["MODEL_STATE"]["llm"], "hot" if enabled else "offline")

    async def test_ollama_failure_does_not_overwrite_stt_readiness(self):
        self.s["requests"].post.side_effect = RuntimeError("offline")
        await self.s["warmup_all_models"]()
        self.assertEqual(self.state["llm"], "offline")
        self.assertEqual(self.state["stt"], "hot")
        self.assertEqual(self.state["status"], "hot")

    async def test_disabled_startup_then_enabled_mode_queues_llm_warmup(self):
        for mode in ("all", "llm_only"):
            with self.subTest(mode=mode):
                self.s = isolated_server()
                self.state = self.s["MODEL_STATE"]
                self.s["DETECTION_MODE"] = "rulebased_only"
                await self.s["warmup_all_models"]()
                self.assertEqual(self.state["llm"], "offline")
                self.s["requests"].post.assert_not_called()
                tasks = self.apply(mode=mode)
                self.assertEqual(self.state["llm"], "loading")
                self.assertEqual(len(tasks.tasks), 1)
                self.s["requests"].post.assert_not_called()
                await tasks.run()
                self.assertEqual(self.state["llm"], "hot")
                self.assertEqual(self.state["stt"], "hot")
                self.s["requests"].post.assert_called_once()

    async def test_enabled_llm_target_changes_queue_warmup_for_new_target(self):
        for values in ({"ollama_host": "new-host"}, {"ollama_model": "new-model"}):
            with self.subTest(values=values):
                self.s = isolated_server()
                self.state = self.s["MODEL_STATE"]
                tasks = self.apply(**values)
                self.assertEqual(self.state["llm"], "loading")
                self.s["requests"].post.assert_not_called()
                await tasks.run()
                request = self.s["requests"].post.call_args
                self.assertEqual(request.args[0], f"http://{self.s['OLLAMA_HOST']}:11434/api/generate")
                self.assertEqual(request.kwargs["json"]["model"], self.s["OLLAMA_MODEL"])
                self.assertEqual(self.state["llm"], "hot")
                self.s["get_model"].assert_not_called()

    async def test_disabled_llm_target_change_stays_offline_without_warmup(self):
        tasks = self.apply(mode="rulebased_only", ollama_host="new-host", ollama_model="new-model")
        self.assertEqual(tasks.tasks, [])
        self.assertEqual(self.state["llm"], "offline")
        await self.s["_warm_llm_background"]()
        self.s["requests"].post.assert_not_called()

    async def test_enabled_mode_or_unchanged_target_does_not_rewarm_llm(self):
        tasks = self.apply(mode="llm_only", ollama_host="localhost", ollama_model="phi3.5")
        self.assertEqual(tasks.tasks, [])
        self.assertEqual(self.state["llm"], "hot")

    async def test_superseded_queued_llm_warmup_does_not_make_request(self):
        old = self.apply(ollama_model="old-model")
        current = self.apply(ollama_model="current-model")
        await old.run()
        self.s["requests"].post.assert_not_called()
        self.assertEqual(self.state["llm"], "loading")
        await current.run()
        self.assertEqual(self.state["llm"], "hot")
        self.assertEqual(self.s["requests"].post.call_args.kwargs["json"]["model"], "current-model")

    async def test_stale_llm_response_cannot_publish_after_config_change(self):
        for change in ("disable", "target", "disable_reenable", "target_away_back"):
            for fail in (False, True):
                with self.subTest(change=change, fail=fail):
                    self.s = isolated_server()
                    self.state = self.s["MODEL_STATE"]
                    started, release = threading.Event(), threading.Event()

                    def post(*args, **kwargs):
                        if not started.is_set():
                            started.set()
                            if not release.wait(5):
                                raise RuntimeError("test release timed out")
                            if fail:
                                raise RuntimeError("superseded HTTP failure")
                        return SimpleNamespace(status_code=200)

                    self.s["requests"].post.side_effect = post
                    old = asyncio.create_task(self.s["_warm_llm_background"]())
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 3))
                        if change in ("disable", "disable_reenable"):
                            current = self.apply(mode="rulebased_only")
                            self.assertEqual(self.state["llm"], "offline")
                            if change == "disable_reenable":
                                current = self.apply(mode="all")
                        else:
                            current = self.apply(ollama_host="new-host", ollama_model="new-model")
                            if change == "target_away_back":
                                current = self.apply(ollama_host="localhost", ollama_model="phi3.5")
                        expected = "offline" if change == "disable" else "loading"
                        self.assertEqual(self.state["llm"], expected)
                        self.assertEqual(self.s["requests"].post.call_count, 1)
                        # The old response must not overwrite either loading or newer readiness.
                        if fail:
                            await current.run()
                            expected = "offline" if change == "disable" else "hot"
                        release.set()
                        await old
                        self.assertEqual(self.state["llm"], expected)
                        if not fail:
                            await current.run()
                            self.assertEqual(self.state["llm"], "offline" if change == "disable" else "hot")
                        first_request = self.s["requests"].post.call_args_list[0]
                        self.assertEqual(first_request.args[0], "http://localhost:11434/api/generate")
                        self.assertEqual(first_request.kwargs["json"]["model"], "phi3.5")
                    finally:
                        release.set()
                        await old

    async def test_provider_switch_resets_and_defers_work_until_after_response(self):
        tasks = self.apply(stt_provider="parakeet")
        self.assertEqual(len(tasks.tasks), 1)
        self.assertEqual(self.state["stt"], "loading")
        self.assertEqual(self.state["status"], "loading")
        self.assertIsNone(self.state["device"])
        self.s["get_model"].assert_not_called()
        self.s["get_nemotron_model"].assert_not_called()
        await tasks.run()
        self.assertEqual(self.state["stt"], "hot")
        self.s["requests"].post.assert_not_called()  # Config STT warmup is independent.

    async def test_whisper_size_change_does_not_call_blocking_reload(self):
        tasks = self.apply(whisper_model="small")
        self.assertEqual(self.s["WHISPER_MODEL_SIZE"], "small")
        self.s["reload_whisper_model"].assert_not_called()
        self.s["get_model"].assert_not_called()
        await tasks.run()
        self.assertEqual(self.state["stt"], "hot")

    async def test_nemo_model_switch_uses_explicit_target_without_loader_lock(self):
        self.s["STT_PROVIDER"] = "parakeet"
        # No nemotron_lock is installed: the config route must never wait for it.
        tasks = self.apply(nvidia_asr_model="custom/model")
        await tasks.run()
        self.s["get_nemotron_model"].assert_called_once_with("custom/model")

    async def test_inactive_model_and_unrelated_changes_do_not_reset_stt(self):
        tasks = self.apply(nvidia_asr_model="custom/model", mode="rulebased_only")
        self.assertEqual(tasks.tasks, [])
        self.assertEqual(self.state["stt"], "hot")
        self.s["STT_PROVIDER"] = "parakeet"
        tasks = self.apply(whisper_model="small")
        self.assertEqual(tasks.tasks, [])

    async def test_sarvam_key_and_model_changes_queue_configuration_check(self):
        self.s["STT_PROVIDER"] = "sarvam"
        tasks = self.apply(sarvam_api_key="new-test-key", sarvam_model="new-model")
        self.assertEqual(self.state["stt"], "loading")
        await tasks.run()
        self.assertEqual(self.state["stt"], "configured")
        self.s["requests"].post.assert_not_called()

    async def test_superseded_queued_task_does_not_load_any_model(self):
        old = self.apply(stt_provider="parakeet")
        current = self.apply(stt_provider="nemotron")
        await old.run()
        self.s["get_nemotron_model"].assert_not_called()
        self.assertEqual(self.state["stt"], "loading")
        await current.run()
        self.s["get_nemotron_model"].assert_called_once_with("nvidia/nemotron-speech-streaming-en-0.6b")

    async def test_switch_away_and_back_rejects_old_revision(self):
        target, revision = self.request()
        self.apply(stt_provider="parakeet")
        self.apply(stt_provider="whisper")
        await self.s["_warm_stt_background"](target, revision)
        self.s["get_model"].assert_not_called()
        self.assertEqual(self.state["stt"], "loading")

    async def test_switch_during_whisper_warmup_serializes_and_discards_stale_results(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                self.s = isolated_server()
                self.state = self.s["MODEL_STATE"]
                started, release = threading.Event(), threading.Event()
                model = self.s["get_model"].return_value

                def blocking_transcribe(*args, **kwargs):
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError("test release timed out")
                    if fail:
                        raise RuntimeError("superseded failure")
                    return iter(()), None

                model.transcribe.side_effect = blocking_transcribe
                old = asyncio.create_task(self.s["_warm_stt_background"](*self.request()))
                new = None
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 3))
                    tasks = self.apply(stt_provider="parakeet")
                    new = asyncio.create_task(tasks.run())
                    await asyncio.sleep(0.03)
                    self.s["get_nemotron_model"].assert_not_called()
                    self.assertEqual(self.state["stt"], "loading")
                    release.set()
                    await old
                    await new
                    self.assertEqual(self.state["stt"], "hot")
                    self.assertEqual(self.state["device"], "cuda:0")
                    self.assertEqual(self.s["_device_used"], "cpu")
                finally:
                    release.set()
                    await old
                    if new is not None:
                        await new

    async def test_switch_during_download_skips_old_inference_and_clears_loader_device(self):
        started, release = threading.Event(), threading.Event()
        model = self.s["get_model"].return_value

        def load():
            started.set()
            if not release.wait(5):
                raise RuntimeError("test release timed out")
            self.state["device"] = "cuda"  # Existing get_model side effect.
            return model

        self.s["get_model"].side_effect = load
        task = asyncio.create_task(self.s["_warm_stt_background"](*self.request()))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            tasks = self.apply(stt_provider="parakeet")
            release.set()
            await task
            model.transcribe.assert_not_called()
            self.assertEqual(self.state["stt"], "loading")
            self.assertIsNone(self.state["device"])
            self.assertIsNone(self.state["last_warmed_at"])
            await tasks.run()
            self.assertEqual(self.state["stt"], "hot")
        finally:
            release.set()
            await task

    async def test_worker_warmup_waits_while_inference_lock_is_held(self):
        self.s["STT_PROVIDER"] = "parakeet"
        target, revision = self.request()
        started = threading.Event()

        def warm():
            started.set()
            self.s["_warm_selected_stt"](target, revision)

        task = None
        try:
            with self.s["_local_asr_lock"]:
                task = asyncio.create_task(asyncio.to_thread(warm))
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                await asyncio.sleep(0.05)
                self.assertFalse(task.done(), "Warmup bypassed the live inference lock")
                self.s["get_nemotron_model"].assert_not_called()
                self.s["get_model"].assert_not_called()
                self.assertEqual(self.state["stt"], "loading")
            await asyncio.wait_for(task, timeout=3)
            self.s["get_nemotron_model"].assert_called_once_with(self.s["NVIDIA_ASR_MODEL"])
            self.s["get_nemotron_model"].return_value.transcribe.assert_called_once()
            self.assertEqual(self.state["stt"], "hot")
        finally:
            if task is not None:
                await asyncio.wait_for(task, timeout=3)

    async def test_silence_file_failure_does_not_claim_nemo_inference_ready(self):
        self.s["STT_PROVIDER"] = "parakeet"
        self.s["tempfile"].mkstemp.side_effect = OSError("no writable temp directory")
        await self.s["warmup_all_models"]()
        self.assertIn("no writable temp directory", self.state["stt"])
        self.assertFalse(self.state["inference_ready"])
        self.s["get_nemotron_model"].return_value.transcribe.assert_not_called()
        self.s["get_model"].assert_not_called()

    async def test_ollama_warmup_is_not_blocked_by_stt_loading(self):
        entered, release, ollama_called = threading.Event(), threading.Event(), threading.Event()

        def load():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release timed out")
            return Mock(transcribe=Mock(return_value=(iter(()), None)))

        def post(*args, **kwargs):
            ollama_called.set()
            return SimpleNamespace(status_code=200)

        self.s["get_model"].side_effect = load
        self.s["requests"].post.side_effect = post
        task = asyncio.create_task(self.s["warmup_all_models"]())
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            self.assertTrue(await asyncio.to_thread(ollama_called.wait, 3))
        finally:
            release.set()
            await task


if __name__ == "__main__":
    unittest.main()
