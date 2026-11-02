"""Isolated dashboard persistence integration; no server/model imports or network.

Run: python -B -m unittest discover -s tests -p test_dashboard_config.py -v
"""

import ast
import builtins
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from runtime_config import RuntimeConfigError, RuntimeConfigStore
from test_model_readiness import (
    BackgroundTasks, HTTPException, SERVER_PATH, config, isolated_server,
)


# Runtime globals are initialized from these persisted environment names.
GLOBAL_ENV = {
    "DETECTION_MODE": "DETECTION_MODE", "STT_PROVIDER": "STT_PROVIDER",
    "WHISPER_MODEL_SIZE": "WHISPER_MODEL", "SARVAM_MODEL": "SARVAM_MODEL",
    "SARVAM_MODE": "SARVAM_MODE", "SARVAM_LANGUAGE": "SARVAM_LANGUAGE",
    "NVIDIA_ASR_ENDPOINT": "NVIDIA_ASR_ENDPOINT", "NVIDIA_ASR_MODEL": "NVIDIA_ASR_MODEL",
    "OLLAMA_HOST": "OLLAMA_HOST", "OLLAMA_MODEL": "OLLAMA_MODEL",
}
SARVAM_KEY = "sarvam-private-dashboard-key-9876"
NVIDIA_KEY = "nvidia-private-dashboard-key-5432"
FULL_REQUEST = {
    "mode": " LLM_ONLY ", "stt_provider": " SARVAM ", "whisper_model": " SMALL ",
    "sarvam_model": " saaras:v3 ", "sarvam_mode": " translate ",
    "sarvam_language": " hi-IN ", "sarvam_api_key": " " + SARVAM_KEY + " ",
    "nvidia_api_key": " " + NVIDIA_KEY + " ",
    "nvidia_asr_endpoint": " http://asr.invalid:9000 ",
    "nvidia_asr_model": " custom/asr-model ",
    "ollama_host": " http://ollama.invalid:11434 ", "ollama_model": " llama3.2 ",
}
FULL_SAVED = {
    "DETECTION_MODE": "llm_only", "STT_PROVIDER": "sarvam", "WHISPER_MODEL": "small",
    "SARVAM_MODEL": "saaras:v3", "SARVAM_MODE": "translate", "SARVAM_LANGUAGE": "hi-IN",
    "SARVAM_API_KEY": SARVAM_KEY, "NVIDIA_API_KEY": NVIDIA_KEY,
    "NVIDIA_ASR_ENDPOINT": "http://asr.invalid:9000", "NVIDIA_ASR_MODEL": "custom/asr-model",
    "OLLAMA_HOST": "http://ollama.invalid:11434", "OLLAMA_MODEL": "llama3.2",
}


def startup_namespace(path, deployment=None):
    """Execute the actual startup loader/call and settings assignments in order."""
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), str(SERVER_PATH))
    nodes = []
    assigned = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_load_dashboard_config":
            nodes.append(node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id == "_load_dashboard_config"):
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & GLOBAL_ENV.keys():
                assigned.update(names)
                nodes.append(node)
    if not set(GLOBAL_ENV).issubset(assigned) or len(nodes) != len(GLOBAL_ENV) + 2:
        raise AssertionError("Missing startup loader, invocation, or runtime assignments")
    environment = dict(deployment or {})
    scope = {
        "os": SimpleNamespace(environ=environment, getenv=environment.get),
        "RUNTIME_CONFIG_STORE": RuntimeConfigStore(path),
        "RuntimeConfigError": RuntimeConfigError, "logger": Mock(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SERVER_PATH), "exec"), scope)
    return scope


class DashboardConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "runtime_config.json"
        self.store = RuntimeConfigStore(self.path)
        self.s = isolated_server()
        self.s["RUNTIME_CONFIG_STORE"] = self.store
        self.s["BASE_DIR"] = str(self.directory)
        self.s["os"].path = os.path
        self.tasks = BackgroundTasks()

    def apply(self, **values):
        return self.s["update_config"](config(**values), self.tasks)

    def snapshot(self):
        return {
            "globals": {name: self.s[name] for name in GLOBAL_ENV},
            "environment": dict(self.s["os"].environ),
            "state": deepcopy(self.s["MODEL_STATE"]),
            "revision": self.s["_stt_revision"],
            "whisper_model": self.s["_whisper_model"],
            "device": self.s["_device_used"],
            "tasks": list(self.tasks.tasks),
        }

    def assert_no_secrets(self, *values):
        rendered = repr(values) + repr(self.s["logger"].mock_calls)
        for secret in (SARVAM_KEY, NVIDIA_KEY):
            self.assertNotIn(secret, rendered)

    def assert_no_model_or_network_calls(self):
        for name in ("get_model", "get_nemotron_model", "reload_whisper_model"):
            self.s[name].assert_not_called()
        self.s["requests"].post.assert_not_called()

    def test_full_configuration_persists_and_overrides_defaults_on_restart(self):
        response = self.apply(**FULL_REQUEST)
        self.assertTrue(response["success"])
        self.assertTrue(response["persisted"])
        self.assertEqual(RuntimeConfigStore(self.path).load(), FULL_SAVED)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), FULL_SAVED)
        self.assertEqual(self.s["os"].environ, FULL_SAVED)
        for name, env_name in GLOBAL_ENV.items():
            self.assertEqual(self.s[name], FULL_SAVED[env_name], name)

        defaults = startup_namespace(self.directory / "missing.json")
        self.assertEqual(defaults["DETECTION_MODE"], "all")
        self.assertEqual(defaults["STT_PROVIDER"], "whisper")
        self.assertEqual(defaults["OLLAMA_HOST"], "localhost")
        self.assertEqual(defaults["OLLAMA_MODEL"], "phi3.5")
        deployment = {key: "deployment-default" for key in FULL_SAVED}
        deployment["DATA_DIR"] = str(self.directory)
        restarted = startup_namespace(self.path, deployment)
        for name, env_name in GLOBAL_ENV.items():
            self.assertEqual(restarted[name], FULL_SAVED[env_name], name)
        self.assertEqual(restarted["os"].environ,
                         {**FULL_SAVED, "DATA_DIR": str(self.directory)})
        self.assertIsNot(restarted["RUNTIME_CONFIG_STORE"], self.store)
        self.assertEqual(restarted["logger"].warning.call_count, 0)
        self.assert_no_secrets(response, restarted["logger"].mock_calls)
        self.assert_no_model_or_network_calls()

    def test_startup_load_occurs_after_dotenv_and_before_settings(self):
        tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
        dotenv_calls = [node.lineno for node in ast.walk(tree)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "load_dotenv"]
        loader_calls = [node.lineno for node in tree.body
                        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "_load_dashboard_config"]
        self.assertTrue(dotenv_calls)
        self.assertEqual(len(loader_calls), 1)
        self.assertLess(max(dotenv_calls), loader_calls[0])
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in GLOBAL_ENV
                for target in node.targets
            ):
                self.assertLess(loader_calls[0], node.lineno)

    def test_partial_update_preserves_keys_and_omitted_settings(self):
        self.apply(**FULL_REQUEST)
        response = self.apply(ollama_model=" qwen2.5 ")
        expected = {**FULL_SAVED, "OLLAMA_MODEL": "qwen2.5"}
        self.assertTrue(response["persisted"])
        self.assertEqual(RuntimeConfigStore(self.path).load(), expected)
        self.assertEqual(self.s["os"].environ, expected)
        restarted = startup_namespace(self.path)
        for name, env_name in GLOBAL_ENV.items():
            self.assertEqual(restarted[name], expected[env_name], name)
        self.assertEqual(restarted["os"].environ["SARVAM_API_KEY"], SARVAM_KEY)
        self.assertEqual(restarted["os"].environ["NVIDIA_API_KEY"], NVIDIA_KEY)
        self.assert_no_secrets(response)
        self.assert_no_model_or_network_calls()

    def test_blank_keys_retain_secrets_without_exposing_them(self):
        self.apply(**FULL_REQUEST)
        before = self.snapshot()
        response = self.apply(sarvam_api_key="   ", nvidia_api_key="")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(RuntimeConfigStore(self.path).load(), FULL_SAVED)
        self.assertTrue(response["sarvam_configured"])
        self.assertEqual(response["sarvam_key_hint"], "sk_..." + SARVAM_KEY[-4:])
        self.assert_no_secrets(response)
        self.assert_no_model_or_network_calls()

    def test_write_failure_returns_503_without_any_runtime_mutation(self):
        self.apply(**FULL_REQUEST)
        previous_file = self.path.read_bytes()
        before = self.snapshot()
        for target in ("os.fsync", "os.replace"):
            with self.subTest(target=target):
                with patch("runtime_config." + target, side_effect=OSError(SARVAM_KEY)):
                    with self.assertRaises(HTTPException) as caught:
                        self.apply(mode="rulebased_only", stt_provider="nemotron",
                                   ollama_host="changed", sarvam_api_key=NVIDIA_KEY)
                self.assertEqual(caught.exception.status_code, 503)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.path.read_bytes(), previous_file)
                self.assertEqual(list(self.directory.iterdir()), [self.path])
                self.assert_no_secrets(caught.exception.detail, str(caught.exception))
        self.assert_no_model_or_network_calls()

    def test_first_write_failure_creates_no_file_or_runtime_state(self):
        before = self.snapshot()
        with patch("runtime_config.os.replace", side_effect=OSError(NVIDIA_KEY)):
            with self.assertRaises(HTTPException) as caught:
                self.apply(**FULL_REQUEST)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assert_no_secrets(caught.exception.detail)

    def test_invalid_config_returns_422_without_file_or_runtime_mutations(self):
        invalid = [
            {"mode": "unsupported"}, {"stt_provider": "unsupported"},
            {"mode": ""}, {"ollama_host": "  "}, {"whisper_model": 42},
            {"sarvam_api_key": SARVAM_KEY + "\n"},
            {"nvidia_api_key": NVIDIA_KEY + "\r"},
            {"ollama_model": "model\ninvalid"},
            {"sarvam_mode": ""}, {"sarvam_language": []},
            {"mode": "llm_only", "stt_provider": "invalid"},
        ]
        invalid.extend({field: SARVAM_KEY + "\x00suffix"} for field in FULL_REQUEST)
        for existing in (False, True):
            if existing:
                self.apply(**FULL_REQUEST)
            for values in invalid:
                with self.subTest(existing=existing, fields=tuple(values)):
                    before = self.snapshot()
                    previous = self.path.read_bytes() if self.path.exists() else None
                    with self.assertRaises(HTTPException) as caught:
                        self.apply(**values)
                    self.assertEqual(caught.exception.status_code, 422)
                    self.assertEqual(self.snapshot(), before)
                    self.assertEqual(self.path.read_bytes() if self.path.exists() else None,
                                     previous)
                    self.assertEqual(set(self.directory.iterdir()),
                                     {self.path} if existing else set())
                    self.assert_no_secrets(caught.exception.detail)
        self.assert_no_model_or_network_calls()

    def test_provider_derived_nemo_defaults_replace_old_model_and_allow_override(self):
        for provider, default_model in (
            ("parakeet", "nvidia/parakeet-ctc-0.6b"),
            ("nemotron", "nvidia/nemotron-speech-streaming-en-0.6b"),
        ):
            with self.subTest(provider=provider):
                self.apply(nvidia_asr_model="previous/custom-model")
                response = self.apply(stt_provider=provider)
                self.assertTrue(response["persisted"])
                self.assertEqual(response["nvidia_asr_model"], default_model)
                self.assertEqual(self.s["NVIDIA_ASR_MODEL"], default_model)
                self.assertEqual(RuntimeConfigStore(self.path).load()["NVIDIA_ASR_MODEL"],
                                 default_model)
                self.apply(stt_provider=provider, nvidia_asr_model="explicit/custom-model")
                restarted = startup_namespace(self.path)
                self.assertEqual(restarted["STT_PROVIDER"], provider)
                self.assertEqual(restarted["NVIDIA_ASR_MODEL"], "explicit/custom-model")
        self.assert_no_model_or_network_calls()

    def test_normalizer_is_pure_and_does_not_persist(self):
        before = self.snapshot()
        request = config(**FULL_REQUEST)
        request_before = vars(request).copy()
        self.assertEqual(self.s["_dashboard_config_updates"](request), FULL_SAVED)
        self.assertEqual(vars(request), request_before)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.path.exists())

    def test_corrupt_saved_config_falls_back_safely_and_api_refuses_overwrite(self):
        corrupt = '{"SARVAM_API_KEY": "' + SARVAM_KEY
        self.path.write_text(corrupt, encoding="utf-8")
        deployment = {"STT_PROVIDER": "whisper", "OLLAMA_MODEL": "phi3.5"}
        restarted = startup_namespace(self.path, deployment)
        self.assertEqual(restarted["os"].environ, deployment)
        self.assertEqual(restarted["STT_PROVIDER"], "whisper")
        self.assertEqual(restarted["OLLAMA_MODEL"], "phi3.5")
        restarted["logger"].warning.assert_called_once()
        before = self.snapshot()
        with self.assertRaises(HTTPException) as caught:
            self.apply(**FULL_REQUEST)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.path.read_text(encoding="utf-8"), corrupt)
        self.assert_no_secrets(caught.exception.detail, restarted["logger"].mock_calls)

    def test_api_never_attempts_to_write_project_dotenv(self):
        env_file = self.directory / ".env"
        original = b"STT_PROVIDER=whisper\nOLLAMA_MODEL=phi3.5\n"
        env_file.write_bytes(original)
        env_file.chmod(0o444)
        self.addCleanup(env_file.chmod, 0o600)
        real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

        def check_write(file, writing):
            if not isinstance(file, int) and Path(file).resolve() == env_file.resolve():
                self.assertFalse(writing, "Dashboard attempted to write .env")

        def guarded_open(file, mode="r", *args, **kwargs):
            check_write(file, any(flag in mode for flag in "wax+"))
            return real_open(file, mode, *args, **kwargs)

        def guarded_io_open(file, mode="r", *args, **kwargs):
            check_write(file, any(flag in mode for flag in "wax+"))
            return real_io_open(file, mode, *args, **kwargs)

        def guarded_os_open(file, flags, *args, **kwargs):
            check_write(file, flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            return real_os_open(file, flags, *args, **kwargs)

        with patch("builtins.open", side_effect=guarded_open), \
                patch("io.open", side_effect=guarded_io_open), \
                patch("os.open", side_effect=guarded_os_open):
            self.assertTrue(self.apply(**FULL_REQUEST)["persisted"])
            self.assertTrue(self.apply(ollama_model="another-model")["persisted"])
            restarted = startup_namespace(self.path)
        self.assertEqual(env_file.read_bytes(), original)
        self.assertEqual(restarted["OLLAMA_MODEL"], "another-model")
        self.assertEqual(RuntimeConfigStore(self.path).load()["SARVAM_API_KEY"], SARVAM_KEY)
        self.s["os"].fdopen.assert_not_called()
        self.s["os"].unlink.assert_not_called()
        self.assert_no_model_or_network_calls()


if __name__ == "__main__":
    unittest.main()
