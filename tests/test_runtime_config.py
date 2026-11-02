"""Stdlib-only persistence regressions; all generated files stay in a temp dir.

Run: python -B -m unittest discover -s tests -p test_runtime_config.py -v
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import traceback
import unittest
from unittest.mock import patch

from runtime_config import RuntimeConfigError, RuntimeConfigStore


KEYS = (
    "DETECTION_MODE", "STT_PROVIDER", "WHISPER_MODEL", "SARVAM_MODEL",
    "SARVAM_MODE", "SARVAM_LANGUAGE", "SARVAM_API_KEY", "NVIDIA_API_KEY",
    "NVIDIA_ASR_ENDPOINT", "NVIDIA_ASR_MODEL", "OLLAMA_HOST", "OLLAMA_MODEL",
)
SECRET = "private-api-key-not-for-errors"


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "runtime_config.json"
        self.store = RuntimeConfigStore(self.path)

    def assert_no_temporary_files(self):
        expected = {self.path.name} if self.path.exists() else set()
        self.assertEqual({path.name for path in self.directory.iterdir()}, expected)

    def assert_sanitized(self, operation):
        try:
            operation()
        except RuntimeConfigError as error:
            rendered = "".join(traceback.format_exception(error))
            self.assertNotIn(SECRET, str(error))
            self.assertNotIn(SECRET, repr(error))
            self.assertNotIn(SECRET, rendered)
            self.assertNotIn(str(self.path), rendered)
        else:
            self.fail("Expected RuntimeConfigError")

    def test_missing_returns_independent_empty_dicts_without_creating_file(self):
        first = self.store.load()
        first["STT_PROVIDER"] = "whisper"
        self.assertEqual(self.store.load(), {})
        self.assert_no_temporary_files()

    def test_save_all_allowed_keys_and_reload_new_instance(self):
        values = {key: "value-" + key for key in KEYS}
        self.assertEqual(self.store.save(values), values)
        self.assertEqual(RuntimeConfigStore(str(self.path)).load(), values)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), values)
        self.assert_no_temporary_files()

    def test_reload_in_separate_process(self):
        values = {"STT_PROVIDER": "whisper", "SARVAM_API_KEY": SECRET}
        self.store.save(values)
        module_directory = str(Path(__file__).resolve().parents[1])
        script = (
            "import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from runtime_config import RuntimeConfigStore; "
            "assert RuntimeConfigStore(sys.argv[2]).load() == json.loads(sys.argv[3])"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script, module_directory,
             str(self.path), json.dumps(values)],
            cwd=self.directory, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(completed.returncode, 0, "Child reload failed")
        self.assert_no_temporary_files()

    def test_partial_updates_preserve_api_keys(self):
        self.store.save({"SARVAM_API_KEY": SECRET, "NVIDIA_API_KEY": "other-key"})
        result = self.store.save({"OLLAMA_MODEL": "phi3.5"})
        self.assertEqual(result, {
            "SARVAM_API_KEY": SECRET, "NVIDIA_API_KEY": "other-key",
            "OLLAMA_MODEL": "phi3.5",
        })
        self.assertEqual(self.store.save({}), result)

    def test_inputs_and_return_values_are_independent(self):
        updates = {"SARVAM_API_KEY": SECRET}
        saved = self.store.save(updates)
        updates["SARVAM_API_KEY"] = "changed-input"
        saved["SARVAM_API_KEY"] = "changed-result"
        loaded = self.store.load()
        loaded.clear()
        self.assertEqual(self.store.load(), {"SARVAM_API_KEY": SECRET})

    def test_empty_strings_unicode_and_newlines_round_trip(self):
        values = {"SARVAM_API_KEY": "", "OLLAMA_MODEL": "தமிழ்\nmodel"}
        self.assertEqual(self.store.save(values), values)
        self.assertEqual(self.store.load(), values)

    def test_store_does_not_modify_environment(self):
        with patch.dict(os.environ, {"STT_PROVIDER": "environment-provider"}):
            self.store.save({"STT_PROVIDER": "stored-provider"})
            self.assertEqual(self.store.load(), {"STT_PROVIDER": "stored-provider"})
            self.assertEqual(os.environ["STT_PROVIDER"], "environment-provider")

    def test_invalid_updates_leave_previous_file_unchanged(self):
        self.store.save({"SARVAM_API_KEY": SECRET})
        before = self.path.read_bytes()
        invalid = [None, [], "text", {"UNKNOWN": SECRET},
                   {"WHISPER_LANGUAGE": "en"}, {1: SECRET},
                   {"SARVAM_API_KEY": None}, {"SARVAM_API_KEY": 123},
                   {"SARVAM_API_KEY": True}, {"SARVAM_API_KEY": []},
                   {"SARVAM_API_KEY": {"secret": SECRET}},
                   {"STT_PROVIDER": "valid", "UNKNOWN": SECRET}]
        for updates in invalid:
            with self.subTest(updates_type=type(updates).__name__):
                self.assert_sanitized(lambda: self.store.save(updates))
                self.assertEqual(self.path.read_bytes(), before)
                self.assert_no_temporary_files()

    def test_nul_updates_rejected_without_changing_previous_file(self):
        self.store.save({"SARVAM_API_KEY": SECRET})
        before = self.path.read_bytes()
        for key in KEYS:
            for value in ("\x00" + SECRET, SECRET + "\x00suffix", SECRET + "\x00"):
                with self.subTest(key=key):
                    self.assert_sanitized(lambda: self.store.save({key: value}))
                    self.assertEqual(self.path.read_bytes(), before)
                    self.assertEqual(self.store.load(), {"SARVAM_API_KEY": SECRET})
                    self.assert_no_temporary_files()

    def test_nul_in_saved_json_rejected_without_overwriting_file(self):
        for key in KEYS:
            with self.subTest(key=key):
                contents = json.dumps({key: SECRET + "\x00suffix"}).encode("utf-8")
                self.path.write_bytes(contents)
                self.assert_sanitized(self.store.load)
                self.assertEqual(self.path.read_bytes(), contents)
                self.assert_sanitized(lambda: self.store.save({"OLLAMA_MODEL": "new"}))
                self.assertEqual(self.path.read_bytes(), contents)
                self.assert_no_temporary_files()

    def test_invalid_file_is_not_silently_overwritten(self):
        invalid = [b'{"SARVAM_API_KEY": "' + SECRET.encode(), b'\xff',
                   b'[]', b'null', b'"text"', b'42',
                   json.dumps({"UNKNOWN": SECRET}).encode(),
                   json.dumps({"SARVAM_API_KEY": 42}).encode(),
                   json.dumps({"STT_PROVIDER": True}).encode(),
                   json.dumps({"OLLAMA_HOST": {"secret": SECRET}}).encode()]
        for contents in invalid:
            with self.subTest(contents_type=type(contents).__name__):
                self.path.write_bytes(contents)
                self.assert_sanitized(self.store.load)
                self.assert_sanitized(lambda: self.store.save({"OLLAMA_MODEL": "new"}))
                self.assertEqual(self.path.read_bytes(), contents)
                self.assert_no_temporary_files()

    def test_read_failure_is_sanitized_and_prevents_writes(self):
        self.store.save({"SARVAM_API_KEY": SECRET})
        before = self.path.read_bytes()
        with patch.object(Path, "open", side_effect=PermissionError(SECRET)):
            self.assert_sanitized(self.store.load)
            self.assert_sanitized(lambda: self.store.save({"OLLAMA_MODEL": "new"}))
        self.assertEqual(self.path.read_bytes(), before)
        self.assert_no_temporary_files()

    def test_atomic_save_failures_preserve_previous_file_and_cleanup(self):
        self.store.save({"SARVAM_API_KEY": SECRET})
        before = self.path.read_bytes()
        for target in ("tempfile.mkstemp", "os.fdopen", "json.dump", "os.fsync", "os.replace"):
            with self.subTest(target=target):
                with patch("runtime_config." + target, side_effect=OSError(SECRET)):
                    self.assert_sanitized(lambda: self.store.save({"OLLAMA_MODEL": "new"}))
                self.assertEqual(self.path.read_bytes(), before)
                self.assert_no_temporary_files()

    def test_failed_first_save_leaves_no_file(self):
        with patch("runtime_config.os.replace", side_effect=OSError(SECRET)):
            self.assert_sanitized(lambda: self.store.save({"SARVAM_API_KEY": SECRET}))
        self.assertFalse(self.path.exists())
        self.assert_no_temporary_files()

    def test_flush_failure_preserves_file_and_cleans_up(self):
        self.store.save({"SARVAM_API_KEY": SECRET})
        before = self.path.read_bytes()
        real_fdopen = os.fdopen

        def failing_stream(*args, **kwargs):
            stream = real_fdopen(*args, **kwargs)
            stream.flush = lambda: (_ for _ in ()).throw(OSError(SECRET))
            return stream

        with patch("runtime_config.os.fdopen", side_effect=failing_stream):
            self.assert_sanitized(lambda: self.store.save({"OLLAMA_MODEL": "new"}))
        self.assertEqual(self.path.read_bytes(), before)
        self.assert_no_temporary_files()

    def test_tempfile_is_in_destination_directory_and_fsynced_before_replace(self):
        real_mkstemp = tempfile.mkstemp
        real_fsync = os.fsync
        real_replace = os.replace
        events = []

        def create(*args, **kwargs):
            self.assertEqual(Path(kwargs["dir"]), self.directory)
            return real_mkstemp(*args, **kwargs)

        def fsync(descriptor):
            events.append("fsync")
            return real_fsync(descriptor)

        def replace(source, destination):
            events.append("replace")
            self.assertEqual(Path(source).parent, self.directory)
            self.assertEqual(Path(destination), self.path)
            self.assertEqual(json.loads(Path(source).read_text(encoding="utf-8")),
                             {"SARVAM_API_KEY": SECRET})
            return real_replace(source, destination)

        with patch("runtime_config.tempfile.mkstemp", side_effect=create), \
                patch("runtime_config.os.fsync", side_effect=fsync), \
                patch("runtime_config.os.replace", side_effect=replace):
            self.store.save({"SARVAM_API_KEY": SECRET})
        self.assertEqual(events, ["fsync", "replace"])
        self.assert_no_temporary_files()

    @unittest.skipUnless(os.name == "posix", "POSIX permission semantics required")
    def test_owner_only_permissions_on_create_and_replacement(self):
        self.store.save({"SARVAM_API_KEY": SECRET})
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.path.chmod(0o644)
        self.store.save({"OLLAMA_MODEL": "new"})
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_concurrent_updates_do_not_lose_keys(self):
        barrier = threading.Barrier(len(KEYS))

        def update(key):
            barrier.wait(timeout=10)
            return self.store.save({key: "value-" + key})

        with ThreadPoolExecutor(max_workers=len(KEYS)) as executor:
            results = list(executor.map(update, KEYS))
        expected = {key: "value-" + key for key in KEYS}
        self.assertEqual(self.store.load(), expected)
        self.assertEqual(max(map(len, results)), len(KEYS))
        self.assert_no_temporary_files()


if __name__ == "__main__":
    unittest.main()
