"""Atomic persistence for dashboard runtime settings, with no environment side effects.

The caller supplies DATA_DIR / "runtime_config.json" and controls environment
precedence. The parent directory must already exist. Share one store instance
between threads; separate instances/processes are not coordinated for writes.
"""

import json
import os
from pathlib import Path
import tempfile
import threading


__all__ = ["RuntimeConfigError", "RuntimeConfigStore"]

_ALLOWED_KEYS = frozenset({
    "DETECTION_MODE", "STT_PROVIDER", "WHISPER_MODEL", "SARVAM_MODEL",
    "SARVAM_MODE", "SARVAM_LANGUAGE", "SARVAM_API_KEY", "NVIDIA_API_KEY",
    "NVIDIA_ASR_ENDPOINT", "NVIDIA_ASR_MODEL", "OLLAMA_HOST", "OLLAMA_MODEL",
    "AUDIO_SYSTEM_PROMPT", "SMS_SYSTEM_PROMPT",
})


class RuntimeConfigError(Exception):
    """A persistence or validation failure with a sanitized public message."""


def _validated_copy(values: object) -> dict[str, str]:
    if not isinstance(values, dict):
        raise ValueError("Invalid runtime configuration")
    result = dict(values)
    if any(not isinstance(key, str) or key not in _ALLOWED_KEYS
           or not isinstance(value, str) or "\x00" in value
           for key, value in result.items()):
        raise ValueError("Invalid runtime configuration")
    return result


class RuntimeConfigStore:
    """Read and merge a flat JSON mapping of allowed environment names to strings."""

    def __init__(self, path: str | os.PathLike[str]):
        self._path = Path(path)
        self._lock = threading.RLock()

    def load(self) -> dict[str, str]:
        """Return a fresh mapping, or {} if missing; never ignore corrupt data."""
        with self._lock:
            try:
                try:
                    stream = self._path.open("r", encoding="utf-8")
                except FileNotFoundError:
                    return {}
                with stream:
                    return _validated_copy(json.load(stream))
            except Exception:
                raise RuntimeConfigError("Unable to load runtime configuration") from None

    def save(self, updates: dict[str, str]) -> dict[str, str]:
        """Validate, read, merge, and atomically replace; return a fresh mapping.

        New files have owner-only permissions on POSIX, including replacements
        of an existing file. No file is replaced until flushing and fsync succeed.
        """
        with self._lock:
            temporary_path = None
            descriptor = None
            try:
                validated = _validated_copy(updates)
                merged = self.load()
                merged.update(validated)
                descriptor, temporary_path = tempfile.mkstemp(
                    prefix=".runtime_config-", suffix=".tmp", dir=self._path.parent,
                )
                stream = os.fdopen(descriptor, "w", encoding="utf-8")
                descriptor = None  # The stream now owns the descriptor.
                with stream:
                    json.dump(merged, stream, ensure_ascii=True, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self._path)
                return dict(merged)
            except Exception:
                raise RuntimeConfigError("Unable to save runtime configuration") from None
            finally:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                if temporary_path is not None:
                    try:
                        os.unlink(temporary_path)
                    except OSError:
                        pass
