"""Import optional model packages only when a real backend is used.

Deferred imports keep `pip install -e .` and CI on the fake pipeline. The
speech, language-model, and speech-synthesis packages are large and are not
required to exercise the demo.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from offline_voice.errors import BackendNotInstalledError


def optional_module(name: str, hint: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise BackendNotInstalledError(hint) from exc
