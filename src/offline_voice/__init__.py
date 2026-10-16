"""Offline on-device voice assistant pipeline.

This package is a demo. Fake speech, language, and speech-synthesis backends
ship with it so the control flow runs without model weights.
"""

from offline_voice.pipeline import VoicePipeline

__version__ = "0.1.0"
__all__ = ["VoicePipeline", "__version__"]
