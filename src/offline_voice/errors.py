"""Errors raised by the voice pipeline."""


class PipelineError(Exception):
    """The pipeline could not finish a turn."""


class BackendNotInstalledError(PipelineError):
    """An optional model package or binary is not installed."""


class MissingModelError(PipelineError):
    """A configured weight file is not on disk."""
