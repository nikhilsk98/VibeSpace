"""Exception hierarchy for the VibeSpace data pipeline.

Each failure point raises a specific subclass of :class:`PipelineError`, so
task logs and alerts say exactly what went wrong and where.
"""


class PipelineError(Exception):
    """Base class for every expected pipeline failure."""


class ConfigError(PipelineError):
    """Invalid or missing configuration."""


class DataAcquisitionError(PipelineError):
    """A dataset could not be downloaded or fetched."""


class ChecksumMismatchError(DataAcquisitionError):
    """A downloaded file does not match its published checksum."""


class CorruptArchiveError(DataAcquisitionError):
    """An archive is unreadable or contains unsafe entries."""


class TMDBAuthError(DataAcquisitionError):
    """TMDB rejected the credentials (HTTP 401). Retrying cannot help."""


class FailureRateExceededError(DataAcquisitionError):
    """Too many TMDB requests failed for the run to be trusted."""


class SchemaError(PipelineError):
    """A file or table does not have the expected columns or types."""


class MatchRateError(PipelineError):
    """Too few MovieLens movies could be matched to TMDB."""


class MappingError(PipelineError):
    """The Movie DNA tag mapping is invalid (e.g. unknown genome tags)."""


class ValidationFailedError(PipelineError):
    """A Great Expectations suite failed; downstream steps must not run."""


class AnomalyError(PipelineError):
    """Blocking (critical) data anomalies were detected."""


class DVCError(PipelineError):
    """A DVC command failed."""
