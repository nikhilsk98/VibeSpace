"""Exception types with clear meanings, so the CLI can map them to exit codes."""


class VibeSpaceModelError(Exception):
    """Base class for all errors raised by the model pipeline."""


class ConfigError(VibeSpaceModelError):
    """The configuration file or an override is invalid."""


class DataValidationError(VibeSpaceModelError):
    """A pipeline output is missing, malformed or inconsistent."""


class DataVersionError(DataValidationError):
    """A pipeline output does not match the hash recorded in ``dvc.lock``."""


class LeakageError(VibeSpaceModelError):
    """A Tag Genome column (or anything derived from the labels) reached the feature matrix."""


class SplitError(VibeSpaceModelError):
    """The saved split is missing or does not match the current data."""


class TestSplitAccessError(VibeSpaceModelError):
    """Test rows were passed to a step that may only use train and validation data."""

    __test__ = False  # not a pytest test class
