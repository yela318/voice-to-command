"""Typed errors so callers can tell failure modes apart."""


class V2CError(Exception):
    """Base class for everything voice-to-command raises on purpose."""


class BackendNotAvailable(V2CError):
    """Requested backend is unknown, or its optional dependency isn't installed."""


class CredentialsMissing(V2CError):
    """A backend needs credentials that aren't in the environment."""


class LanguageRequired(V2CError):
    """The backend needs an explicit language and 'auto' was given."""


class AudioTooLong(V2CError):
    """Audio exceeds a backend's hard limit."""


class DecodeError(V2CError):
    """Could not decode the audio input to PCM samples."""
