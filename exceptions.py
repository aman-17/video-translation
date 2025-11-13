"""
Custom exceptions.
"""


class PipelineError(Exception):
    """
    Base exception for all pipeline-related errors.

    All custom exceptions in the pipeline inherit from this class,
    making it easy to catch any pipeline-specific error.
    """
    pass


class VideoFileNotFoundError(PipelineError):
    """
    Raised when a required file is not found.

    This exception is raised when input files (video, transcript, etc.)
    cannot be located at the specified path.
    """
    pass


class TranslationError(PipelineError):
    """
    Raised when translation fails.

    This exception is raised when the translation service fails to
    translate text from source to target language.
    """
    pass


class TTSError(PipelineError):
    """
    Raised when Text-to-Speech generation fails.

    This exception is raised when the TTS model fails to generate
    audio from text, typically due to model loading issues or
    generation errors.
    """
    pass


class LatentSyncError(PipelineError):
    """
    Raised when LatentSync lip-sync processing fails.

    This exception is raised when the LatentSync model fails to
    process the video, typically due to missing checkpoints,
    CUDA issues, or processing errors.
    """
    pass
