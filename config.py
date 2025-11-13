"""
Configuration module.
"""


class Config:
    """
    Global configuration constants for the pipeline.

    This class centralizes all configuration values including language settings,
    model parameters, file paths, and default values for processing.
    """

    SOURCE_LANG = "en"
    TARGET_LANG = "de"

    TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    TTS_TEMPERATURE = 0.75  # Lower=more consistent voice (0.5-0.7), Higher=more varied (0.8-1.0)
    TTS_SPEED = 1.0  # Speech speed multiplier
    REFERENCE_AUDIO_DURATION = 20.0  # Duration in seconds for voice cloning reference

    LATENTSYNC_CHECKPOINT = "latentsync/checkpoints/latentsync_unet.pt"
    LATENTSYNC_CONFIG = "latentsync/configs/unet/stage2_512.yaml"
    DEFAULT_INFERENCE_STEPS = 25  # No. of diffusion steps (higher=better quality)
    DEFAULT_GUIDANCE_SCALE = 2.0  # Controls adherence to audio
    DEFAULT_SEED = 1247  # Random seed

    TEMP_DIR_NAME = "temp_audio_segments"
    OUTPUT_SUFFIX_AUDIO = "_audio.wav"
    OUTPUT_SUFFIX_SRT = "_translated.srt"
