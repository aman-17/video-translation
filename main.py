"""
Video Translation Pipeline: English to German with Voice Cloning and Lip-Sync

This module provides a complete pipeline for translating videos from English to German,
preserving the original speaker's voice characteristics and synchronizing lip movements
using LatentSync diffusion model.

Author: Aman Rangapur
"""

import argparse
import logging
import os
import shutil
import sys
from typing import List, Tuple

import torch
from moviepy.editor import VideoFileClip
from TTS.api import TTS

from config import Config
from exceptions import (
    VideoFileNotFoundError,
    TTSError,
    LatentSyncError
)
from subtitles import SubtitleSegment, parse_srt_file, save_translated_srt
from translation import translate_segments


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging for the application.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger('VideoTranslationPipeline')
    return logger


def extract_reference_audio(
    video_path: str,
    output_path: str,
    duration: float,
    logger: logging.Logger
) -> None:
    """
    Extract reference audio clip from video for voice cloning.

    Args:
        video_path: Path to input video file
        output_path: Path where audio will be saved
        duration: Duration of audio to extract (seconds)
        logger: Logger instance
    """
    logger.info(f"Extracting reference audio (first {duration}s) for voice cloning")

    try:
        video = VideoFileClip(video_path)
        actual_duration = min(duration, video.duration)

        audio = video.audio.subclip(0, actual_duration)
        audio.write_audiofile(output_path, logger=None)

        video.close()
        audio.close()

        logger.info(f"Reference audio saved: {output_path} ({actual_duration:.1f}s)")

    except Exception as e:
        raise IOError(f"Failed to extract reference audio: {e}")


def generate_segment_audio(
    tts_model: TTS,
    text: str,
    reference_audio: str,
    output_path: str,
    language: str,
    temperature: float,
    speed: float,
    logger: logging.Logger
) -> bool:
    """
    Generate audio for a text segment using TTS with voice cloning.

    Args:
        tts_model: Initialized TTS model instance
        text: Text to synthesize
        reference_audio: Path to reference audio for voice cloning
        output_path: Path where generated audio will be saved
        language: Target language code
        temperature: Controls voice consistency
        speed: Speech speed multiplier
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        tts_model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=reference_audio,
            language=language,
            temperature=temperature,
            speed=speed,
            enable_text_splitting=True
        )
        return True
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return False


def create_synthesized_audio(
    segments: List[SubtitleSegment],
    reference_audio: str,
    output_dir: str,
    target_lang: str,
    logger: logging.Logger
) -> str:
    """
    Create complete synthesized audio track with voice cloning.

    Args:
        segments: List of subtitle segments to synthesize
        reference_audio: Path to reference audio for voice cloning
        output_dir: Directory where output audio will be saved
        target_lang: Target language code
        logger: Logger instance

    Returns:
        Path to generated audio file
    """
    logger.info("Generating synthesized audio with voice cloning")

    full_text = " ".join([segment.text for segment in segments])
    logger.info(f"Combined text length: {len(full_text)} characters")
    logger.debug(f"Text preview: {full_text[:200]}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cpu":
        logger.warning("CUDA not available. TTS will be slow on CPU.")

    try:
        tts = TTS(Config.TTS_MODEL_NAME, gpu=(device == "cuda"))
    except Exception as e:
        raise TTSError(f"Failed to initialize TTS model: {e}")

    final_audio_path = os.path.join(output_dir, "translated_audio_full.wav")
    logger.info(f"Generating audio with temperature={Config.TTS_TEMPERATURE}")

    success = generate_segment_audio(
        tts_model=tts,
        text=full_text,
        reference_audio=reference_audio,
        output_path=final_audio_path,
        language=target_lang,
        temperature=Config.TTS_TEMPERATURE,
        speed=Config.TTS_SPEED,
        logger=logger
    )

    if not success:
        raise TTSError("Failed to generate audio")

    logger.info("Audio generated successfully")
    return final_audio_path


def run_latentsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    logger: logging.Logger
) -> bool:
    """
    Run LatentSync for lip-sync.

    Args:
        video_path: Path to input video
        audio_path: Path to audio track
        output_path: Path for output video
        inference_steps: No. of diffusion inference steps
        guidance_scale: Guidance scale for diffusion model
        seed: Random seed for reproducibility
        logger: Logger instance
    """
    logger.info("Running LatentSync for lip-sync")

    # Add latentsync directory to Python path
    latentsync_dir = os.path.join(os.path.dirname(__file__), 'latentsync')
    if latentsync_dir not in sys.path:
        sys.path.insert(0, latentsync_dir)

    try:
        from scripts.inference import main as latentsync_main
        from omegaconf import OmegaConf
    except ImportError as e:
        raise LatentSyncError(f"Failed to import LatentSync modules: {e}")

    checkpoint_path = Config.LATENTSYNC_CHECKPOINT
    config_path = Config.LATENTSYNC_CONFIG

    if not os.path.exists(checkpoint_path):
        raise LatentSyncError(
            f"LatentSync checkpoint not found at {checkpoint_path}"
        )

    if not os.path.exists(config_path):
        raise LatentSyncError(f"LatentSync config not found at {config_path}")

    try:
        config = OmegaConf.load(config_path)
        config["run"] = config.get("run", {})
        config["run"].update({
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        })
    except Exception as e:
        raise LatentSyncError(f"Failed to load LatentSync config: {e}")

    class Args:
        """
        Arguments for LatentSync.
        """
        pass

    args = Args()
    args.inference_ckpt_path = os.path.abspath(checkpoint_path)
    args.video_path = os.path.abspath(video_path)
    args.audio_path = os.path.abspath(audio_path)
    args.video_out_path = os.path.abspath(output_path)
    args.inference_steps = inference_steps
    args.guidance_scale = guidance_scale
    args.temp_dir = "temp"
    args.seed = seed
    args.enable_deepcache = True

    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Inference steps: {inference_steps}")
    logger.info(f"Guidance scale: {guidance_scale}")
    logger.info(f"Seed: {seed}")

    original_dir = os.getcwd()
    latentsync_dir = os.path.join(os.path.dirname(__file__), 'latentsync')

    try:
        os.chdir(latentsync_dir)
        latentsync_main(config=config, args=args)
        logger.info(f"Lip-synced video saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"LatentSync processing failed: {e}", exc_info=True)
        return False

    finally:
        os.chdir(original_dir)


class VideoTranslationPipeline:
    """
    Class for video translation with voice cloning and lip-sync.
    """

    def __init__(
        self,
        video_path: str,
        transcript_path: str,
        output_path: str,
        inference_steps: int = Config.DEFAULT_INFERENCE_STEPS,
        guidance_scale: float = Config.DEFAULT_GUIDANCE_SCALE,
        seed: int = Config.DEFAULT_SEED,
        keep_temp: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the video translation pipeline.

        Args:
            video_path: Path to input video file
            transcript_path: Path to input SRT transcript file
            output_path: Path for output video file
            inference_steps: No. of inference steps for LatentSync
            guidance_scale: Guidance scale for LatentSync
            seed: Random seed for reproducibility
            keep_temp: If True, keep temporary files after processing
            verbose: If True, enable debug logging
        """
        self.video_path = video_path
        self.transcript_path = transcript_path
        self.output_path = output_path
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.keep_temp = keep_temp

        self.logger = setup_logging(verbose)
        self._validate_inputs()
        self._setup_directories()

    def _validate_inputs(self) -> None:
        """
        Validate that input files exist.
        """
        if not os.path.exists(self.video_path):
            raise VideoFileNotFoundError(f"Video file not found: {self.video_path}")

        if not os.path.exists(self.transcript_path):
            raise VideoFileNotFoundError(f"Transcript file not found: {self.transcript_path}")

    def _setup_directories(self) -> None:
        """
        Create necessary output dir's.
        """
        self.output_dir = os.path.dirname(self.output_path) or "."
        self.temp_dir = os.path.join(self.output_dir, Config.TEMP_DIR_NAME)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.logger.debug(f"Output directory: {self.output_dir}")
        self.logger.debug(f"Temporary directory: {self.temp_dir}")

    def run(self) -> Tuple[str, str, str]:
        """
        Execute the complete video translation pipeline.
        """

        try:
            self.logger.info("\nParsing subtitle file")
            segments = parse_srt_file(self.transcript_path, self.logger)

            self.logger.info("\nTranslating subtitles to German")
            translated_segments = translate_segments(
                segments=segments,
                target_lang=Config.TARGET_LANG,
                source_lang=Config.SOURCE_LANG,
                logger=self.logger
            )

            translated_srt_path = self.output_path.replace('.mp4', Config.OUTPUT_SUFFIX_SRT)
            save_translated_srt(translated_segments, translated_srt_path, self.logger)

            self.logger.info("\nExtracting reference audio for voice cloning")
            reference_audio = os.path.join(self.temp_dir, "reference.wav")
            extract_reference_audio(
                video_path=self.video_path,
                output_path=reference_audio,
                duration=Config.REFERENCE_AUDIO_DURATION,
                logger=self.logger
            )

            self.logger.info("\nGenerating German audio with voice cloning")
            final_audio_path = create_synthesized_audio(
                segments=translated_segments,
                reference_audio=reference_audio,
                output_dir=self.temp_dir,
                target_lang=Config.TARGET_LANG,
                logger=self.logger
            )

            output_audio_path = self.output_path.replace('.mp4', Config.OUTPUT_SUFFIX_AUDIO)
            shutil.copy(final_audio_path, output_audio_path)
            self.logger.info(f"Final audio saved to: {output_audio_path}")

            self.logger.info("\nCreating final video with LatentSync lip-sync")
            success = run_latentsync(
                video_path=self.video_path,
                audio_path=final_audio_path,
                output_path=self.output_path,
                inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
                logger=self.logger
            )

            self.logger.info("\nTranslation Complete.")
            self.logger.info(f"Output video: {self.output_path}")
            self.logger.info(f"Output audio: {output_audio_path}")
            self.logger.info(f"Translated subtitles: {translated_srt_path}")

            return self.output_path, output_audio_path, translated_srt_path

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """
        Cleaning up temporary files.
        """
        if not self.keep_temp and os.path.exists(self.temp_dir):
            self.logger.info("Cleaning up temporary files...")
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.debug(f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary files: {e}")



def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Translate video from English to German with voice cloning and "
            "LatentSync lip-sync. This pipeline preserves the original speaker's "
            "voice characteristics while generating synchronized German audio."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic usage
            python main.py --video input.mp4 --transcript input.srt --output outputs/translated.mp4

            # With custom LatentSync parameters
            python main.py --video input.mp4 --transcript input.srt --output outputs/translated.mp4 \\
            --inference-steps 30 --guidance-scale 2.5 --seed 42
            """
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file"
    )

    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to input transcript file in SRT format"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for output video file"
    )

    latentsync_group = parser.add_argument_group('LatentSync parameters')

    latentsync_group.add_argument(
        "--inference-steps",
        type=int,
        default=Config.DEFAULT_INFERENCE_STEPS,
        help=f"Number of diffusion inference steps (default: {Config.DEFAULT_INFERENCE_STEPS})."
    )

    latentsync_group.add_argument(
        "--guidance-scale",
        type=float,
        default=Config.DEFAULT_GUIDANCE_SCALE,
        help=f"Guidance scale for diffusion model (default: {Config.DEFAULT_GUIDANCE_SCALE})."
    )

    latentsync_group.add_argument(
        "--seed",
        type=int,
        default=Config.DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {Config.DEFAULT_SEED})"
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary audio files after processing"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )

    return parser


def main():
    """
    Main function.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        pipeline = VideoTranslationPipeline(
            video_path=args.video,
            transcript_path=args.transcript,
            output_path=args.output,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            keep_temp=args.keep_temp,
            verbose=args.verbose
        )

        pipeline.run()
        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
