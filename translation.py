"""
Translation module that handles all translation operations, converting 
subtitle segments from one language to another.
"""

import logging
from typing import List

from deep_translator import GoogleTranslator

from exceptions import TranslationError
from subtitles import SubtitleSegment


def translate_segments(
    segments: List[SubtitleSegment],
    target_lang: str,
    source_lang: str,
    logger: logging.Logger
) -> List[SubtitleSegment]:
    """
    Translate subtitle segments from source to target language.

    This function translates each subtitle segment independently while
    preserving timing information.

    Args:
        segments: List of subtitle segments to translate
        target_lang: Target language code
        source_lang: Source language code
        logger: Logger instance for logging operations

    Returns:
        List of translated SubtitleSegment objects with same timing
    """
    logger.info(f"Translating {len(segments)} segments from {source_lang} to {target_lang}")

    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
    except Exception as e:
        raise TranslationError(f"Failed to initialize translator: {e}")

    translated_segments = []
    failed_count = 0

    for segment in segments:
        try:
            translated_text = translator.translate(segment.text)
            translated_segment = SubtitleSegment(
                index=segment.index,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=translated_text
            )
            translated_segments.append(translated_segment)

            logger.debug(f"[{segment.index}] '{segment.text}' -> '{translated_text}'")

        except Exception as e:
            logger.warning(f"Translation failed for segment {segment.index}: {e}. "
                          f"Using original text.")
            translated_segments.append(segment)
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"Failed to translate {failed_count} out of {len(segments)} segments")
    else:
        logger.info("All segments translated successfully")

    return translated_segments
