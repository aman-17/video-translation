"""
Subtitle processing that handles all subtitle-related operations including parsing SRT files,
representing subtitle segments, and saving translated subtitles.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from exceptions import VideoFileNotFoundError


@dataclass
class SubtitleSegment:
    """
    Represents a single subtitle segment with timing information.

    This dataclass encapsulates all information about a subtitle segment
    including its position, timing, and text content.

    Args:
        index: Sequential index of the subtitle
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        text: Subtitle text content
    """
    index: int
    start_time: float
    end_time: float
    text: str

    def __post_init__(self):
        """
        Validate and clean segment data after initialization.
        """
        self.text = self.text.strip()
        if self.start_time < 0 or self.end_time < 0:
            raise ValueError("Timestamps cannot be negative")
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")

    @property
    def duration(self) -> float:
        """
        Calculate segment duration in seconds.
        """
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        """
        String representation of the subtitle segment.
        """
        return (f"SubtitleSegment(index={self.index}, "
                f"start={self.start_time:.2f}s, "
                f"end={self.end_time:.2f}s, "
                f"text='{self.text[:30]}...')")


def parse_srt_timestamp(timestamp: str) -> float:
    """
    Convert SRT timestamp format to seconds.

    Args:
        timestamp: Timestamp string in format 'HH:MM:SS,mmm'

    Returns:
        Timestamp converted to seconds

    Example:
        >>> parse_srt_timestamp('00:01:23,456')
        83.456
    """
    try:
        time_parts = timestamp.replace(',', '.').split(':')
        if len(time_parts) != 3:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])

        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse timestamp '{timestamp}': {e}")


def parse_srt_file(srt_path: str, logger: logging.Logger) -> List[SubtitleSegment]:
    """
    Parse SRT subtitle file and extract segments with timing information.

    Args:
        srt_path: Path to the SRT file
        logger: Logger instance for logging operations

    Returns:
        List of SubtitleSegment objects
    """
    if not Path(srt_path).exists():
        raise VideoFileNotFoundError(f"SRT file not found: {srt_path}")

    logger.info(f"Parsing SRT file: {srt_path}")
    segments = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Splitting by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            logger.debug(f"Skipping malformed block: {block[:50]}...")
            continue

        try:
            index = int(lines[0])

            # Parse timestamp line: "00:00:01,000 --> 00:00:03,500"
            timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
            timestamp_match = re.match(timestamp_pattern, lines[1])

            if not timestamp_match:
                logger.warning(f"Invalid timestamp format in block {index}")
                continue

            start_time = parse_srt_timestamp(timestamp_match.group(1))
            end_time = parse_srt_timestamp(timestamp_match.group(2))
            text = ' '.join(lines[2:])  # joining remaining lines as text

            segment = SubtitleSegment(index, start_time, end_time, text)
            segments.append(segment)

        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing subtitle block: {e}")
            continue

    logger.info(f"Successfully parsed {len(segments)} subtitle segments")
    return segments


def save_translated_srt(
    segments: List[SubtitleSegment],
    output_path: str,
    logger: logging.Logger
) -> None:
    """
    Save translated subtitle segments to SRT file.

    Args:
        segments: List of subtitle segments to save
        output_path: Path where SRT file will be saved
        logger: Logger instance
    """
    logger.info(f"Saving translated subtitles to: {output_path}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            start_h = int(segment.start_time // 3600)
            start_m = int((segment.start_time % 3600) // 60)
            start_s = int(segment.start_time % 60)
            start_ms = int((segment.start_time % 1) * 1000)

            end_h = int(segment.end_time // 3600)
            end_m = int((segment.end_time % 3600) // 60)
            end_s = int(segment.end_time % 60)
            end_ms = int((segment.end_time % 1) * 1000)

            f.write(f"{segment.index}\n")
            f.write(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> ")
            f.write(f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}\n")
            f.write(f"{segment.text}\n\n")

    logger.info(f"Saved {len(segments)} translated segments")
