# Video Translation: English to German

A solution for translating videos from English to German with voice cloning and lip-sync capabilities using LatentSync.

## Overview

This pipeline takes an English video with subtitles and produces a German-translated version that:
- Preserves the original speaker's voice characteristics
- Maintains synchronization between audio and video
- Applies lip-sync to match the new German audio


## Features

- **SRT Parsing**: Handles standard SRT subtitle files with precise timing
- **Text Translation**: Uses Google Translate via `deep-translator` for English to German translation
- **Voice Cloning**: Employs Coqui XTTS v2 for multilingual voice cloning
- **Timing Preservation**: Maintains subtitle timing and synchronization
- **Lip-Sync**: Uses LatentSync for realistic lip movement synchronization


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aman-17/video-translation.git
cd video-translation
```

### 2. Installing dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

```bash
# creates environment and downloads the dependencies
source setup_env.sh
```

```bash
pip install git+https://github.com/coqui-ai/TTS.git@dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e
```
> You might encounter a NumPy-related issue while installing TTS. Please ignore it as all the required sub-packages will be installed except NumPy.

#### Docker Installation (Optional)

```bash
# Build the image
docker-compose build

# Run with your video
docker-compose run --rm video-translation \
  python3 main.py \
    --video /app/inputs/input_video.mp4 \
    --transcript /app/inputs/input_subtitles.srt \
    --output /app/outputs/translated_video.mp4
```

### 3. Download the models

```bash
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir latentsync/checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir latentsync/checkpoints
```

## Usage

### Basic Usage

```bash
python main.py \
  --video path/to/input_video.mp4 \
  --transcript path/to/subtitles.srt \
  --output outputs/translated_video.mp4
```

### Advanced Options

```bash
python main.py \
  --video input.mp4 \
  --transcript input.srt \
  --output outputs/translated.mp4 \
  --inference-steps 30 \
  --guidance-scale 2.5 \
  --seed 42 \
  --keep-temp
```
### Example

```bash
python main.py \
  --video ./Tanzania-2.mp4 \
  --transcript ./Tanzania-caption.srt \
  --output outputs/translated_video.mp4
```

### Command-Line Arguments

- `--video`: Input video file path (required)
- `--transcript`: Input SRT subtitle file path (required)
- `--output`: Output video file path (required)
- `--inference-steps`: Number of diffusion steps for LatentSync (default: 25, higher = better quality but slower)
- `--guidance-scale`: Guidance scale for LatentSync (default: 2.0, controls adherence to audio)
- `--seed`: Random seed for reproducibility (default: 1247)
- `--keep-temp`: Keep temporary audio segment files for debugging

### Expected Output Files

After successful execution, you'll find in [outputs](https://github.com/aman-17/video-translation/tree/master/outputs) directory:
- `outputs/translated_video.mp4` - The final translated video with lip-sync
- `outputs/translated_video_audio.wav` - The generated German audio track
- `outputs/translated_video_translated.srt` - Translated German subtitles

## Limitations

1. Voice cloning is not 100% accurate. I did not find any open-source voice cloning models that are great for this task.

