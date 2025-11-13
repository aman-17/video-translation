FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libgl1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install git+https://github.com/coqui-ai/TTS.git@dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e
RUN pip3 install --no-cache-dir -U "huggingface_hub[cli]"

COPY . .

RUN mkdir -p latentsync/checkpoints && \
    huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir latentsync/checkpoints && \
    huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir latentsync/checkpoints

RUN mkdir -p outputs

ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
    python3 -c "from config import Config; print('Config OK')" && \
    python3 -c "from subtitles import SubtitleSegment; print('Subtitles OK')" && \
    python3 -c "from translation import translate_segments; print('Translation OK')"

CMD ["python3", "main.py", "--help"]
