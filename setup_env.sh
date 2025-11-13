#!/bin/bash

conda create -y -n video_gen python=3.10.13
conda activate video_gen

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

conda install -y -c conda-forge ffmpeg
sudo apt -y install libgl1

pip install -r requirements.txt --no-cache
pip install -U "huggingface_hub[cli]"