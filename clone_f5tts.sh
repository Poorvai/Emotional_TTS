#!/bin/bash

# Clone F5-TTS repository
git clone https://github.com/SWivid/F5-TTS.git


echo "F5-TTS cloned successfully with all submodules!"

# Install Hugging Face Hub and download checkpoints
pip install huggingface_hub 
python F5-TTS/ckpts/download_ckpts.py 

# Installing requirements
pip install ./F5-TTS

# Navigate into the repo
cd F5-TTS