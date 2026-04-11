#!/bin/bash

# Clone E2-TTS repository
git clone https://github.com/SWivid/E2-TTS.git


echo "E2-TTS cloned successfully with all submodules!"

python E2-TTS/ckpts/download_ckpts.py