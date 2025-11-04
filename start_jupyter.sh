#!/usr/bin/env bash
# Start Jupyter Lab with Docker
# This script starts Jupyter Lab in a Docker container with GPU support

docker run -d \
  --name f5-tts-jupyter \
  --gpus all \
  --shm-size=2g \
  -p 8888:8888 \
  -v "${PWD}/data:/workspace/F5-TTS/data" \
  -v "${PWD}/ckpts:/workspace/F5-TTS/ckpts" \
  -v "${PWD}/src:/workspace/F5-TTS/src" \
  -v "${PWD}/notebooks:/workspace/F5-TTS/notebooks" \
  -v "${PWD}/tests:/workspace/F5-TTS/tests" \
  -w /workspace/F5-TTS \
  f5-tts:latest \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

echo "Jupyter Lab started!"
echo "Access it at: http://localhost:8888"
echo ""
echo "To stop the container: docker stop f5-tts-jupyter"
echo "To remove the container: docker rm f5-tts-jupyter"

