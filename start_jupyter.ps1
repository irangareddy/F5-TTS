# Start Jupyter Lab with Docker
# This script starts Jupyter Lab in a Docker container with GPU support

docker run -d `
  --name f5-tts-jupyter `
  --gpus all `
  --shm-size=2g `
  -p 8888:8888 `
  -v "${PWD}/data:/workspace/F5-TTS/data" `
  -v "${PWD}/ckpts:/workspace/F5-TTS/ckpts" `
  -v "${PWD}/src:/workspace/F5-TTS/src" `
  -v "${PWD}/notebooks:/workspace/F5-TTS/notebooks" `
  -v "${PWD}/tests:/workspace/F5-TTS/tests" `
  -w /workspace/F5-TTS `
  f5-tts:latest `
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

Write-Host "Jupyter Lab started!" -ForegroundColor Green
Write-Host "Access it at: http://localhost:8888" -ForegroundColor Green
Write-Host ""
Write-Host "To stop the container: docker stop f5-tts-jupyter" -ForegroundColor Yellow
Write-Host "To remove the container: docker rm f5-tts-jupyter" -ForegroundColor Yellow

