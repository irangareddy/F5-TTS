# Start Jupyter Lab with Docker
# This script starts Jupyter Lab in a Docker container with GPU support

# Check if container exists and stop/remove it
$containerExists = docker ps -a --filter "name=f5-tts-jupyter" --format "{{.Names}}"
if ($containerExists -eq "f5-tts-jupyter") {
    Write-Host "Stopping existing container..." -ForegroundColor Yellow
    docker stop f5-tts-jupyter 2>$null
    Write-Host "Removing existing container..." -ForegroundColor Yellow
    docker rm f5-tts-jupyter 2>$null
}

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

