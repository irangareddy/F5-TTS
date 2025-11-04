#!/usr/bin/env pwsh
# Task runner wrapper - forwards commands to Task CLI
# Usage: .\task check, .\task train, .\task infer, etc.

$taskExePath = "C:\Users\ranga\AppData\Local\Microsoft\WinGet\Packages\Task.Task_Microsoft.Winget.Source_8wekyb3d8bbwe\task.exe"

if (-not (Test-Path $taskExePath)) {
    Write-Error "Task CLI not found at: $taskExePath"
    Write-Host "Please install Task CLI: winget install Task.Task"
    exit 1
}

# Forward all arguments to task.exe
& $taskExePath @args
exit $LASTEXITCODE
