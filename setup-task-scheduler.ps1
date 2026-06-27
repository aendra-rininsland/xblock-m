# Run this ONCE from a regular (non-admin) PowerShell window on your Windows desktop.
# It creates a Task Scheduler job that starts the XBlock service whenever you log in.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File "\\wsl.localhost\debian\home\aendra\xblock-docker\setup-task-scheduler.ps1"

$taskName = "XBlock-Worker"
$wslExe   = "$env:SystemRoot\System32\wsl.exe"
$wslArgs  = "-d Debian bash /home/aendra/xblock-docker/start-xblock.sh"

# Remove old task if present
schtasks /Delete /TN $taskName /F 2>$null

# Create the logon trigger task
$result = schtasks /Create `
    /TN  $taskName `
    /TR  "`"$wslExe`" $wslArgs" `
    /SC  ONLOGON `
    /RL  LIMITED `
    /F 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Task '$taskName' registered successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "To start it right now without logging out, run:"
    Write-Host "  schtasks /Run /TN '$taskName'"
    Write-Host ""
    Write-Host "Dashboard will be at: http://localhost:8080"
} else {
    Write-Host "Failed to register task:" -ForegroundColor Red
    Write-Host $result
}
