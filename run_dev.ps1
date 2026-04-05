Write-Host "==================================" -ForegroundColor Green
Write-Host "   Starting ChaosLab SRE Simulator" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host "Starting ASGI Backend (Port 8000)..."
Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location '$PSScriptRoot'; python -m src.server`""

Write-Host "Starting Next.js Frontend (Port 3000)..."
Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location '$PSScriptRoot\frontend'; npm run dev`""

Write-Host "Done! Note: Two new terminal windows have opened." -ForegroundColor Cyan
