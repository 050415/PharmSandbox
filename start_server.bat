@echo off
echo ============================================
echo  PharmSandbox - Auto-restart Server
echo  Keep this window open!
echo ============================================

:loop
echo [%date% %time%] Starting Flask...
start "PharmSandbox-Flask" /MIN python run.py

echo Waiting for Flask to initialize (60s)...
timeout /t 60 /nobreak >nul

echo [%date% %time%] Starting ngrok...
start "PharmSandbox-ngrok" /MIN ngrok http 5000

echo [%date% %time%] Server running: https://unfasten-matador-knoll.ngrok-free.dev
echo Health check every 30 seconds. Auto-restart if down.
echo =
echo DO NOT CLOSE THIS WINDOW
echo =

:healthcheck
timeout /t 30 /nobreak >nul
curl -s --max-time 5 http://127.0.0.1:5000/api/health >nul 2>&1
if errorlevel 1 (
  echo [%date% %time%] Flask DOWN - Restarting...
  taskkill /F /IM python.exe 2>nul
  goto loop
)
curl -s --max-time 10 https://unfasten-matador-knoll.ngrok-free.dev/api/health >nul 2>&1
if errorlevel 1 (
  echo [%date% %time%] ngrok DOWN - Restarting...
  taskkill /F /IM ngrok.exe 2>nul
  start "PharmSandbox-ngrok" /MIN ngrok http 5000
)
goto healthcheck
