@echo off
REM Auto-restarting launcher for the AI Quant live engine (step9_live_engine.py).
REM Safe to double-click, or register with Windows Task Scheduler (see AUTOSTART_SETUP.md).

cd /d "C:\dhan algo\Research"

:loop
echo [%date% %time%] Starting step9_live_engine.py >> logs\engine_launcher.log
python step9_live_engine.py >> logs\engine_launcher.log 2>&1
echo [%date% %time%] Engine exited (crash or Ctrl+C). Restarting in 15s... >> logs\engine_launcher.log
timeout /t 15 /nobreak >nul
goto loop
