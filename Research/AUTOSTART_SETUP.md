# Auto-starting the live engine

`start_live_engine.bat` runs `step9_live_engine.py` in a loop, restarting it 15s after
any crash or manual stop, and logs everything to `logs\engine_launcher.log`. Register
it with Windows Task Scheduler so it starts without you opening a terminal each day.

## Option A - one command (recommended)

Open **PowerShell as Administrator** and run:

```powershell
schtasks /Create /TN "AI Quant Live Engine" /TR "\"C:\dhan algo\Research\start_live_engine.bat\"" /SC ONSTART /RU "%USERNAME%" /RL HIGHEST /F
```

This registers a task that starts the engine automatically every time you log into
Windows. To instead start it every trading morning at a fixed time (e.g. 9:00 AM,
15 minutes before market open) use a daily trigger instead:

```powershell
schtasks /Create /TN "AI Quant Live Engine" /TR "\"C:\dhan algo\Research\start_live_engine.bat\"" /SC DAILY /ST 09:00 /RU "%USERNAME%" /RL HIGHEST /F
```

Check it's registered:
```powershell
schtasks /Query /TN "AI Quant Live Engine"
```

Remove it later if needed:
```powershell
schtasks /Delete /TN "AI Quant Live Engine" /F
```

## Option B - Task Scheduler GUI

1. Open **Task Scheduler** -> **Create Task**.
2. **General** tab: name it `AI Quant Live Engine`, select "Run whether user is logged
   on or not" if you want it to survive logout, and check "Run with highest privileges".
3. **Triggers** tab: New -> "At log on" (or "Daily" at 09:00).
4. **Actions** tab: New -> Program/script: `C:\dhan algo\Research\start_live_engine.bat`.
5. **Settings** tab: check "If the task fails, restart every: 1 minute" for extra
   resilience on top of the batch file's own restart loop.
6. Save (it will ask for your Windows password if "run whether logged on or not" is set).

## Notes

- The engine only does real work during NSE market hours (09:15-15:30 IST); outside
  that window it just sleeps, so leaving the scheduled task running all day is fine.
- `logs\engine_launcher.log` accumulates over time - periodically clear it if it gets large.
- The dashboard (`streamlit run step8_dashboard.py` or `python -m streamlit run
  step8_dashboard.py`) is a separate process and is NOT auto-started by this task -
  start it manually when you want to look at the terminal, since it opens a browser tab.
