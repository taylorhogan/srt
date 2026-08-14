@ECHO off
REM Records what every weather model predicted, for scoring against the sky
REM camera later. Scheduled hourly by Windows Task Scheduler ("IrisForecastLog").
REM
REM Runs at :51. Every minute is within 2.5 minutes of one of the 5-minute camera
REM jobs, so there is no clean slot; :51 is simply about as far from both
REM IrisSkyMonitor and IrisLiveSkymap as the grid allows. It matters less here
REM than it does for those two -- this job makes three HTTP calls and touches no
REM camera, so a normal run is a few seconds and it is not competing for the
REM hardware N.I.N.A is using.
REM
REM 2-minute timeout. Same reasoning as sky_monitor.bat: a run that hangs on a
REM network call keeps the log handle open and every later run then fails at the
REM redirect before python starts, which turns one bad fetch into a permanent
REM outage. Well under the hourly interval, so a killed run cannot overlap the
REM next trigger.
REM
REM Single > rather than >>, so a stuck handle cannot silently block the run and
REM the log cannot grow without bound. The data goes to local\forecast_log.jsonl,
REM which IS append-only -- this log is only the last run's console output.
REM
REM `$null = $p.Handle` is load-bearing and must not be tidied away. Without it
REM Start-Process -PassThru (no -Wait) never caches the process handle, so
REM $p.ExitCode reads back EMPTY and every python-level failure is reported to
REM Task Scheduler as success. See sky_monitor.bat, where that bug hid a camera
REM outage as a clean run on 2026-08-13.
cd /d C:\Users\iriso\Documents\development\srt
set PYTHONIOENCODING=utf-8
REM Same cache the main process uses (see start_srt.bat). Not needed for the
REM fetch itself, but configs/config.py pulls in astropy on import and would
REM otherwise re-download IERS tables over the network on every run.
SET ASTROPY_CACHE_DIR=C:\Users\iriso\Documents\development\srt\local\astropy_cache
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'scripts\forecast_log.py' -RedirectStandardOutput 'local\forecast_log.log' -RedirectStandardError 'local\forecast_log.err' -NoNewWindow -PassThru; $null = $p.Handle;" ^
  "if (-not $p.WaitForExit(120000)) { $p.Kill(); Write-Output 'TIMEOUT: killed after 120s'; exit 2 }" ^
  "exit $p.ExitCode"
EXIT /b %ERRORLEVEL%
