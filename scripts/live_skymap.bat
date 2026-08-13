@ECHO off
REM Regenerates the live all-sky chart and pushes it to the web host.
REM Scheduled every 5 minutes by Windows Task Scheduler ("IrisLiveSkymap").
REM
REM Runs under a hard 4-minute timeout. On 2026-08-07 two runs hung -- almost
REM certainly a network call with no timeout inside rank_targets_tonight -- and
REM sat for 43 minutes holding the log handle open, which made every later run
REM fail at the redirect before python started. The scheduler's own
REM ExecutionTimeLimit did not reap them, so the kill happens here instead.
REM
REM 4 minutes, not 5: a run must be dead before the next trigger fires, or two
REM live processes contend for the log again and the whole failure repeats.
REM
REM Single > rather than >>, so a stuck handle cannot silently block the run,
REM and the log cannot grow without bound. Exit code passed through -- this
REM used to end EXIT /b 0, which is why 45 minutes of doing nothing all
REM reported success.
REM
REM The kill also appends one line to local\live_skymap_timeouts.log. That file
REM is written ONLY on a kill, so it holds no handle during a normal run and
REM cannot grow the way an appending stdout log would -- the point is that a
REM timeout otherwise leaves no trace at all except a task exit code, and
REM local\live_skymap.log is overwritten by the next run within 5 minutes.
REM Task Scheduler reports the `exit 2` below as 0x80070002 / 2147942402
REM (HRESULT_FROM_WIN32(2)), which reads like "file not found" but is not.
cd /d C:\Users\iriso\Documents\development\srt
set PYTHONIOENCODING=utf-8
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'scripts\live_skymap.py' -RedirectStandardOutput 'local\live_skymap.log' -RedirectStandardError 'local\live_skymap.err' -NoNewWindow -PassThru;" ^
  "if (-not $p.WaitForExit(240000)) { $p.Kill(); $m = (Get-Date -Format s) + ' TIMEOUT: killed after 240s'; Write-Output $m; Add-Content -Path 'local\live_skymap_timeouts.log' -Value $m; exit 2 }" ^
  "exit $p.ExitCode"
EXIT /b %ERRORLEVEL%
