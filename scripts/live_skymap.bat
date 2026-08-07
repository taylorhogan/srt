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
cd /d C:\Users\iriso\Documents\development\srt
set PYTHONIOENCODING=utf-8
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'scripts\live_skymap.py' -RedirectStandardOutput 'local\live_skymap.log' -RedirectStandardError 'local\live_skymap.err' -NoNewWindow -PassThru;" ^
  "if (-not $p.WaitForExit(240000)) { $p.Kill(); Write-Output 'TIMEOUT: killed after 240s'; exit 2 }" ^
  "exit $p.ExitCode"
EXIT /b %ERRORLEVEL%
