@ECHO off
REM Photographs the sky, counts stars if it is dark, pushes to the live panel.
REM Scheduled every 5 minutes by Windows Task Scheduler ("IrisSkyMonitor"),
REM deliberately offset ~2.5 minutes from IrisLiveSkymap so the two jobs do not
REM run at once: each takes about a minute and the observatory PC is also
REM running N.I.N.A.
REM
REM Hard 4-minute timeout, for the same reason live_skymap.bat has one: a run
REM that hangs on a network call holds the log handle open and every later run
REM then fails at the redirect before python starts. The scheduler's own
REM ExecutionTimeLimit did not reap those, so the kill happens here. 4 minutes
REM and not 5: a run must be dead before the next trigger fires, or two live
REM processes contend for the log and the whole failure repeats. A normal run
REM is under a minute -- roughly 20s to pull a frame off the camera, 35s to
REM count, a few more to verify the plate solution -- so 4 minutes is a hang.
REM
REM Single > rather than >>, so a stuck handle cannot silently block the run
REM and the log cannot grow without bound. Exit code passed through: a camera
REM that did not answer must show up as a failed run, not a silent success.
cd /d C:\Users\iriso\Documents\development\srt
set PYTHONIOENCODING=utf-8
REM Same cache the main process uses (see start_srt.bat). Without it astropy
REM falls back to %USERPROFILE%\.astropy, which is not writable here, so the
REM sun-altitude call re-downloads IERS tables over the network on EVERY run
REM and warns its way through the log before falling back to the bundled table.
SET ASTROPY_CACHE_DIR=C:\Users\iriso\Documents\development\srt\local\astropy_cache
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'scripts\sky_monitor.py' -RedirectStandardOutput 'local\sky_monitor.log' -RedirectStandardError 'local\sky_monitor.err' -NoNewWindow -PassThru;" ^
  "if (-not $p.WaitForExit(240000)) { $p.Kill(); Write-Output 'TIMEOUT: killed after 240s'; exit 2 }" ^
  "exit $p.ExitCode"
EXIT /b %ERRORLEVEL%
