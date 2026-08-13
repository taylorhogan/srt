@ECHO off
REM Photographs the sky through the open roof with the ASI all-sky camera,
REM measures it, and publishes it to the live panel -- but only while the roof
REM is confirmed open. See scripts/allsky_monitor.py for why that gate exists:
REM with the roof shut this camera takes a well-exposed picture of the roof's
REM own underside, and the star detector returns hundreds of false detections
REM off the texture.
REM
REM Scheduled every 5 minutes by Windows Task Scheduler ("IrisAllSkyMonitor"),
REM offset from IrisSkyMonitor and IrisLiveSkymap so the three jobs do not run
REM at once: each takes about a minute and the observatory PC is also running
REM N.I.N.A.
REM
REM Hard 4-minute timeout, for the same reason the other two have one: a run
REM that hangs on a network call holds the log handle open and every later run
REM then fails at the redirect before python starts. 4 minutes and not 5: a run
REM must be dead before the next trigger fires. A normal run is well under two
REM minutes -- up to 30s of exposure, ~16s to count, a few more to verify the
REM plate solution, and on a clear night the star test settles the roof question
REM without touching the safety camera at all.
REM
REM Single > rather than >>, so a stuck handle cannot silently block the run and
REM the log cannot grow without bound. Exit code passed through: a camera that
REM did not answer must show up as a failed run, not a silent success.
cd /d C:\Users\iriso\Documents\development\srt
set PYTHONIOENCODING=utf-8
REM Same cache the main process uses (see start_srt.bat). Without it astropy
REM falls back to %USERPROFILE%\.astropy, which is not writable here, so the
REM sun-altitude call re-downloads IERS tables over the network on EVERY run and
REM warns its way through the log before falling back to the bundled table.
SET ASTROPY_CACHE_DIR=C:\Users\iriso\Documents\development\srt\local\astropy_cache
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'scripts\allsky_monitor.py' -RedirectStandardOutput 'local\allsky_monitor.log' -RedirectStandardError 'local\allsky_monitor.err' -NoNewWindow -PassThru;" ^
  "if (-not $p.WaitForExit(240000)) { $p.Kill(); Write-Output 'TIMEOUT: killed after 240s'; exit 2 }" ^
  "exit $p.ExitCode"
EXIT /b %ERRORLEVEL%
