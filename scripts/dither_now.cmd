@echo off
rem Dither via PWI4, called by the NINA sequence after each exposure in place
rem of NINA's own Direct Guider dither (which cannot move RA on this mount --
rem see scripts/dither_now.py for the measurements).
rem
rem ALWAYS EXITS 0. A failed dither costs one sub's worth of decorrelation; an
rem aborted sequence costs the night. Failures are visible in the log below and
rem in NINA's own output, which is where they belong.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\dither_now.py %* >> local\dither.log 2>&1
exit /b 0
