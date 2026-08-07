@ECHO off
REM Alerts if the live sky chart stops updating. Runs SEPARATELY from the
REM generator on purpose -- a watchdog sharing a process with what it watches
REM cannot report that process hanging.
cd /d C:\Users\iriso\Documents\development\srt
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\live_watchdog.py > local\live_watchdog.log 2>&1
EXIT /b %ERRORLEVEL%
