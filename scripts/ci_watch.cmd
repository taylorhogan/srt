@echo off
rem Every 5 minutes (scheduled task IrisCIWatch): compare the main and release
rem branch tips on origin and announce, once each, when CI has gone green
rem (release caught main -- safe to `update`) or when main has sat ahead of
rem release longer than a run takes (red, or not coming). See ci_watch.py.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\ci_watch.py >> local\ci_watch.log 2>&1
