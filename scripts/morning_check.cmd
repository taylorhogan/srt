@echo off
rem Daily 09:00 rest-state check (scheduled task IrisMorningCheck): roof closed,
rem scope parked, mount off, Pegasus ports 1-3 off, Kasa plugs reachable.
rem Pushes to Pushover only when something is wrong. See scripts/morning_check.py.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\morning_check.py >> local\morning_check.log 2>&1
