@echo off
rem Daily 09:00 rest-state check (scheduled task IrisMorningCheck): roof closed,
rem scope parked, mount off, Pegasus ports 1-3 off, Kasa plugs reachable.
rem Pushes to Pushover only when something is wrong -- except through 2026-09-19,
rem when it pushes every morning to prove the task runs. See scripts/morning_check.py.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\morning_check.py --always-push-until 2026-09-19 >> local\morning_check.log 2>&1
