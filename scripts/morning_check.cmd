@echo off
rem Daily 09:00 rest-state check (scheduled task IrisMorningCheck): roof closed,
rem scope parked, mount off, Pegasus ports 1-3 off, Kasa plugs reachable.
rem Pushes to Pushover ONLY when something is wrong (operator's call, confirmed
rem 2026-09-19 after the 09-17..09-19 daily trial). The speed test is measured and
rem logged every morning; the numbers ride along on an alert. See morning_check.py.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\morning_check.py >> local\morning_check.log 2>&1
