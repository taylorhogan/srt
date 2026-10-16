@echo off
rem Daily 12:30 check (scheduled task IrisNoonWatch): today's noon check finished
rem and the web chat server answers. Pushes to Pushover ONLY when something is
rem wrong. Added after 2026-10-15, when both hung silently. See noon_watch.py.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\noon_watch.py >> local\noon_watch.log 2>&1
