@echo off
rem Morning shadow report (scheduled task IrisShadowReport): did the shadow
rem conductor's journal match last night's reality? Posts to the webchat.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe apps\shadow_report.py >> local\shadow_report.log 2>&1
