@echo off
rem Daily 09:15 (scheduled task IrisConvergenceRefresh): recompute the
rem convergence curves of every target imaged last night whose
rem local/convergence.json entry is older than its newest frame, so the
rem noon plan's auto-stop (control/auto_stop) judges current numbers.
rem See convergence_refresh.py and docs/AUTOMATION_PLAN.md (A1).
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\convergence_refresh.py >> local\convergence_refresh.log 2>&1
