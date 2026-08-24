@echo off
rem Daily sky-camera plate-solution health check (scheduled task IrisSkysolveAuto).
rem
rem Runs mid-MORNING on purpose: a blind solve needs a DARK FRAME, not darkness,
rem so working from the night's best archived frame avoids racing dawn and
rem avoids competing with an imaging run for CPU. It re-solves only when the
rem stored solution has demonstrably stopped fitting -- see the module docstring
rem in scripts/skysolve_auto.py for why this is not run every clear night.
rem
rem Exits quietly on a night that fits, which is almost every night.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\skysolve_auto.py >> local\skysolve_auto.log 2>&1
