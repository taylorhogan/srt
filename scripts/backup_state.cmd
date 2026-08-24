@echo off
rem Daily backup of small irreplaceable observatory state to OneDrive
rem (scheduled task IrisBackupState). See scripts/backup_state.py for what is
rem included and why the 5.5 GB of bulk capture in local/ is not.
rem
rem Secrets (configs/config_private.py) are NOT included. Add --include-secrets
rem here if you decide API keys in OneDrive is an acceptable trade.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\backup_state.py >> local\backup_state.log 2>&1
