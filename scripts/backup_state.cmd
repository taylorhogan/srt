@echo off
rem Daily backup of small irreplaceable observatory state to OneDrive
rem (scheduled task IrisBackupState). See scripts/backup_state.py for what is
rem included and why the 5.5 GB of bulk capture in local/ is not.
rem
rem Secrets (configs/config_private.py) ARE included since 2026-09-25 (owner's
rem call): the zips are pulled to the Spark by sync_state_to_spark.bsh, and
rem losing the private config means re-entering every credential.
cd /d C:\Users\iriso\Documents\development\srt
.venv\Scripts\python.exe scripts\backup_state.py --include-secrets >> local\backup_state.log 2>&1
