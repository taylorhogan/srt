  @ECHO off
  cd /d C:\Users\iriso\Documents\development\srt
  uv run end_points\start_srt.py >>cron.log 2>&1
  EXIT /b 0
