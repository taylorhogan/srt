  @ECHO off
  cd /d C:\Users\iriso\Documents\development\srt
  SET ASTROPY_CACHE_DIR=C:\Users\iriso\Documents\development\srt\local\astropy_cache
  uv run end_points\start_srt.py
  EXIT /b 0
