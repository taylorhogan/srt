  @ECHO off
  cd /d C:\Users\iriso\Documents\development\srt
  git pull

  REM Launch Pegasus Unity in the background and wait for it to be ready
  start "" "C:\Program Files\Pegasus Astro\Unity Platform\UnityPlatform.exe"
  :WAIT_UNITY
  curl -s -o NUL http://localhost:32000/Server/DeviceManager/Connected
  IF %ERRORLEVEL% NEQ 0 (
    timeout /t 3 /nobreak >NUL
    GOTO WAIT_UNITY
  )
  echo Pegasus Unity is ready.

  SET ASTROPY_CACHE_DIR=C:\Users\iriso\Documents\development\srt\local\astropy_cache
  uv run end_points\start_srt.py
  EXIT /b 0
