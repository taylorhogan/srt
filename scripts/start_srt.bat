  @ECHO off
  cd /d C:\Users\iriso\Documents\development\srt
  REM Deploy the last GREEN commit: CI fast-forwards `release` only when the
  REM tests pass, so a red main never reaches the running observatory. ff-only
  REM because this checkout is also the dev machine -- if local work is ahead
  REM of release the merge refuses and the boot runs the local code as-is.
  git fetch origin release
  git merge --ff-only origin/release

  REM Launch Pegasus Unity in the background and wait for it to be ready
  start "" "C:\Program Files (x86)\Pegasus Astro\Unity Platform\Peg.UI.exe"
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
