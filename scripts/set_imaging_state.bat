@echo off
REM Set the imaging state from outside Python.
REM Usage: set_imaging_state.bat <STATE>
REM Valid states: NONE  ACTIVE  IN_PRELUDE  DONE_PRELUDE  IN_MAIN  DONE_MAIN

if "%1"=="" (
    echo Usage: set_imaging_state.bat ^<STATE^>
    echo Valid states: NONE  ACTIVE  IN_PRELUDE  DONE_PRELUDE  IN_MAIN  DONE_MAIN
    exit /b 1
)

set STATE=%1
set LOG="%~dp0..\imaging_state.log"

if /I "%STATE%"=="NONE"         goto valid
if /I "%STATE%"=="ACTIVE"       goto valid
if /I "%STATE%"=="IN_PRELUDE"   goto valid
if /I "%STATE%"=="DONE_PRELUDE" goto valid
if /I "%STATE%"=="IN_MAIN"      goto valid
if /I "%STATE%"=="DONE_MAIN"    goto valid

echo Unknown state: %STATE%
echo %DATE% %TIME% ERROR: Unknown state: %STATE%>>%LOG%
echo Valid states: NONE  ACTIVE  IN_PRELUDE  DONE_PRELUDE  IN_MAIN  DONE_MAIN
exit /b 1

:valid
echo IMAGING_STATE %STATE%>"%~dp0..\imaging.txt"
echo Imaging state set to %STATE%
echo %DATE% %TIME% IMAGING_STATE set to %STATE%>>%LOG%
