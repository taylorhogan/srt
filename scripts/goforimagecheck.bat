@ECHO off
CLS
pushd
cd %~dp0
uv run C:\Users\iriso\Documents\development\srt\end_points\goforimagecheck.py %cd%
popd
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0