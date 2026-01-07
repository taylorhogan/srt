@ECHO off
CLS
pushd
cd %~dp0
C:\Users\iriso\Documents\development\srt\venv\Scripts\python C:\Users\iriso\Documents\development\srt\end_points\end.py %cd%
popd
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0