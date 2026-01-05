@ECHO off
CLS
pushd
cd %~dp0
C:\Users\iriso\Documents\development\iris\venv\Scripts\python C:\Users\iriso\Documents\development\iris\end_points\end.py %cd%
popd
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0