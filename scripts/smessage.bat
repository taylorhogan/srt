@ECHO off
CLS
C:\Users\iriso\Documents\development\iris\venv\Scripts\python C:\Users\iriso\Documents\development\iris\end_points\smessage.py  %1
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0

