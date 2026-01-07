@ECHO off
CLS
C:\Users\iriso\Documents\development\srt\venv\Scripts\python C:\Users\iriso\Documents\development\srt\end_points\smessage.py  %1
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0

