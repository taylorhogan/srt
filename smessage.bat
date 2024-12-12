@ECHO off
CLS
python C:\Users\iriso\Documents\development\tmh\smessage.py %1
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0

