@ECHO off
CLS
python C:\home\taylorhogan\Documents\tmh\start.py %1
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0

