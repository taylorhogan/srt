@ECHO off
CLS
pushd
cd %~dp0
python C:\home\taylorhogan\Documents\tmh\end.py %cd%
popd
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0