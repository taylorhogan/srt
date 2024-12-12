@ECHO off
CLS
pushd
cd %~dp0
python C:\Users\iriso\Documents\development\tmh\end.py %cd%
popd
set myvar=12345
setx NINAESRC %myvar%
EXIT /b 0