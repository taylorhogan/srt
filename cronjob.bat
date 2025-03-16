@ECHO off
CLS
del C:\Users\iriso\Documents\development\iris\iris.log
del C:\Users\iriso\Documents\development\iris\logger.log
C:\Users\iriso\Documents\development\iris\venv\Scripts\python C:\Users\iriso\Documents\development\iris\social_server.py >>C:\Users\iriso\Documents\development\iris\logger.log
EXIT /b 0
