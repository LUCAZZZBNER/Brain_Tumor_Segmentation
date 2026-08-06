@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Kaggle3MCleanSixModelMultiSeed.ps1" %*
exit /b %ERRORLEVEL%
