@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Kaggle3MCleanM4PFinalComponentAblation.ps1" %*
exit /b %ERRORLEVEL%
