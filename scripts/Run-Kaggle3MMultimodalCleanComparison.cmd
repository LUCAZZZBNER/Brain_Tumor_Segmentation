@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Kaggle3MMultimodalCleanComparison.ps1" %*
exit /b %ERRORLEVEL%
