@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Kaggle3MM4PretrainingAblation.ps1" %*
exit /b %ERRORLEVEL%
