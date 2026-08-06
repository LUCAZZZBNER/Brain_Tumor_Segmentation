@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Kaggle3MCleanInputAugmentationAblation.ps1" %*
exit /b %ERRORLEVEL%
