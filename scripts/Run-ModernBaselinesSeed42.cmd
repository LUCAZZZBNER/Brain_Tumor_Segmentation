@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-ModernBaselinesSeed42.ps1" %*
exit /b %ERRORLEVEL%
