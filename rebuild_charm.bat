@echo off
call conda.bat activate simnibs
cd /d C:\Users\mdrs\Documents\simnibs
pyinstaller --clean charm.spec
if %ERRORLEVEL% EQU 0 (
    echo Build successful!
) else (
    echo Build failed with error code %ERRORLEVEL%
)
pause
