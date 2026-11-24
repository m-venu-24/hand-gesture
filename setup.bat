@echo off
cd /d "%~dp0"
py -3.11 --version || (echo Python 3.11 64-bit not found & pause & exit /b 1)
if not exist .venv py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. Next: run_collect.bat
pause
