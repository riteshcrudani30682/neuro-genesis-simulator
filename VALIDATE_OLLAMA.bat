@echo off
setlocal
cd /d "%~dp0"
echo Neuro-Genesis V2 - Live Ollama validation
echo Open Ollama first. This checks an installed model without downloading anything.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m experiments.ollama_validation --interactive
  goto finished
)
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -m experiments.ollama_validation --interactive
  goto finished
)
py -3 --version >nul 2>&1
if not errorlevel 1 (
  py -3 -m experiments.ollama_validation --interactive
  goto finished
)
python --version >nul 2>&1
if not errorlevel 1 (
  python -m experiments.ollama_validation --interactive
  goto finished
)
echo Python 3 was not found. Install Python or activate the project's environment.
pause
exit /b 2
:finished
set "NG_VALIDATION_EXIT=%errorlevel%"
echo.
echo Validation exit code: %NG_VALIDATION_EXIT%  [0=passed, 1=failed, 2=blocked]
echo Report location is printed above. Send report.json back for review.
pause
exit /b %NG_VALIDATION_EXIT%
