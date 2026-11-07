@echo off
cd /d %~dp0
if not exist .venv (
  echo [setup] creating venv and installing deps...
  uv venv
  uv pip install --python .venv/Scripts/python.exe -r requirements.txt
)
echo [relay] starting... (dashboard: http://127.0.0.1:8800/)
.venv\Scripts\python run.py
pause
