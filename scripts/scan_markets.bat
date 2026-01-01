@echo off
REM Runs a single scan of all markets
REM Schedule this with Windows Task Scheduler every 15-30 minutes

cd /d C:\Projects\Code_Hub\Python\sports_trading
call .venv\Scripts\activate.bat

REM Run scan (logs to file)
python -m src.trading.scanner --days 2 >> logs\scanner.log 2>&1

REM Exit cleanly
exit /b 0
