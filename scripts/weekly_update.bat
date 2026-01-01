@echo off
REM Complete weekly data update script with full logging
cd /d C:\Projects\Code_Hub\Python\sports_trading
call .venv\Scripts\activate.bat

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Set log file
set LOGFILE=logs\weekly_update.log

REM Redirect all output to log file AND console
REM This keeps a full record while showing progress

echo ============================================================ >> "%LOGFILE%"
echo Weekly Data Update Started - %date% %time% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

echo ============================================================
echo Weekly Data Update - %date% %time%
echo ============================================================

REM Settle any pending bets from last week
echo Settling pending bets... >> "%LOGFILE%"
python -m src.trading.scanner --results >> "%LOGFILE%" 2>&1

REM Run the entire update process
python -m src.data.fetcher --all-leagues --update >> "%LOGFILE%" 2>&1
set LOADER_RESULT=%errorlevel%
if %LOADER_RESULT% equ 2 goto :no_updates
if %LOADER_RESULT% neq 0 goto :error

python -m src.data.cleaner --all-leagues >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 goto :error

python -m src.data.db_writer --all >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 goto :error

REM Export weekly bet report
python -m src.trading.scanner --export >> "%LOGFILE%" 2>&1

:success
echo [OK] Update complete! >> "%LOGFILE%"
echo [OK] Update complete!
echo See full log: %LOGFILE%
goto :end

:no_updates
echo No new matches found >> "%LOGFILE%"
echo No new matches found - no updates needed
goto :end

:error
echo [ERROR] Check log file >> "%LOGFILE%"
echo [ERROR] Check log: %LOGFILE%
color 0C

:end
echo ============================================================ >> "%LOGFILE%"
echo.
echo Log file: %LOGFILE%

REM Keep window open (comment out for Task Scheduler)
echo.
pause

exit /b %errorlevel%