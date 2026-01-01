@echo off
REM Settle yesterday's bets and fetch new results
REM Schedule this daily at 10am

cd /d C:\Projects\Code_Hub\Python\sports_trading
call .venv\Scripts\activate.bat

echo ============================================ >> logs\daily.log
echo Daily Settlement - %date% %time% >> logs\daily.log
echo ============================================ >> logs\daily.log

REM Fetch latest results
python -m src.data.fetcher --all-leagues --update >> logs\daily.log 2>&1

REM Load into database
python -m src.data.db_writer --all >> logs\daily.log 2>&1

REM Settle any bets from yesterday
python -m src.trading.scanner --results >> logs\daily.log 2>&1

REM Show status
python -m src.trading.scanner --status >> logs\daily.log 2>&1

exit /b 0
