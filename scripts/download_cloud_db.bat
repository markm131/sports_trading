@echo off
REM Download the latest database from S3 to your local machine
REM Run this on your laptop when you want to analyze the cloud data

set S3_BUCKET=sports-trading-backups
set LOCAL_PATH=C:\Projects\Code_Hub\Python\sports_trading\db

echo ==========================================
echo Downloading latest database from S3...
echo ==========================================

REM Download latest backup
aws s3 cp s3://%S3_BUCKET%/db/betfair-latest.db %LOCAL_PATH%\betfair-cloud.db

echo.
echo Download complete!
echo Database saved to: %LOCAL_PATH%\betfair-cloud.db
echo.
echo You can now run analysis commands with this database:
echo   python -m src.trading.scanner --status
echo   python -m src.trading.scanner --analytics
echo.
pause
