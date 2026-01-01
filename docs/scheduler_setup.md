# Sports Trading - Scheduler Setup Guide

## Option 1: Windows Task Scheduler (Recommended for Trial)

Your database stays local. Computer must be on during match times.

### Setup Steps

1. **Open Task Scheduler**
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create Scan Task** (runs every 15 mins during match times)
   - Click "Create Task"
   - **General tab**: Name = "Sports Trading - Scan Markets"
   - **Triggers tab**: 
     - New → Daily, Start 11:00, Repeat every 15 minutes for 14 hours
     - This covers 11am-1am (most European match times)
   - **Actions tab**:
     - New → Start a program
     - Program: `C:\Projects\Code_Hub\Python\sports_trading\scripts\scan_markets.bat`
   - **Conditions tab**: Uncheck "Start only if on AC power"
   - **Settings tab**: Check "Run task as soon as possible after scheduled start is missed"

3. **Create Daily Settlement Task** (runs once daily)
   - Click "Create Task"
   - **General tab**: Name = "Sports Trading - Daily Settlement"
   - **Triggers tab**: New → Daily at 10:00 AM
   - **Actions tab**: 
     - Program: `C:\Projects\Code_Hub\Python\sports_trading\scripts\daily_settlement.bat`

### Quick Setup via PowerShell (Admin)

```powershell
# Run as Administrator
$scanAction = New-ScheduledTaskAction -Execute "C:\Projects\Code_Hub\Python\sports_trading\scripts\scan_markets.bat"
$scanTrigger = New-ScheduledTaskTrigger -Daily -At 11:00 -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Hours 14)
Register-ScheduledTask -TaskName "SportsTrading-Scan" -Action $scanAction -Trigger $scanTrigger -Description "Scan Betfair markets every 15 mins"

$settleAction = New-ScheduledTaskAction -Execute "C:\Projects\Code_Hub\Python\sports_trading\scripts\daily_settlement.bat"
$settleTrigger = New-ScheduledTaskTrigger -Daily -At 10:00
Register-ScheduledTask -TaskName "SportsTrading-Settle" -Action $settleAction -Trigger $settleTrigger -Description "Daily bet settlement"
```

---

## Option 2: AWS EC2 (Always-On)

For when you want 24/7 operation without keeping your PC on.

### Architecture
```
┌─────────────────────────────────────┐
│           AWS EC2 (t3.micro)        │
│  ┌─────────────────────────────┐    │
│  │  Scanner daemon             │    │
│  │  SQLite database            │    │
│  │  Cron for settlement        │    │
│  └─────────────────────────────┘    │
│           │                         │
│           ▼                         │
│  ┌─────────────────────────────┐    │
│  │  S3 Bucket (backup)         │    │
│  │  - Daily DB backup          │    │
│  │  - Logs                     │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
           │
           ▼ (download when needed)
    ┌──────────────┐
    │  Your PC     │
    │  - Analysis  │
    │  - Reporting │
    └──────────────┘
```

### Cost
- EC2 t3.micro: ~£7/month
- S3 storage: ~£0.10/month
- **Total: ~£7-8/month**

### Setup Summary
1. Launch EC2 instance (Amazon Linux 2 or Ubuntu)
2. Upload project and certificates
3. Install Python, dependencies
4. Run scanner in daemon mode via systemd
5. Cron job for daily settlement
6. S3 sync for backups

See `docs/aws_deployment.md` for full instructions.

---

## Database Sync Strategy

### If using EC2:

**Daily backup to S3:**
```bash
# Add to crontab on EC2
0 2 * * * aws s3 cp /home/ec2-user/sports_trading/db/betfair.db s3://your-bucket/backups/betfair-$(date +\%Y\%m\%d).db
```

**Download to local for analysis:**
```powershell
aws s3 cp s3://your-bucket/backups/betfair-latest.db .\db\betfair-cloud.db
```

### If using local Task Scheduler:
- Database is already local
- Just backup periodically: `copy db\betfair.db db\backups\betfair-%date%.db`

---

## Monitoring

### Check scanner is running:
```powershell
# View recent log
Get-Content logs\scanner.log -Tail 50

# Check last scan time
python -m src.trading.scanner --status
```

### Discord/Email Alerts (Optional)
Set environment variables in `.env`:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

The notification system will alert you on:
- Bets placed
- Daily P/L summary
- Errors

---

## Recommended Schedule for Trial Month

| Task          | Frequency               | Script                 |
| ------------- | ----------------------- | ---------------------- |
| Scan markets  | Every 15 mins, 11am-1am | `scan_markets.bat`     |
| Settle bets   | Daily at 10am           | `daily_settlement.bat` |
| Export report | Weekly (manual)         | `--export`             |
| Check status  | Daily (manual)          | `--status`             |

---

## Quick Commands Reference

```powershell
# Run a scan now
python -m src.trading.scanner

# Check status
python -m src.trading.scanner --status

# View recent bets
python -m src.trading.scanner --bets

# Settle yesterday's bets
python -m src.trading.scanner --results

# View analytics (after bets settle)
python -m src.trading.scanner --analytics

# Export all bets to CSV
python -m src.trading.scanner --export

# Run backtest
python -m src.trading.backtest --start 2025-01-01 --end 2025-12-31
```
