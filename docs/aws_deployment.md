# AWS EC2 Deployment Guide

Complete guide for deploying the Sports Trading Scanner to AWS EC2 for 24/7 operation.

## Cost Breakdown

| Resource                       | Monthly Cost    |
| ------------------------------ | --------------- |
| EC2 t3.micro (1 vCPU, 1GB RAM) | ~£6-7           |
| S3 (database backups, <1GB)    | ~£0.02          |
| Data transfer                  | ~£0.50          |
| **Total**                      | **~£7-8/month** |

## Why EC2 (not Lambda)?

|                     | Lambda                           | EC2                                     |
| ------------------- | -------------------------------- | --------------------------------------- |
| **How it works**    | Runs for max 15 mins, then stops | Always running (like a PC in the cloud) |
| **For our scanner** | ❌ Can't run daemon mode          | ✅ Perfect for continuous scanning       |
| **Database**        | ❌ No persistent storage          | ✅ SQLite lives on the instance          |
| **Cost**            | Pay per invocation               | ~£7/month flat                          |

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS EC2                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  /home/ec2-user/sports_trading/                     │    │
│  │    ├── db/betfair.db        ← MAIN DATABASE         │    │
│  │    ├── src/                 ← Your code             │    │
│  │    ├── certs/               ← Betfair certs         │    │
│  │    └── logs/                                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          │ Daily backup (cron)               │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  S3 Bucket: sports-trading-backups                  │    │
│  │    └── db/betfair-2026-01-01.db                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Download when you want to analyze
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     YOUR LAPTOP                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  C:\Projects\sports_trading\                        │    │
│  │    ├── db/betfair-cloud.db  ← Downloaded copy       │    │
│  │    └── (run analysis, backtests locally)            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **AWS Account** - Sign up at https://aws.amazon.com
2. **AWS CLI** installed locally - https://aws.amazon.com/cli/
3. **Betfair Certificates** ready to upload

---

## Step 1: Create S3 Bucket for Backups

1. Go to AWS Console → S3
2. Click **Create bucket**
3. Name: `sports-trading-backups` (must be globally unique, add your initials)
4. Region: `eu-west-2` (London)
5. Leave defaults, click **Create bucket**

---

## Step 2: Create EC2 Key Pair

1. Go to AWS Console → EC2 → Key Pairs
2. Click **Create key pair**
3. Name: `sports-trading-key`
4. Type: RSA
5. Format: `.pem` (for SSH)
6. Save the downloaded key file safely!

---

## Step 3: Launch EC2 Instance

1. Go to AWS Console → EC2 → **Launch Instance**

2. **Name**: `sports-trading-bot`

3. **AMI**: Amazon Linux 2023 (free tier eligible)

4. **Instance type**: `t3.micro` (1 vCPU, 1GB RAM)

5. **Key pair**: Select `sports-trading-key`

6. **Network settings**:
   - Create security group
   - Allow SSH from "My IP" only (secure)
   - No HTTP/HTTPS needed (bot doesn't serve web)

7. **Configure storage**: 8 GB gp3 (default is fine)

8. Click **Launch instance**

9. Note the **Public IPv4 address** once running

---

## Step 4: Upload Your Project

From your local Windows machine (PowerShell):

```powershell
# Set variables - UPDATE THESE
$KEY = "C:\Users\YourName\.ssh\sports-trading-key.pem"
$IP = "YOUR-EC2-PUBLIC-IP"
$PROJECT = "C:\Projects\Code_Hub\Python\sports_trading"

# Create directories on EC2
ssh -i $KEY ec2-user@$IP "mkdir -p ~/sports_trading/db ~/sports_trading/logs ~/sports_trading/certs"

# Upload project files (excluding venv and large data)
scp -i $KEY -r $PROJECT\src ec2-user@${IP}:~/sports_trading/
scp -i $KEY -r $PROJECT\scripts ec2-user@${IP}:~/sports_trading/
scp -i $KEY $PROJECT\requirements.txt ec2-user@${IP}:~/sports_trading/
scp -i $KEY $PROJECT\.env ec2-user@${IP}:~/sports_trading/

# Upload Betfair certificates
scp -i $KEY -r $PROJECT\certs\* ec2-user@${IP}:~/sports_trading/certs/

# Upload database
scp -i $KEY $PROJECT\db\betfair.db ec2-user@${IP}:~/sports_trading/db/
```

---

## Step 5: Connect and Set Up Environment

```powershell
ssh -i $KEY ec2-user@$IP
```

Then on EC2:

```bash
cd ~/sports_trading

# Install Python 3.11
sudo yum install python3.11 python3.11-pip git sqlite -y

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test the scanner works
python -m src.trading.scanner --status
```

---

## Step 6: Install as a System Service

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Install the service
sudo bash scripts/install_service.sh

# Check it's running
sudo systemctl status sports-scanner
```

The scanner now runs 24/7, auto-restarts on failure, and starts on boot!

---

## Step 7: Set Up Daily Backups to S3

First, update the S3 bucket name in `scripts/backup_db.sh`, then:

```bash
# Install crontab for daily backups at 3 AM
(crontab -l 2>/dev/null; echo "0 3 * * * /home/ec2-user/sports_trading/scripts/backup_db.sh >> /home/ec2-user/sports_trading/logs/backup.log 2>&1") | crontab -

# Install crontab for daily settlement at 11:30 PM
(crontab -l 2>/dev/null; echo "30 23 * * * cd /home/ec2-user/sports_trading && .venv/bin/python -m src.trading.scanner --results >> logs/settlement.log 2>&1") | crontab -

# Verify crontab
crontab -l
```

---

## Step 8: Configure AWS CLI on EC2 (for S3 backups)

Best option: Attach an IAM role with S3 access to your EC2 instance via AWS Console.

Or manually:
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Region: eu-west-2
# Output format: json
```

---

## Monitoring & Management

### View live logs:
```bash
tail -f ~/sports_trading/logs/scanner.log
```

### Check service status:
```bash
sudo systemctl status sports-scanner
```

### Restart scanner:
```bash
sudo systemctl restart sports-scanner
```

### Check analytics:
```bash
cd ~/sports_trading && source .venv/bin/activate
python -m src.trading.scanner --analytics
```

---

## Download Cloud Data to Laptop

When you want to analyze your data locally, run on your Windows laptop:

```batch
scripts\download_cloud_db.bat
```

This downloads the latest database backup to `db\betfair-cloud.db`.

---

## Cost Optimization Tips

1. **Use Reserved Instance**: Pay upfront for 1 year → ~£4/month (40% savings)
2. **Free tier**: First year is free for t2.micro

---

## Troubleshooting

### Scanner not starting?
```bash
sudo journalctl -u sports-scanner -n 50
```

### Can't connect to Betfair?
```bash
ls -la ~/sports_trading/certs/
cat ~/sports_trading/.env
```

### Database locked?
```bash
sudo systemctl restart sports-scanner
```

---

## Quick Reference

| Command                                     | What it does          |
| ------------------------------------------- | --------------------- |
| `sudo systemctl status sports-scanner`      | Check if running      |
| `sudo systemctl restart sports-scanner`     | Restart the scanner   |
| `tail -f logs/scanner.log`                  | Watch live logs       |
| `python -m src.trading.scanner --status`    | Check bankroll & bets |
| `python -m src.trading.scanner --analytics` | Performance stats     |
| `scripts\download_cloud_db.bat`             | Download DB to laptop |
