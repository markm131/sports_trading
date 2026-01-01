#!/bin/bash
# Database backup script - runs daily via cron
# Backs up SQLite database to S3

PROJECT_DIR="/home/ec2-user/sports_trading"
S3_BUCKET="sports-trading-backups"  # Change this to your bucket name
DATE=$(date +%Y-%m-%d)

# Create backup directory
mkdir -p ${PROJECT_DIR}/backups

# Copy database (safe copy while running)
sqlite3 ${PROJECT_DIR}/db/betfair.db ".backup '${PROJECT_DIR}/backups/betfair-${DATE}.db'"

# Upload to S3
aws s3 cp ${PROJECT_DIR}/backups/betfair-${DATE}.db s3://${S3_BUCKET}/db/betfair-${DATE}.db

# Also keep a "latest" copy for easy download
aws s3 cp ${PROJECT_DIR}/backups/betfair-${DATE}.db s3://${S3_BUCKET}/db/betfair-latest.db

# Clean up old local backups (keep last 7 days)
find ${PROJECT_DIR}/backups -name "betfair-*.db" -mtime +7 -delete

# Clean up old S3 backups (keep last 30 days) - optional
# aws s3 ls s3://${S3_BUCKET}/db/ | while read -r line; do
#     createDate=$(echo $line | awk {'print $1" "$2'})
#     createDate=$(date -d "$createDate" +%s)
#     olderThan=$(date -d "-30 days" +%s)
#     if [[ $createDate -lt $olderThan ]]; then
#         fileName=$(echo $line | awk {'print $4'})
#         aws s3 rm s3://${S3_BUCKET}/db/$fileName
#     fi
# done

echo "$(date): Backup completed - betfair-${DATE}.db uploaded to S3"
