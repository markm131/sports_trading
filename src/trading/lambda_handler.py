# src/trading/lambda_handler.py
"""
AWS Lambda handler for the scanner.

Deploy this to run scans on a schedule (e.g., every 15 mins via EventBridge).

Environment variables required:
    - BETFAIR_USERNAME
    - BETFAIR_PASSWORD
    - BETFAIR_APP_KEY
    - S3_BUCKET (for state storage)
"""

import json
import os

# For Lambda, we need to handle the path differently
import sys
from datetime import datetime

import boto3

sys.path.insert(0, "/var/task")

from src.trading.scanner import (
    BETFAIR_COMPETITIONS,
    SCAN_CONFIG,
    BetfairClient,
    Scanner,
    is_quiet_hours,
)


def sync_state_from_s3(bucket: str):
    """Download state files from S3 before scanning."""
    s3 = boto3.client("s3")

    state_files = [
        "paper_trading/bankroll.json",
        "paper_trading/bets.csv",
        "paper_trading/opportunities.json",
    ]

    for key in state_files:
        local_path = f"/tmp/{key}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            s3.download_file(bucket, key, local_path)
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise


def sync_state_to_s3(bucket: str):
    """Upload state files to S3 after scanning."""
    s3 = boto3.client("s3")

    state_files = [
        "paper_trading/bankroll.json",
        "paper_trading/bets.csv",
        "paper_trading/opportunities.json",
    ]

    for key in state_files:
        local_path = f"/tmp/{key}"
        if os.path.exists(local_path):
            s3.upload_file(local_path, bucket, key)


def handler(event, context):
    """
    Lambda handler for scheduled scans.

    Triggered by EventBridge rule (e.g., rate(15 minutes)).
    """
    bucket = os.environ.get("S3_BUCKET")

    # Check quiet hours
    if is_quiet_hours():
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Quiet hours - skipping scan"}),
        }

    # Sync state from S3
    if bucket:
        sync_state_from_s3(bucket)

    # Run scan
    scanner = Scanner()
    scanner.tracker.cleanup_old()

    client = BetfairClient()

    all_bets = []

    try:
        for league_name, comp_id in BETFAIR_COMPETITIONS.items():
            matches = client.get_football_matches(comp_id, SCAN_CONFIG["days_ahead"])

            if matches:
                bets = scanner.find_value_bets(matches, league_name)
                all_bets.extend(bets)

    finally:
        client.logout()

    # Place paper bets
    if all_bets:
        scanner.place_paper_bets(all_bets)

    # Sync state back to S3
    if bucket:
        sync_state_to_s3(bucket)

    # Return summary
    pending_opps = len(scanner.tracker.get_pending_opportunities())

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "scan_time": datetime.now().isoformat(),
                "bets_placed": len(all_bets),
                "opportunities_tracking": pending_opps,
                "bankroll": scanner.bankroll.bankroll,
            }
        ),
    }


def settle_handler(event, context):
    """
    Lambda handler for settling bets.

    Run once daily after results are updated.
    """
    bucket = os.environ.get("S3_BUCKET")
    date = event.get("date")  # Optional: specific date to settle

    if bucket:
        sync_state_from_s3(bucket)

    scanner = Scanner()
    scanner.settle_bets(date)

    if bucket:
        sync_state_to_s3(bucket)

    status = scanner.bankroll.status()

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "available": status["available"],
                "in_play": status["in_play"],
                "total_bankroll": status["total_bankroll"],
                "realised_pnl": status["realised_pnl"],
                "roi_pct": status["roi_pct"],
            }
        ),
    }
