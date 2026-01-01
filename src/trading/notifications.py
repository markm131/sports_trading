# src/trading/notifications.py
"""
Notification system for bet alerts.

Supports Discord webhooks and email notifications.
"""

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()


class DiscordNotifier:
    """Send notifications via Discord webhook."""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, title: str, message: str, color: int = 0x00FF00) -> bool:
        """Send a Discord embed message."""
        if not self.webhook_url:
            return False

        embed = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": "Sports Trading Bot"},
                }
            ]
        }

        try:
            req = Request(
                self.webhook_url,
                data=json.dumps(embed).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urlopen(req)
            return True
        except Exception as e:
            print(f"Discord notification failed: {e}")
            return False

    def send_bet_placed(self, bet: Dict) -> bool:
        """Send notification when a bet is placed."""
        title = "🎯 New Bet Placed"
        message = (
            f"**{bet['league']}**\n"
            f"{bet['home_team']} vs {bet['away_team']}\n\n"
            f"Selection: **{bet['selection'].upper()}** @ {bet['odds']:.2f}\n"
            f"Stake: £{bet['stake']:.2f}\n"
            f"Edge: {bet['edge'] * 100:.1f}%\n"
            f"Potential profit: £{bet['stake'] * (bet['odds'] - 1):.2f}"
        )
        return self.send(title, message, color=0x3498DB)

    def send_bet_settled(self, bet: Dict) -> bool:
        """Send notification when a bet is settled."""
        won = bet["status"] == "won"
        title = "✅ Bet Won!" if won else "❌ Bet Lost"
        color = 0x2ECC71 if won else 0xE74C3C

        message = (
            f"**{bet['league']}**\n"
            f"{bet['home_team']} vs {bet['away_team']}\n\n"
            f"Selection: {bet['selection'].upper()} @ {bet['odds']:.2f}\n"
            f"Stake: £{bet['stake']:.2f}\n"
            f"Profit: £{bet['profit']:+.2f}"
        )
        return self.send(title, message, color=color)

    def send_daily_summary(self, stats: Dict) -> bool:
        """Send daily performance summary."""
        title = f"📊 Daily Summary - {stats['date']}"

        message = (
            f"**Bets Placed:** {stats['bets_placed']}\n"
            f"**Stake Placed:** £{stats['stake_placed']:.2f}\n\n"
            f"**Bets Settled:** {stats['bets_settled']}\n"
            f"**Wins:** {stats['settled_wins']}\n"
            f"**Win Rate:** {stats['win_rate']:.1f}%\n"
            f"**Profit:** £{stats['settled_profit']:+.2f}"
        )

        color = 0x2ECC71 if stats["settled_profit"] >= 0 else 0xE74C3C
        return self.send(title, message, color=color)


class EmailNotifier:
    """Send notifications via email (SMTP)."""

    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.recipient = os.getenv("NOTIFICATION_EMAIL")

    def is_configured(self) -> bool:
        return all([self.smtp_user, self.smtp_pass, self.recipient])

    def send(self, subject: str, body: str, html: str = None) -> bool:
        """Send an email notification."""
        if not self.is_configured():
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Sports Trading] {subject}"
            msg["From"] = self.smtp_user
            msg["To"] = self.recipient

            msg.attach(MIMEText(body, "plain"))
            if html:
                msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.smtp_user, self.recipient, msg.as_string())

            return True
        except Exception as e:
            print(f"Email notification failed: {e}")
            return False

    def send_weekly_report(self, stats: Dict, bets: List[Dict]) -> bool:
        """Send weekly performance report."""
        subject = f"Weekly Report - £{stats['total_profit']:+.2f}"

        body = f"""
Sports Trading Weekly Report
=============================

Period Summary:
- Total Bets: {stats["total_bets"]}
- Wins: {stats["wins"]} | Losses: {stats["losses"]}
- Win Rate: {stats["win_rate"]:.1f}%
- Total Staked: £{stats["total_staked"]:.2f}
- Total Profit: £{stats["total_profit"]:+.2f}
- Yield: {stats["yield_pct"]:.2f}%

Risk Metrics:
- Max Drawdown: £{stats.get("max_drawdown", 0):.2f}
- Max Win Streak: {stats.get("max_win_streak", 0)}
- Max Loss Streak: {stats.get("max_loss_streak", 0)}

---
Sports Trading Bot
"""

        return self.send(subject, body)


class NotificationManager:
    """Unified notification manager."""

    def __init__(self):
        self.discord = DiscordNotifier()
        self.email = EmailNotifier()

    def notify_bet_placed(self, bet: Dict):
        """Notify all channels about a new bet."""
        if self.discord.is_configured():
            self.discord.send_bet_placed(bet)

    def notify_bet_settled(self, bet: Dict):
        """Notify all channels about a settled bet."""
        if self.discord.is_configured():
            self.discord.send_bet_settled(bet)

    def notify_daily_summary(self, stats: Dict):
        """Send daily summary to all channels."""
        if self.discord.is_configured():
            self.discord.send_daily_summary(stats)

    def notify_weekly_report(self, stats: Dict, bets: List[Dict]):
        """Send weekly report."""
        if self.email.is_configured():
            self.email.send_weekly_report(stats, bets)
        if self.discord.is_configured():
            self.discord.send(
                "📈 Weekly Report",
                f"Bets: {stats['total_bets']} | Profit: £{stats['total_profit']:+.2f} | Yield: {stats['yield_pct']:.2f}%",
                color=0x9B59B6,
            )
