# src/trading/notifications.py
"""
Enhanced Discord notifications with live dashboard.
"""

import json
import os
from datetime import datetime
from typing import Dict, List
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()


class DiscordDashboard:
    """Rich Discord notifications with embeds."""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def _send(self, payload: dict) -> bool:
        """Send payload to Discord."""
        if not self.webhook_url:
            return False

        try:
            req = Request(
                self.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "User-Agent": "SportsBot/1.0"},
            )
            urlopen(req)
            return True
        except Exception as e:
            print(f"Discord error: {e}")
            return False

    def send_bet_placed(self, bet: Dict) -> bool:
        """Notify when a bet is placed."""
        embed = {
            "title": "🎯 NEW BET PLACED",
            "color": 3447003,
            "fields": [
                {
                    "name": "Match",
                    "value": f"**{bet['home_team']}** vs **{bet['away_team']}**",
                    "inline": False,
                },
                {"name": "League", "value": bet["league"], "inline": True},
                {
                    "name": "Selection",
                    "value": bet["selection"].upper(),
                    "inline": True,
                },
                {"name": "Odds", "value": f"{bet['odds']:.2f}", "inline": True},
                {"name": "Stake", "value": f"£{bet['stake']:.2f}", "inline": True},
                {"name": "Edge", "value": f"{bet['edge'] * 100:.1f}%", "inline": True},
                {
                    "name": "Potential Profit",
                    "value": f"£{bet['stake'] * (bet['odds'] - 1):.2f}",
                    "inline": True,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
        return self._send({"embeds": [embed]})

    def send_bet_settled(self, bet: Dict) -> bool:
        """Notify when a bet settles."""
        won = bet.get("status") == "won"
        profit = bet.get("profit", 0)

        embed = {
            "title": "✅ BET WON!" if won else "❌ BET LOST",
            "color": 3066993 if won else 15158332,
            "fields": [
                {
                    "name": "Match",
                    "value": f"**{bet['home_team']}** vs **{bet['away_team']}**",
                    "inline": False,
                },
                {
                    "name": "Selection",
                    "value": f"{bet['selection'].upper()} @ {bet['odds']:.2f}",
                    "inline": True,
                },
                {
                    "name": "Result",
                    "value": bet.get("actual_result", "N/A"),
                    "inline": True,
                },
                {"name": "P&L", "value": f"**£{profit:+.2f}**", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
        return self._send({"embeds": [embed]})

    def send_daily_summary(self, stats: Dict) -> bool:
        """Send daily P&L summary."""
        profit = stats.get("total_profit", 0)
        color = 3066993 if profit >= 0 else 15158332

        embed = {
            "title": "📊 DAILY SUMMARY",
            "color": color,
            "fields": [
                {
                    "name": "💰 Bankroll",
                    "value": f"**£{stats['bankroll']:.2f}**",
                    "inline": True,
                },
                {
                    "name": "📈 Total P&L",
                    "value": f"**£{stats['total_profit']:+.2f}**",
                    "inline": True,
                },
                {"name": "📊 ROI", "value": f"**{stats['roi']:.1f}%**", "inline": True},
                {
                    "name": "🎯 Total Bets",
                    "value": str(stats.get("total_bets", 0)),
                    "inline": True,
                },
                {"name": "✅ Wins", "value": str(stats.get("wins", 0)), "inline": True},
                {
                    "name": "❌ Losses",
                    "value": str(stats.get("losses", 0)),
                    "inline": True,
                },
                {
                    "name": "📋 Pending",
                    "value": str(stats.get("pending", 0)),
                    "inline": True,
                },
                {
                    "name": "🎲 Win Rate",
                    "value": f"{stats.get('win_rate', 0):.1f}%",
                    "inline": True,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Sports Trading Bot"},
        }
        return self._send({"embeds": [embed]})

    def send_full_dashboard(
        self, stats: Dict, recent_bets: List[Dict], pending: List[Dict]
    ) -> bool:
        """Send comprehensive dashboard update."""
        profit = stats.get("total_profit", 0)
        main_color = 3066993 if profit >= 0 else 15158332

        stats_embed = {
            "title": "📊 LIVE DASHBOARD",
            "color": main_color,
            "fields": [
                {
                    "name": "💰 Bankroll",
                    "value": f"```£{stats['bankroll']:.2f}```",
                    "inline": True,
                },
                {
                    "name": "📈 Total P&L",
                    "value": f"```£{stats['total_profit']:+.2f}```",
                    "inline": True,
                },
                {
                    "name": "📊 ROI",
                    "value": f"```{stats['roi']:.1f}%```",
                    "inline": True,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        if recent_bets:
            results_text = ""
            for bet in recent_bets[:10]:
                emoji = (
                    "✅"
                    if bet.get("status") == "won"
                    else "❌"
                    if bet.get("status") == "lost"
                    else "⏳"
                )
                profit_str = (
                    f"£{bet.get('profit', 0):+.2f}" if bet.get("profit") else "pending"
                )
                results_text += f"{emoji} {bet['home_team'][:10]} vs {bet['away_team'][:10]} | {bet['selection'].upper()} @ {bet['odds']:.2f} | {profit_str}\n"
            results_embed = {
                "title": "📋 RECENT BETS",
                "description": f"```{results_text}```",
                "color": 9807270,
            }
        else:
            results_embed = {
                "title": "📋 RECENT BETS",
                "description": "No bets yet",
                "color": 9807270,
            }

        if pending:
            pending_text = ""
            for bet in pending[:5]:
                pending_text += f"⏳ {bet['home_team'][:10]} vs {bet['away_team'][:10]} | {bet['selection'].upper()} @ {bet['odds']:.2f} | £{bet['stake']:.2f}\n"
            pending_embed = {
                "title": "⏳ PENDING BETS",
                "description": f"```{pending_text}```",
                "color": 16776960,
            }
        else:
            pending_embed = {
                "title": "⏳ PENDING BETS",
                "description": "No pending bets",
                "color": 16776960,
            }

        return self._send({"embeds": [stats_embed, results_embed, pending_embed]})


# Global instance
dashboard = DiscordDashboard()


def notify_bet_placed(bet: Dict):
    """Convenience function."""
    dashboard.send_bet_placed(bet)


def notify_bet_settled(bet: Dict):
    """Convenience function."""
    dashboard.send_bet_settled(bet)


def send_dashboard_update():
    """Send full dashboard - call from cron or manually."""
    from src.data.db_writer import (
        get_bets_summary,
        get_current_bankroll,
        get_recent_bets,
    )

    stats = get_bets_summary()
    stats["bankroll"] = get_current_bankroll()
    stats["total_profit"] = stats["bankroll"] - 1000
    stats["roi"] = (stats["total_profit"] / 1000) * 100

    recent = get_recent_bets(10).to_dict("records")
    pending_df = get_recent_bets(20)
    pending = pending_df[pending_df["status"] == "pending"].to_dict("records")

    dashboard.send_full_dashboard(stats, recent, pending)


if __name__ == "__main__":
    send_dashboard_update()
