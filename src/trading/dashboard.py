# src/trading/dashboard.py
"""
Premium HTML dashboard for sports trading.
Generates a mobile-first, app-like experience.

Usage:
    python -m src.trading.dashboard          # Generate locally
    python -m src.trading.dashboard --upload # Upload to S3
    python -m src.trading.dashboard --preview # Open in browser
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data.db_writer import get_bets_summary, get_current_bankroll, get_db_connection

load_dotenv()

S3_BUCKET = os.getenv("S3_DASHBOARD_BUCKET", "mark-sports-dashboard")
OUTPUT_DIR = Path("/tmp")
DASHBOARD_FILE = OUTPUT_DIR / "dashboard.html"


def get_dashboard_data() -> dict:
    """Fetch all data needed for dashboard."""
    conn = get_db_connection()

    # Current bankroll
    bankroll = get_current_bankroll()

    # Summary stats
    stats = get_bets_summary()

    # Today's bets
    today = datetime.now().strftime("%Y-%m-%d")
    today_bets = pd.read_sql(
        """
        SELECT bet_id, placed_at, league, match_date, home_team, away_team,
               selection, odds, stake, edge, status, profit
        FROM bets
        WHERE match_date = ?
        ORDER BY placed_at DESC
        """,
        conn,
        params=(today,),
    )

    # Recent settled (last 7 days)
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    recent_bets = pd.read_sql(
        """
        SELECT bet_id, placed_at, settled_at, league, match_date, home_team, away_team,
               selection, odds, stake, edge, status, profit, clv
        FROM bets
        WHERE match_date >= ? AND status IN ('won', 'lost')
        ORDER BY settled_at DESC
        LIMIT 20
        """,
        conn,
        params=(week_ago,),
    )

    # All bets for streak calculation
    all_bets = pd.read_sql(
        """
        SELECT status FROM bets
        WHERE status IN ('won', 'lost')
        ORDER BY settled_at DESC
        LIMIT 20
        """,
        conn,
    )

    # Bankroll history for chart
    bankroll_history = pd.read_sql(
        """
        SELECT timestamp, balance_after
        FROM bankroll
        ORDER BY id
        """,
        conn,
    )

    # Daily P&L for week chart
    daily_pnl = pd.read_sql(
        """
        SELECT DATE(settled_at) as date, SUM(profit) as pnl, COUNT(*) as bets
        FROM bets
        WHERE status IN ('won', 'lost') AND settled_at >= ?
        GROUP BY DATE(settled_at)
        ORDER BY date
        """,
        conn,
        params=(week_ago,),
    )

    conn.close()

    # Calculate streak
    streak = 0
    streak_type = None
    for _, row in all_bets.iterrows():
        if streak == 0:
            streak_type = row["status"]
            streak = 1
        elif row["status"] == streak_type:
            streak += 1
        else:
            break

    # Today's stats
    today_settled = today_bets[today_bets["status"].isin(["won", "lost"])]
    today_pnl = float(today_settled["profit"].sum()) if len(today_settled) > 0 else 0.0
    today_wins = (
        int((today_settled["status"] == "won").sum()) if len(today_settled) > 0 else 0
    )
    today_losses = (
        int((today_settled["status"] == "lost").sum()) if len(today_settled) > 0 else 0
    )
    today_pending = int((today_bets["status"] == "pending").sum())

    # CLV stats
    clv_bets = recent_bets[recent_bets["clv"].notna()]
    avg_clv = float(clv_bets["clv"].mean() * 100) if len(clv_bets) > 0 else 0.0

    return {
        "bankroll": bankroll,
        "stats": stats,
        "today_bets": today_bets,
        "recent_bets": recent_bets,
        "bankroll_history": bankroll_history,
        "daily_pnl": daily_pnl,
        "streak": streak,
        "streak_type": streak_type,
        "today_pnl": today_pnl,
        "today_wins": today_wins,
        "today_losses": today_losses,
        "today_pending": today_pending,
        "avg_clv": avg_clv,
        "generated_at": datetime.now(),
    }


def generate_html(data: dict) -> str:
    """Generate the dashboard HTML."""

    bankroll = data["bankroll"]
    stats = data["stats"]
    total_profit = stats.get("total_profit", 0)
    roi = stats.get("roi", 0)
    win_rate = stats.get("win_rate", 0)
    total_bets = stats.get("total_bets", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pending = stats.get("pending", 0)

    streak = data["streak"]
    streak_type = data["streak_type"]
    streak_display = (
        f"{streak}{'W' if streak_type == 'won' else 'L'}" if streak > 0 else "—"
    )

    today_pnl = data["today_pnl"]
    today_wins = data["today_wins"]
    today_losses = data["today_losses"]
    today_pending = data["today_pending"]
    avg_clv = data["avg_clv"]

    generated_at = data["generated_at"]

    # Chart data
    bankroll_history = data["bankroll_history"]
    chart_labels = [row["timestamp"][:10] for _, row in bankroll_history.iterrows()]
    chart_data = [
        round(row["balance_after"], 2) for _, row in bankroll_history.iterrows()
    ]

    # If no history, show starting point
    if not chart_data:
        chart_labels = [generated_at.strftime("%Y-%m-%d")]
        chart_data = [bankroll]

    # Build bet cards
    def build_bet_card(bet, show_date=False):
        status = bet["status"]
        if status == "won":
            status_class = "won"
            status_icon = "✓"
            status_text = "WON"
        elif status == "lost":
            status_class = "lost"
            status_icon = "✗"
            status_text = "LOST"
        else:
            status_class = "pending"
            status_icon = "◯"
            status_text = "PENDING"

        profit = bet.get("profit")
        if pd.notna(profit) and status != "pending":
            profit_str = f"£{profit:+.2f}"
        else:
            profit_str = "—"

        edge = bet.get("edge")
        edge_str = f"{edge * 100:.1f}%" if pd.notna(edge) else "—"

        clv = bet.get("clv")
        clv_html = ""
        if pd.notna(clv) and status != "pending":
            clv_pct = clv * 100
            clv_class = "positive" if clv > 0 else "negative"
            clv_html = f'<span class="clv {clv_class}">CLV {clv_pct:+.1f}%</span>'

        date_html = ""
        if show_date:
            date_html = f'<span class="bet-date">{bet["match_date"]}</span>'

        return f"""
        <div class="bet-card {status_class}">
            <div class="bet-header">
                <div class="bet-meta">
                    <span class="bet-league">{bet["league"]}</span>
                    {date_html}
                </div>
                <span class="bet-status {status_class}">{status_icon} {status_text}</span>
            </div>
            <div class="bet-teams">{bet["home_team"]} vs {bet["away_team"]}</div>
            <div class="bet-info">
                <div class="bet-pick">
                    <span class="pick-label">Pick</span>
                    <span class="pick-value">{bet["selection"].upper()}</span>
                </div>
                <div class="bet-odds">
                    <span class="odds-label">Odds</span>
                    <span class="odds-value">{bet["odds"]:.2f}</span>
                </div>
                <div class="bet-stake">
                    <span class="stake-label">Stake</span>
                    <span class="stake-value">£{bet["stake"]:.2f}</span>
                </div>
                <div class="bet-edge">
                    <span class="edge-label">Edge</span>
                    <span class="edge-value">{edge_str}</span>
                </div>
            </div>
            <div class="bet-result">
                {clv_html}
                <span class="profit {status_class}">{profit_str}</span>
            </div>
        </div>
        """

    # Build today section
    today_bets = data["today_bets"]
    if len(today_bets) > 0:
        today_cards = "".join([build_bet_card(row) for _, row in today_bets.iterrows()])
    else:
        today_cards = """
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <div class="empty-text">No bets today yet</div>
            <div class="empty-subtext">Bets placed 60 mins before kickoff</div>
        </div>
        """

    # Build recent section
    recent_bets = data["recent_bets"]
    if len(recent_bets) > 0:
        recent_cards = "".join(
            [build_bet_card(row, show_date=True) for _, row in recent_bets.iterrows()]
        )
    else:
        recent_cards = """
        <div class="empty-state">
            <div class="empty-icon">🎯</div>
            <div class="empty-text">No settled bets yet</div>
            <div class="empty-subtext">Results appear after matches finish</div>
        </div>
        """

    # Determine hero card color
    hero_class = "negative" if total_profit < 0 else ""
    today_class = "negative" if today_pnl < 0 else "positive" if today_pnl > 0 else ""
    streak_class = (
        "positive"
        if streak_type == "won"
        else "negative"
        if streak_type == "lost"
        else ""
    )
    clv_class = "positive" if avg_clv > 0 else "negative" if avg_clv < 0 else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="theme-color" content="#0a0a12">
    <meta http-equiv="refresh" content="300">
    <title>Sports Trading</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }}
        
        :root {{
            --bg-base: #0a0a12;
            --bg-raised: #12121c;
            --bg-overlay: #1a1a28;
            --bg-highlight: #242436;
            
            --text-primary: #ffffff;
            --text-secondary: #9898a8;
            --text-muted: #5c5c6e;
            
            --accent-green: #22c55e;
            --accent-green-soft: rgba(34, 197, 94, 0.15);
            --accent-red: #ef4444;
            --accent-red-soft: rgba(239, 68, 68, 0.15);
            --accent-amber: #f59e0b;
            --accent-amber-soft: rgba(245, 158, 11, 0.15);
            --accent-blue: #3b82f6;
            
            --border: rgba(255, 255, 255, 0.06);
            --border-strong: rgba(255, 255, 255, 0.1);
            
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
            --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
            
            --safe-top: env(safe-area-inset-top, 0px);
            --safe-bottom: env(safe-area-inset-bottom, 0px);
        }}
        
        html {{
            background: var(--bg-base);
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
            background: var(--bg-base);
            color: var(--text-primary);
            line-height: 1.5;
            min-height: 100vh;
            min-height: 100dvh;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            padding-top: var(--safe-top);
            padding-bottom: var(--safe-bottom);
        }}
        
        .app {{
            max-width: 480px;
            margin: 0 auto;
            padding: 0 16px 120px;
        }}
        
        /* Header */
        .header {{
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 16px 0 20px;
            background: linear-gradient(to bottom, var(--bg-base) 0%, var(--bg-base) 70%, transparent 100%);
        }}
        
        .header-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .brand-icon {{
            font-size: 24px;
        }}
        
        .brand-name {{
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.3px;
        }}
        
        .refresh-btn {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--bg-overlay);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        
        .refresh-btn:active {{
            transform: scale(0.92);
            background: var(--bg-highlight);
        }}
        
        .timestamp {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 6px;
        }}
        
        .live-dot {{
            display: inline-block;
            width: 6px;
            height: 6px;
            background: var(--accent-green);
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}
        
        /* Hero Card */
        .hero {{
            background: linear-gradient(135deg, #1e3a2f 0%, #0f2318 100%);
            border-radius: var(--radius-xl);
            padding: 24px;
            margin-bottom: 16px;
            border: 1px solid rgba(34, 197, 94, 0.2);
            box-shadow: var(--shadow-lg), 0 0 60px rgba(34, 197, 94, 0.1);
            position: relative;
            overflow: hidden;
        }}
        
        .hero.negative {{
            background: linear-gradient(135deg, #3a1e1e 0%, #231010 100%);
            border-color: rgba(239, 68, 68, 0.2);
            box-shadow: var(--shadow-lg), 0 0 60px rgba(239, 68, 68, 0.1);
        }}
        
        .hero::before {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
            pointer-events: none;
        }}
        
        .hero-label {{
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 500;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        
        .hero-value {{
            font-size: 42px;
            font-weight: 800;
            letter-spacing: -2px;
            line-height: 1.1;
        }}
        
        .hero-meta {{
            display: flex;
            gap: 16px;
            margin-top: 12px;
        }}
        
        .hero-stat {{
            display: flex;
            flex-direction: column;
        }}
        
        .hero-stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        
        .hero-stat-value {{
            font-size: 16px;
            font-weight: 600;
        }}
        
        .hero-stat-value.positive {{
            color: var(--accent-green);
        }}
        
        .hero-stat-value.negative {{
            color: var(--accent-red);
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 24px;
        }}
        
        .stat-card {{
            background: var(--bg-raised);
            border-radius: var(--radius-md);
            padding: 16px;
            border: 1px solid var(--border);
        }}
        
        .stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .stat-value {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        
        .stat-value.positive {{
            color: var(--accent-green);
        }}
        
        .stat-value.negative {{
            color: var(--accent-red);
        }}
        
        /* Section */
        .section {{
            margin-bottom: 28px;
        }}
        
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        
        .section-title {{
            font-size: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .section-badge {{
            font-size: 11px;
            color: var(--text-muted);
            background: var(--bg-overlay);
            padding: 4px 10px;
            border-radius: 100px;
        }}
        
        /* Chart */
        .chart-card {{
            background: var(--bg-raised);
            border-radius: var(--radius-lg);
            padding: 16px;
            border: 1px solid var(--border);
        }}
        
        .chart-container {{
            height: 140px;
        }}
        
        /* Today Summary */
        .today-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }}
        
        .today-stat {{
            background: var(--bg-raised);
            border-radius: var(--radius-md);
            padding: 12px;
            text-align: center;
            border: 1px solid var(--border);
        }}
        
        .today-stat .value {{
            font-size: 18px;
            font-weight: 700;
        }}
        
        .today-stat .value.positive {{
            color: var(--accent-green);
        }}
        
        .today-stat .value.negative {{
            color: var(--accent-red);
        }}
        
        .today-stat .label {{
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 2px;
        }}
        
        /* Bet Cards */
        .bet-card {{
            background: var(--bg-raised);
            border-radius: var(--radius-md);
            padding: 14px;
            margin-bottom: 8px;
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-amber);
            transition: transform 0.1s ease;
        }}
        
        .bet-card:active {{
            transform: scale(0.99);
        }}
        
        .bet-card.won {{
            border-left-color: var(--accent-green);
        }}
        
        .bet-card.lost {{
            border-left-color: var(--accent-red);
        }}
        
        .bet-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 6px;
        }}
        
        .bet-meta {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        
        .bet-league {{
            font-size: 11px;
            color: var(--text-muted);
        }}
        
        .bet-date {{
            font-size: 10px;
            color: var(--text-muted);
        }}
        
        .bet-status {{
            font-size: 10px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 100px;
            letter-spacing: 0.3px;
        }}
        
        .bet-status.won {{
            background: var(--accent-green-soft);
            color: var(--accent-green);
        }}
        
        .bet-status.lost {{
            background: var(--accent-red-soft);
            color: var(--accent-red);
        }}
        
        .bet-status.pending {{
            background: var(--accent-amber-soft);
            color: var(--accent-amber);
        }}
        
        .bet-teams {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            line-height: 1.3;
        }}
        
        .bet-info {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
            padding: 10px 0;
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
        }}
        
        .bet-info > div {{
            text-align: center;
        }}
        
        .bet-info span:first-child {{
            display: block;
            font-size: 9px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 2px;
        }}
        
        .bet-info span:last-child {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
        }}
        
        .bet-result {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }}
        
        .clv {{
            font-size: 11px;
            font-weight: 500;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        
        .clv.positive {{
            background: var(--accent-green-soft);
            color: var(--accent-green);
        }}
        
        .clv.negative {{
            background: var(--accent-red-soft);
            color: var(--accent-red);
        }}
        
        .profit {{
            font-size: 18px;
            font-weight: 700;
        }}
        
        .profit.won {{
            color: var(--accent-green);
        }}
        
        .profit.lost {{
            color: var(--accent-red);
        }}
        
        .profit.pending {{
            color: var(--text-muted);
        }}
        
        /* Empty State */
        .empty-state {{
            text-align: center;
            padding: 40px 20px;
            background: var(--bg-raised);
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
        }}
        
        .empty-icon {{
            font-size: 32px;
            margin-bottom: 8px;
            opacity: 0.6;
        }}
        
        .empty-text {{
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
        }}
        
        .empty-subtext {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 32px 16px;
            font-size: 11px;
            color: var(--text-muted);
        }}
        
        .footer-line {{
            margin-top: 4px;
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 0;
            height: 0;
        }}
    </style>
</head>
<body>
    <div class="app">
        <header class="header">
            <div class="header-row">
                <div class="brand">
                    <span class="brand-icon">⚽</span>
                    <span class="brand-name">Sports Trading</span>
                </div>
                <button class="refresh-btn" onclick="location.reload()" aria-label="Refresh">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
                        <path d="M21 3v5h-5"/>
                    </svg>
                </button>
            </div>
            <div class="timestamp">
                <span class="live-dot"></span>
                Updated {generated_at.strftime("%d %b, %H:%M")} · Auto-refresh 5m
            </div>
        </header>
        
        <div class="hero {hero_class}">
            <div class="hero-label">Bankroll</div>
            <div class="hero-value">£{bankroll:,.2f}</div>
            <div class="hero-meta">
                <div class="hero-stat">
                    <span class="hero-stat-label">P&L</span>
                    <span class="hero-stat-value {"positive" if total_profit >= 0 else "negative"}">£{total_profit:+,.2f}</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-label">ROI</span>
                    <span class="hero-stat-value {"positive" if roi >= 0 else "negative"}">{roi:+.1f}%</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-label">CLV</span>
                    <span class="hero-stat-value {clv_class}">{avg_clv:+.1f}%</span>
                </div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value">{win_rate:.0f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Record</div>
                <div class="stat-value">{wins}W-{losses}L</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Bets</div>
                <div class="stat-value">{total_bets}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Streak</div>
                <div class="stat-value {streak_class}">{streak_display}</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">📈 Performance</h2>
            </div>
            <div class="chart-card">
                <div class="chart-container">
                    <canvas id="chart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">📅 Today</h2>
                <span class="section-badge">{len(data["today_bets"])} bets</span>
            </div>
            <div class="today-grid">
                <div class="today-stat">
                    <div class="value {today_class}">£{today_pnl:+.2f}</div>
                    <div class="label">P&L</div>
                </div>
                <div class="today-stat">
                    <div class="value">{today_wins}W {today_losses}L</div>
                    <div class="label">Results</div>
                </div>
                <div class="today-stat">
                    <div class="value">{today_pending}</div>
                    <div class="label">Pending</div>
                </div>
            </div>
            {today_cards}
        </div>
        
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">🕐 Recent</h2>
                <span class="section-badge">7 days</span>
            </div>
            {recent_cards}
        </div>
        
        <footer class="footer">
            <div>Paper Trading · Dixon-Coles · Kelly 10%</div>
            <div class="footer-line">{pending} pending · Updated every 15m</div>
        </footer>
    </div>
    
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 140);
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.25)');
        gradient.addColorStop(1, 'rgba(34, 197, 94, 0)');
        
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {chart_labels},
                datasets: [{{
                    data: {chart_data},
                    borderColor: '#22c55e',
                    backgroundColor: gradient,
                    fill: true,
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHitRadius: 20
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }},
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        backgroundColor: '#1a1a28',
                        titleColor: '#fff',
                        bodyColor: '#9898a8',
                        borderColor: '#242436',
                        borderWidth: 1,
                        padding: 10,
                        cornerRadius: 8,
                        displayColors: false,
                        titleFont: {{ weight: '600' }},
                        callbacks: {{
                            label: ctx => '£' + ctx.parsed.y.toFixed(2)
                        }}
                    }}
                }},
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        display: true,
                        grid: {{
                            color: 'rgba(255,255,255,0.03)',
                            drawBorder: false
                        }},
                        ticks: {{
                            color: '#5c5c6e',
                            font: {{ size: 10 }},
                            padding: 8,
                            callback: v => '£' + v
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    return html


def generate_dashboard(upload: bool = False, preview: bool = False):
    """Generate dashboard and optionally upload/preview."""
    print(f"Generating dashboard at {datetime.now()}")

    try:
        data = get_dashboard_data()
        html = generate_html(data)

        # Write to file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(DASHBOARD_FILE, "w") as f:
            f.write(html)

        print(f"Dashboard saved to {DASHBOARD_FILE}")

        if upload:
            # Upload to S3
            cmd = [
                "aws",
                "s3",
                "cp",
                str(DASHBOARD_FILE),
                f"s3://{S3_BUCKET}/index.html",
                "--content-type",
                "text/html",
                "--cache-control",
                "max-age=60",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                url = f"http://{S3_BUCKET}.s3-website.eu-west-2.amazonaws.com"
                print(f"Uploaded to S3: {url}")
            else:
                print(f"Upload failed: {result.stderr}")
                return False

        if preview:
            # Open in browser
            import webbrowser

            webbrowser.open(f"file://{DASHBOARD_FILE}")

        return True

    except Exception as e:
        print(f"Error generating dashboard: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate sports trading dashboard")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--preview", action="store_true", help="Open in browser")

    args = parser.parse_args()

    success = generate_dashboard(upload=args.upload, preview=args.preview)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
