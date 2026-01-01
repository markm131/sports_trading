# src/api/auth.py
"""
Betfair API authentication utilities.

The main scanner uses BetfairClient in scanner.py directly.
This module provides standalone authentication for testing.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_betfair_client():
    """Create and return a logged-in Betfair API client."""
    try:
        from betfairlightweight import APIClient
    except ImportError:
        raise ImportError(
            "betfairlightweight not installed. Run: pip install betfairlightweight"
        )

    username = os.getenv("BETFAIR_USERNAME")
    password = os.getenv("BETFAIR_PASSWORD")
    app_key = os.getenv("BETFAIR_APP_KEY")

    if not all([username, password, app_key]):
        raise ValueError("Missing Betfair credentials in .env file")

    # Get certs directory
    certs_dir = os.getenv("BETFAIR_CERT_PATH")
    if certs_dir:
        certs_dir = str(Path(certs_dir).parent)
    else:
        # Default to project certs folder
        root = Path(__file__).resolve().parents[2]
        certs_dir = str(root / "certs")

    client = APIClient(
        username,
        password,
        app_key=app_key,
        certs=certs_dir,
    )

    return client


def test_connection():
    """Test Betfair API connection."""
    client = get_betfair_client()
    client.login()
    print("Logged in successfully")
    client.logout()
    print("Logged out")


if __name__ == "__main__":
    test_connection()
