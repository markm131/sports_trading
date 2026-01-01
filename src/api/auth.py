from pathlib import Path
import os
from betfairlightweight import APIClient
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("BETFAIR_USERNAME")
password = os.getenv("BETFAIR_PASSWORD")
app_key  = os.getenv("BETFAIR_APP_KEY")

# absolute path to the certs directory
root = Path(__file__).resolve().parents[1]
certs_dir = (root / "certs").as_posix()   # string path, not tuple

# sanity check
print("certs_dir:", certs_dir, "| exists:", (root / "certs").exists())

trading = APIClient(
    username,
    password,
    app_key=app_key,
    certs=certs_dir     # <-- pass the directory path only
)

trading.login()
print("✅ Logged in successfully")
trading.logout()
