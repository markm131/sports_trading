import os
from pathlib import Path
from dotenv import load_dotenv
from betfairlightweight import APIClient, filters

# --- setup ---
load_dotenv()
USERNAME = os.getenv("BETFAIR_USERNAME")
PASSWORD = os.getenv("BETFAIR_PASSWORD")
APP_KEY  = os.getenv("BETFAIR_APP_KEY")

root = Path(__file__).resolve().parents[1]
CERTS_DIR = (root / "certs").as_posix()   # pass the folder path

trading = APIClient(USERNAME, PASSWORD, app_key=APP_KEY, certs=CERTS_DIR)

# --- login ---
trading.login()
print("✅ Logged in; pulling markets...")

# --- find today's football Match Odds markets ---
filt = filters.market_filter(
    event_type_ids=["1"],                # 1 = Soccer
    market_type_codes=["MATCH_ODDS"],    # main 1X2 market
)

catalogue = trading.betting.list_market_catalogue(
    filter=filt,
    market_projection=["EVENT","MARKET_START_TIME","RUNNER_METADATA"],
    sort="FIRST_TO_START",
    max_results=5,                       # keep small for a first test
)

for m in catalogue:
    print(f"\n{m.event.venue or m.event.name} -- {m.market_start_time}  ({m.market_id})")
    for r in m.runners:
        print("  ", r.selection_id, r.runner_name)

# --- get prices for the first market ---
if catalogue:
    market_id = catalogue[0].market_id
    books = trading.betting.list_market_book(
        market_ids=[market_id],
        price_projection=filters.price_projection(price_data=["EX_BEST_OFFERS","EX_TRADED"])
    )
    book = books[0]
    print(f"\nPrices for {market_id}:")
    for r in book.runners:
        best_b = r.ex.available_to_back[0].price if r.ex.available_to_back else None
        best_l = r.ex.available_to_lay[0].price  if r.ex.available_to_lay  else None
        print(f"  {r.selection_id}: back {best_b} | lay {best_l}")

# --- logout ---
trading.logout()
print("\n✅ Done")
