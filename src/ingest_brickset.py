"""Step 4a:  pull popularity data from Brickset  ->  Parquet  ->  DuckDB.

Rebrickable gives us the catalogue (features); Brickset gives us the missing
popularity LABEL: how many users own / want each set, plus its rating.

Brickset's getSets endpoint returns up to 500 sets per request and includes
`collections.ownedBy` / `wantedBy` / `rating` — so we page through it by YEAR
(a valid search filter) instead of hitting the API once per set.

Free Brickset keys have a DAILY QUOTA. If we hit it mid-run, the script saves
whatever it has gathered so far and exits cleanly — re-run another day to top up.

Run from the project root:
    .\\venv\\Scripts\\python.exe src\\ingest_brickset.py
"""
import json
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

from config import BRICKSET_API_KEY

ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "data" / "raw" / "brickset_popularity.parquet"
DB_PATH = ROOT / "lego.db"

ENDPOINT = "https://brickset.com/api/v3.asmx/getSets"
PAGE_SIZE = 500
SLEEP_SECONDS = 0.5
MIN_YEAR, MAX_YEAR = 1949, 2027


def get_page(year: int, page: int) -> dict:
    """One getSets call for a given year + page. Returns the parsed JSON."""
    params = {"year": str(year), "pageSize": PAGE_SIZE, "pageNumber": page}
    resp = requests.post(
        ENDPOINT,
        data={"apiKey": BRICKSET_API_KEY, "userHash": "", "params": json.dumps(params)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def extract(s: dict) -> dict:
    """Pull just the fields we need out of one Brickset set record."""
    coll = s.get("collections") or {}
    return {
        "set_num": f"{s.get('number')}-{s.get('numberVariant')}",
        "year": s.get("year"),
        "theme": s.get("theme"),
        "pieces": s.get("pieces"),
        "minifigs": s.get("minifigs"),
        "rating": s.get("rating"),
        "review_count": s.get("reviewCount"),
        "owned_by": coll.get("ownedBy"),
        "wanted_by": coll.get("wantedBy"),
    }


def main() -> None:
    if not BRICKSET_API_KEY:
        raise RuntimeError("BRICKSET_API_KEY not set in .env")

    rows: list[dict] = []
    quota_hit = False

    for year in range(MIN_YEAR, MAX_YEAR + 1):
        page = 1
        while True:
            data = get_page(year, page)
            if data.get("status") != "success":
                # Most likely the daily quota — stop, but keep what we have.
                print(f"  STOP at year {year} p{page}: {data.get('message')}")
                quota_hit = True
                break
            sets = data.get("sets", [])
            rows.extend(extract(s) for s in sets)
            print(f"  year {year} p{page}: +{len(sets)} (total {len(rows):,})")
            if len(sets) < PAGE_SIZE:
                break          # last page for this year
            page += 1
            time.sleep(SLEEP_SECONDS)
        if quota_hit:
            break
        time.sleep(SLEEP_SECONDS)

    if not rows:
        print("No data pulled (quota or auth issue). Nothing saved.")
        return

    df = pd.DataFrame(rows)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)

    con = duckdb.connect(str(DB_PATH))
    con.execute(
        f"CREATE OR REPLACE TABLE raw_brickset AS "
        f"SELECT * FROM read_parquet('{PARQUET_PATH.as_posix()}')"
    )
    n = con.execute("SELECT COUNT(*) FROM raw_brickset").fetchone()[0]
    con.close()

    status = "PARTIAL (quota hit)" if quota_hit else "COMPLETE"
    print(f"\n{status}: saved {len(df):,} sets -> {PARQUET_PATH}")
    print(f"DuckDB raw_brickset = {n:,} rows")


if __name__ == "__main__":
    main()
