"""Step 1 of the pipeline:  Rebrickable API  ->  Parquet  ->  DuckDB.

Pulls the full LEGO catalogue (every set + every theme) from Rebrickable,
saves a local Parquet copy (so we never have to hit the API again), then
loads both into DuckDB as `raw_sets` / `raw_themes`.

Run from the project root:
    .\\venv\\Scripts\\python.exe src\\ingest_rebrickable.py
"""
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

from config import REBRICKABLE_HEADERS

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "lego.db"

BASE = "https://rebrickable.com/api/v3/lego"
PAGE_SIZE = 1000          # Rebrickable's max page size — fewer requests
SLEEP_SECONDS = 1.0       # be polite: stay under the API rate limit


def fetch_all(endpoint: str) -> pd.DataFrame:
    """Page through a Rebrickable list endpoint and return every row.

    Rebrickable paginates: each response has a `results` list and a `next`
    URL (None on the last page). We keep following `next` until it runs out.
    """
    url = f"{BASE}/{endpoint}/?page_size={PAGE_SIZE}"
    rows: list[dict] = []
    page = 1
    while url:
        resp = requests.get(url, headers=REBRICKABLE_HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data["results"])
        print(f"  {endpoint}: page {page:>2} (+{len(data['results'])}, total {len(rows):,})")
        url = data.get("next")
        page += 1
        if url:
            time.sleep(SLEEP_SECONDS)
    return pd.DataFrame(rows)


def main() -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Pull from the API ---------------------------------------------------
    print("Fetching sets from Rebrickable...")
    sets = fetch_all("sets")
    print("Fetching themes from Rebrickable...")
    themes = fetch_all("themes")

    # Paging a live catalogue can return the same row on adjacent pages if the
    # underlying order shifts mid-pull, so de-duplicate on the primary key.
    sets = sets.drop_duplicates(subset="set_num").reset_index(drop=True)
    themes = themes.drop_duplicates(subset="id").reset_index(drop=True)

    # 2) Persist to Parquet (columnar, compressed, fast to re-read) ----------
    sets_path = PARQUET_DIR / "sets.parquet"
    themes_path = PARQUET_DIR / "themes.parquet"
    sets.to_parquet(sets_path, index=False)
    themes.to_parquet(themes_path, index=False)
    print(f"Saved {len(sets):,} sets   -> {sets_path}")
    print(f"Saved {len(themes):,} themes -> {themes_path}")

    # 3) Load Parquet into DuckDB as the local source of truth ---------------
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        f"CREATE OR REPLACE TABLE raw_sets AS "
        f"SELECT * FROM read_parquet('{sets_path.as_posix()}')"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE raw_themes AS "
        f"SELECT * FROM read_parquet('{themes_path.as_posix()}')"
    )
    n_sets = con.execute("SELECT COUNT(*) FROM raw_sets").fetchone()[0]
    n_themes = con.execute("SELECT COUNT(*) FROM raw_themes").fetchone()[0]
    con.close()
    print(f"DuckDB ({DB_PATH.name}) loaded: raw_sets={n_sets:,}, raw_themes={n_themes:,}")


if __name__ == "__main__":
    main()
