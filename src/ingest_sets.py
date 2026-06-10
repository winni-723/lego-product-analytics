import requests
import json
from datetime import datetime

from config import REBRICKABLE_HEADERS, snowflake_connection

start_url = "https://rebrickable.com/api/v3/lego/sets/"
headers = REBRICKABLE_HEADERS

conn = snowflake_connection()

cursor = conn.cursor()

insert_sql = """
INSERT INTO REBRICKABLE_SETS_JSON
(INGESTED_AT, ENDPOINT, REQUESTS_PARAMS, PAYLOAD)
SELECT
    %s,
    %s,
    PARSE_JSON(%s),
    PARSE_JSON(%s)
"""

url = start_url
page_num = 1

while url:
    print(f"Fetching page {page_num}: {url}")
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    cursor.execute(
        insert_sql,
        (
            datetime.utcnow(),
            "/lego/sets/",
            json.dumps({"page": page_num}),
            json.dumps(data)
        )
    )
    conn.commit()

    url = data.get("next")
    page_num += 1

print("All set pages inserted successfully.")

cursor.close()
conn.close()