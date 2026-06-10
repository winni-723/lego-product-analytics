"""Central configuration: loads secrets from the environment (.env).

No credentials live in source code. Copy `.env.example` to `.env` and fill in
your values. `.env` is gitignored and must never be committed.
"""
import os

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill in your values."
        )
    return value


# Rebrickable
REBRICKABLE_API_KEY = _require("REBRICKABLE_API_KEY")
REBRICKABLE_HEADERS = {"Authorization": f"key {REBRICKABLE_API_KEY}"}

# Brickset (popularity labels: rating, ownedBy/wantedBy). Optional until the
# popularity-model step, so we don't hard-require it.
BRICKSET_API_KEY = os.getenv("BRICKSET_API_KEY")


def snowflake_connection():
    """Open a Snowflake connection using credentials from the environment."""
    import snowflake.connector

    return snowflake.connector.connect(
        user=_require("SNOWFLAKE_USER"),
        password=_require("SNOWFLAKE_PASSWORD"),
        account=_require("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "lego_warehouse"),
        database=os.getenv("SNOWFLAKE_DATABASE", "lego_analytics"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "RAW"),
    )
