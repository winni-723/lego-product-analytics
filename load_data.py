import duckdb
import pandas as pd

con = duckdb.connect("lego.db")

# con.execute("""
# CREATE OR REPLACE TABLE dim_sets AS
# SELECT *
# FROM read_csv_auto('data/lego_project_DIM_SETS.csv')
# """)

# con.execute("""
# CREATE OR REPLACE TABLE dim_themes AS
# SELECT *
# FROM read_csv_auto('data/lego_project_DIM_THEMES.csv')""")

print("Data loaded successfully into the database.")

tables = [
    "lego_project_DIM_SETS",
    "lego_project_DIM_THEMES",
    "lego_project_THEME_LIFECYCLE",
    "lego_project_THEME_YEARLY_GROWTH",
    "lego_project_COMPLEXITY_TREND",
    "lego_project_NEW_VS_RETURNING_THEMES",
    "lego_project_PORTFOLIO_CONCENTRATION",
    "lego_project_SET_SIZE_SEGMENT",
    "lego_project_THEME_DEPTH",
    "lego_project_THEME_PORTFOLIO",
    "lego_project_THEME_SIZE_MIX",
    "lego_project_YEARLY_PORTFOLIO"
]

for table in tables:
    con.execute(f"""
        CREATE OR REPLACE TABLE {table.lower()} AS
        SELECT *
        FROM read_csv_auto('data/{table}.csv')
    """)