"""Step 5:  Content-based recommendation engine.

"If you like set X, you'll probably like these."  We have no user-behaviour
data (no purchases/ratings per user), so collaborative filtering is out.
Instead we recommend by CONTENT similarity: turn each set into a feature
vector (theme, size, era, minifigs) and find its nearest neighbours by
cosine similarity.

Scale note: a full 26k x 26k similarity matrix would be ~2.8 GB. Instead we
fit a NearestNeighbors index and query one set against all others on demand.

Run from the project root (builds + prints demo recommendations):
    .\\venv\\Scripts\\python.exe src\\recommender.py
"""
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "lego.db"
MODEL_DIR = ROOT / "models"

NUMERIC = ["year", "log_parts", "minifigs"]
CATEGORICAL = ["theme"]


def load_sets(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per set with the attributes we'll compare on."""
    df = con.execute(
        """
        SELECT s.set_num, s.name, t.name AS theme, s.year, s.num_parts,
               COALESCE(b.minifigs, 0) AS minifigs
        FROM raw_sets s
        LEFT JOIN raw_themes  t ON s.theme_id = t.id
        LEFT JOIN raw_brickset b ON s.set_num = b.set_num
        WHERE s.year BETWEEN 1949 AND 2027
        """
    ).df()
    df["theme"] = df["theme"].fillna("Unknown")
    df["minifigs"] = df["minifigs"].fillna(0)
    df["log_parts"] = np.log1p(df["num_parts"].fillna(0))  # tame the size skew
    return df.reset_index(drop=True)


def build_matrix(df: pd.DataFrame):
    """Encode sets into a numeric feature matrix (scaled numerics + one-hot theme)."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    X = pre.fit_transform(df[NUMERIC + CATEGORICAL])
    return pre, X


def recommend(set_num, df, nn, X, n=10) -> pd.DataFrame:
    """Return the n most content-similar sets to `set_num`."""
    idx_map = {sn: i for i, sn in enumerate(df["set_num"])}
    if set_num not in idx_map:
        raise KeyError(f"{set_num} not found")
    i = idx_map[set_num]
    # ask for n+1 because the set itself is always its own closest neighbour
    dist, nbrs = nn.kneighbors(X[i], n_neighbors=n + 1)
    out = df.iloc[nbrs[0]].copy()
    out["similarity"] = 1 - dist[0]          # cosine distance -> similarity
    return out[out["set_num"] != set_num].head(n)[
        ["set_num", "name", "theme", "year", "num_parts", "minifigs", "similarity"]
    ]


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    df = load_sets(con)
    pre, X = build_matrix(df)
    print(f"Built feature matrix: {X.shape[0]:,} sets x {X.shape[1]} features")

    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)

    joblib.dump(
        {"nn": nn, "transformer": pre, "meta": df, "features": NUMERIC + CATEGORICAL},
        MODEL_DIR / "recommender.joblib",
    )
    print(f"Saved recommender -> {MODEL_DIR / 'recommender.joblib'}")

    # ── Sanity demos ────────────────────────────────────────────────────────
    demos = df[df["name"].str.contains("Millennium Falcon", case=False, na=False)]
    seed = demos["set_num"].iloc[0] if len(demos) else df["set_num"].iloc[0]
    print(f"\nExample: sets similar to {seed} "
          f"({df.loc[df.set_num == seed, 'name'].iloc[0]})")
    recs = recommend(seed, df, nn, X, n=8)
    print(recs.to_string(index=False))

    # Offline quality proxy: how theme-coherent are recommendations?
    sample = df.sample(min(300, len(df)), random_state=42)["set_num"]
    coh = []
    for sn in sample:
        r = recommend(sn, df, nn, X, n=10)
        seed_theme = df.loc[df.set_num == sn, "theme"].iloc[0]
        coh.append((r["theme"] == seed_theme).mean())
    print(f"\nMean theme-coherence@10 over 300 sampled sets: {np.mean(coh):.0%}")
    con.close()


if __name__ == "__main__":
    main()
