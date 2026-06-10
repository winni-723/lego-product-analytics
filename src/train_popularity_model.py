"""Step 4b:  Popularity Prediction model.

Business question:  before a set ships, how popular will it be — i.e. how many
collectors will end up owning it?

Label:   owned_by  (Brickset collectors who own the set), log-transformed
         because popularity is heavily right-skewed (a few mega-hits, a long
         tail of niche sets).

Features (only what's known at DESIGN time — anything post-release would leak):
    year, num_parts, minifigs, theme

DELIBERATELY EXCLUDED (post-release outcomes => data leakage):
    rating, review_count, wanted_by

Pipeline:
    DuckDB raw_sets  JOIN  raw_brickset  (on set_num)
      -> features X + log(owned_by) target
      -> Ridge baseline + Random Forest regressor
      -> report R^2 (log scale) and MAE (real owners)
      -> what drives popularity (feature importance)
      -> save model + metrics + scored table

Run from the project root:
    .\\venv\\Scripts\\python.exe src\\train_popularity_model.py
"""
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "lego.db"
MODEL_DIR = ROOT / "models"

NUMERIC = ["year", "num_parts", "minifigs"]
CATEGORICAL = ["theme"]
FEATURES = NUMERIC + CATEGORICAL


def load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Join catalogue (Rebrickable) to popularity (Brickset)."""
    df = con.execute(
        """
        SELECT s.set_num, s.year, s.num_parts,
               b.minifigs, b.theme, b.owned_by
        FROM raw_sets s
        JOIN raw_brickset b ON s.set_num = b.set_num
        WHERE b.owned_by IS NOT NULL AND b.owned_by > 0
          AND s.year BETWEEN 1949 AND 2027
        """
    ).df()
    df["minifigs"] = df["minifigs"].fillna(0)
    df["num_parts"] = df["num_parts"].fillna(0)
    df["theme"] = df["theme"].fillna("Unknown")
    return df


def build_pipeline(model) -> Pipeline:
    """One-hot rare-grouped themes + scaled numerics, then the estimator."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name, pipe, X_tr, X_te, y_tr, y_te):
    """Fit and report R^2 (log scale) + MAE in real owner counts."""
    pipe.fit(X_tr, y_tr)
    pred_log = pipe.predict(X_te)
    r2 = r2_score(y_te, pred_log)
    cv_r2 = cross_val_score(pipe, pd.concat([X_tr, X_te]),
                            pd.concat([y_tr, y_te]), cv=5, scoring="r2")
    # Back-transform to real owner counts for an interpretable error.
    mae_real = mean_absolute_error(np.expm1(y_te), np.expm1(pred_log))
    print(f"\n===== {name} =====")
    print(f"R^2 (log scale)   : {r2:.3f}")
    print(f"5-fold CV R^2     : {cv_r2.mean():.3f} (+/- {cv_r2.std():.3f})")
    print(f"MAE (real owners) : {mae_real:,.0f}")
    return pipe, r2, {"r2": round(float(r2), 3),
                      "cv_r2_mean": round(float(cv_r2.mean()), 3),
                      "cv_r2_std": round(float(cv_r2.std()), 3),
                      "mae_real_owners": int(mae_real)}


def feature_importance(pipe) -> pd.Series:
    """Aggregate one-hot theme importances back to one 'theme' number."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    grouped = {}
    for n, v in imp.items():
        key = "theme" if n.startswith("cat__") else n.split("__", 1)[1]
        grouped[key] = grouped.get(key, 0.0) + v
    return pd.Series(grouped).sort_values(ascending=False)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    df = load_data(con)
    print(f"Training rows: {len(df):,} "
          f"(owned_by median={int(df['owned_by'].median())}, max={int(df['owned_by'].max())})")

    X = df[FEATURES]
    y = np.log1p(df["owned_by"])           # log target (handles skew)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    ridge, r2_ridge, m_ridge = evaluate(
        "Ridge (linear baseline)", build_pipeline(Ridge(alpha=1.0)),
        X_tr, X_te, y_tr, y_te)
    rf_pipe, r2_rf, m_rf = evaluate(
        "Random Forest", build_pipeline(
            RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        X_tr, X_te, y_tr, y_te)

    imp = feature_importance(rf_pipe)
    print("\n===== What drives popularity? (RF importance) =====")
    print(imp.to_string())

    # Pick winner, refit on all data, score every joined set.
    if r2_rf >= r2_ridge:
        best_name, best_pipe, best_metrics = "random_forest", rf_pipe, m_rf
    else:
        best_name, best_pipe, best_metrics = "ridge", ridge, m_ridge
    best_pipe.fit(X, y)
    df["pred_owned_by"] = np.expm1(best_pipe.predict(X)).round().astype(int)

    scored = df[["set_num", "theme", "year", "num_parts", "minifigs",
                 "owned_by", "pred_owned_by"]]
    con.register("pop_tmp", scored)
    con.execute("CREATE OR REPLACE TABLE popularity_scored AS SELECT * FROM pop_tmp")
    con.unregister("pop_tmp")

    joblib.dump({"model": best_pipe, "features": FEATURES, "kind": best_name},
                MODEL_DIR / "popularity_model.joblib")
    best_metrics.update({
        "best_model": best_name,
        "n_rows": int(len(df)),
        "owned_by_median": int(df["owned_by"].median()),
        "feature_importances": imp.round(4).to_dict(),
    })
    (MODEL_DIR / "popularity_metrics.json").write_text(json.dumps(best_metrics, indent=2))
    print(f"\nSaved best model ({best_name}, R^2={max(r2_rf, r2_ridge):.3f})")
    con.close()


if __name__ == "__main__":
    main()
