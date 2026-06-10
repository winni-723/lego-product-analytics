"""Step 2:  Theme Survival Prediction model.

Business question:  given a LEGO theme's structural characteristics, can we
predict whether it is still ACTIVE or has been RETIRED?

Pipeline:
    DuckDB raw_sets/raw_themes
      -> build a theme-level feature table  (persisted as `theme_features`)
      -> label each theme Active / Retired
      -> train + evaluate two classifiers (Logistic Regression, Random Forest)
      -> report metrics + which factors drive survival
      -> save the trained model to models/

Run from the project root:
    .\\venv\\Scripts\\python.exe src\\train_survival_model.py
"""
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "lego.db"
MODEL_DIR = ROOT / "models"

# A theme counts as ACTIVE if it released a set within this many years of the
# most recent year in the (cleaned) data.
ACTIVE_WINDOW_YEARS = 1
# Only trust years in this range; everything else is dirty data (e.g. 20276).
MIN_YEAR, MAX_YEAR = 1949, 2027

# Features the model is ALLOWED to use. Note what is deliberately EXCLUDED:
#   last_year / lifespan  -> these are computed from the same fact (recent
#   activity) that defines the label, so feeding them in would be DATA LEAKAGE:
#   the model would "cheat" and score ~100% while learning nothing useful.
# We model TOP-LEVEL themes only (real "product lines"), each rolled up over
# its whole sub-tree. So `is_top_level` is constant and not a feature.
FEATURE_COLS = [
    "first_year",      # when the line launched (older lines retire more)
    "n_sets",          # how many sets it ever shipped (investment / scale)
    "avg_parts",       # typical complexity
    "median_parts",    # complexity, robust to outliers
    "max_parts",       # does it have flagship-scale sets?
    "num_sub_themes",  # breadth of the theme's sub-tree
]


def build_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Aggregate raw sets/themes into one row per theme (tree-rolled), and label it.

    LEGO themes form a tree (parent -> sub-themes), and Rebrickable attaches
    sets to the *leaf* sub-themes. So we roll sets UP the tree: a theme's
    features include every set from itself and all its descendants. Otherwise a
    big parent like "Star Wars" looks tiny (few sets attached directly to it).
    """
    # A recursive CTE that maps every theme (root) to all of its descendants
    # (including itself). parent_id is a float in raw data, so we cast it.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE theme_tree AS
        WITH RECURSIVE descendants AS (
            SELECT id AS root_id, id AS node_id FROM raw_themes
            UNION ALL
            SELECT d.root_id, t.id
            FROM descendants d
            JOIN raw_themes t ON CAST(t.parent_id AS BIGINT) = d.node_id
        )
        SELECT * FROM descendants
        """
    )

    # Tree-rolled set aggregates: every set under a theme's whole sub-tree.
    sets_agg = con.execute(
        f"""
        SELECT tt.root_id          AS theme_id,
               COUNT(*)            AS n_sets,
               MIN(s.year)         AS first_year,
               MAX(s.year)         AS last_year,
               AVG(s.num_parts)    AS avg_parts,
               MEDIAN(s.num_parts) AS median_parts,
               MAX(s.num_parts)    AS max_parts
        FROM theme_tree tt
        JOIN raw_sets s ON s.theme_id = tt.node_id
                       AND s.year BETWEEN {MIN_YEAR} AND {MAX_YEAR}
        GROUP BY tt.root_id
        """
    ).df()

    # Sub-theme count = descendants minus the theme itself (whole tree).
    sub_counts = con.execute(
        "SELECT root_id AS theme_id, COUNT(*) - 1 AS num_sub_themes "
        "FROM theme_tree GROUP BY root_id"
    ).df()

    themes = con.execute("SELECT id, parent_id, name FROM raw_themes").df()
    themes["is_top_level"] = themes["parent_id"].isna().astype(int)

    # Join facts + metadata, then keep TOP-LEVEL themes only (one row per
    # product line, rolled up over its whole sub-tree).
    df = (
        sets_agg
        .merge(sub_counts, on="theme_id", how="left")
        .merge(themes[["id", "name", "is_top_level"]],
               left_on="theme_id", right_on="id", how="inner")
    )
    df = df[df["is_top_level"] == 1].reset_index(drop=True)

    # Label: Active if it shipped a set within ACTIVE_WINDOW_YEARS of the latest year.
    latest_year = int(df["last_year"].max())
    df["is_active"] = (df["last_year"] >= latest_year - ACTIVE_WINDOW_YEARS).astype(int)

    # Persist the feature table back to DuckDB so it's reusable + inspectable.
    con.register("theme_features_tmp", df)
    con.execute("CREATE OR REPLACE TABLE theme_features AS SELECT * FROM theme_features_tmp")
    con.unregister("theme_features_tmp")

    print(f"Built theme_features: {len(df)} themes "
          f"(latest year={latest_year}, "
          f"active={int(df['is_active'].sum())}, retired={int((1-df['is_active']).sum())})")
    return df


def evaluate(name, model, X_train, X_test, y_train, y_test):
    """Fit, predict, and print a metrics report for one model.

    Returns the fitted model, its test ROC-AUC, and a metrics dict (so the
    dashboard can display the same numbers without re-training).
    """
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    cv_auc = cross_val_score(
        model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]),
        cv=5, scoring="roc_auc",
    )
    report = classification_report(
        y_test, preds, target_names=["Retired", "Active"], output_dict=True
    )
    print(f"\n===== {name} =====")
    print(f"Test ROC-AUC      : {auc:.3f}")
    print(f"5-fold CV ROC-AUC : {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")
    print("Confusion matrix [rows=actual, cols=pred] (0=Retired,1=Active):")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["Retired", "Active"]))
    metrics = {
        "test_roc_auc": round(float(auc), 3),
        "cv_roc_auc_mean": round(float(cv_auc.mean()), 3),
        "cv_roc_auc_std": round(float(cv_auc.std()), 3),
        "accuracy": round(float(report["accuracy"]), 3),
        "active_recall": round(float(report["Active"]["recall"]), 3),
        "active_precision": round(float(report["Active"]["precision"]), 3),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return model, auc, metrics


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    df = build_features(con)

    X = df[FEATURE_COLS].fillna(0)
    y = df["is_active"]

    # Stratified split keeps the Active/Retired ratio the same in train & test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Model 1: Logistic Regression — interpretable baseline (needs scaling).
    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logreg, auc_lr, metrics_lr = evaluate("Logistic Regression", logreg,
                                          X_train, X_test, y_train, y_test)

    # Model 2: Random Forest — stronger, captures non-linear interactions.
    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced",
    )
    rf, auc_rf, metrics_rf = evaluate("Random Forest", rf,
                                      X_train, X_test, y_train, y_test)

    # Which factors drive survival? (Random Forest importances.)
    importances = (
        pd.Series(rf.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )
    print("\n===== What drives theme survival? (RF feature importance) =====")
    print(importances.to_string())

    # Logistic Regression coefficients (direction of effect).
    coefs = pd.Series(
        logreg.named_steps["clf"].coef_[0], index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\n===== Direction of effect (LogReg coefficients, + => more likely Active) =====")
    print(coefs.to_string())

    # Pick the better model.
    if auc_rf >= auc_lr:
        best_name, best_model, best_metrics = "random_forest", rf, metrics_rf
    else:
        best_name, best_model, best_metrics = "logreg", logreg, metrics_lr

    # Score every theme with OUT-OF-FOLD probabilities: each theme is scored by
    # a model that did NOT see it during training (5-fold). This gives honest,
    # non-overconfident scores for the dashboard watchlists — unlike in-sample
    # predictions, which would be optimistic because the model memorised them.
    df["pred_active_proba"] = cross_val_predict(
        best_model, X, y, cv=5, method="predict_proba"
    )[:, 1]

    # Separately, fit the winner on ALL data — this is the model we SAVE, used
    # by the interactive what-if predictor (best use of all available data).
    best_model.fit(X, y)

    # Persist the scored feature table back to DuckDB for the dashboard.
    scored = df[["theme_id", "name", "is_active", "pred_active_proba"] + FEATURE_COLS]
    con.register("scored_tmp", scored)
    con.execute("CREATE OR REPLACE TABLE theme_features AS SELECT * FROM scored_tmp")
    con.unregister("scored_tmp")

    # Save the model.
    out = MODEL_DIR / "survival_model.joblib"
    joblib.dump({"model": best_model, "features": FEATURE_COLS, "kind": best_name}, out)

    # Save metrics + feature importances as JSON for the dashboard to read.
    best_metrics.update({
        "best_model": best_name,
        "latest_year": int(df["last_year"].max()),
        "n_themes": int(len(df)),
        "n_active": int(df["is_active"].sum()),
        "n_retired": int((1 - df["is_active"]).sum()),
        "feature_importances": importances.round(4).to_dict(),
    })
    (MODEL_DIR / "survival_metrics.json").write_text(json.dumps(best_metrics, indent=2))
    print(f"\nSaved best model ({best_name}, AUC={max(auc_rf, auc_lr):.3f}) -> {out}")
    print(f"Saved metrics -> {MODEL_DIR / 'survival_metrics.json'}")
    con.close()


if __name__ == "__main__":
    main()
