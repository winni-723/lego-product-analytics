"""Step 6:  A/B test simulation — content-based vs popularity recommender.

We have no real click logs, so we (1) define a transparent synthetic click
model as the "ground truth", (2) simulate a randomized experiment on top of
it, then (3) analyse it with the same statistics a product analyst would use
on a real test. The point is to demonstrate the full A/B methodology end to end.

Design:
    Control  (A):  show the most-OWNED set in the user's favourite theme
                   ("theme-popularity" baseline).
    Treatment(B):  show the most CONTENT-SIMILAR set (our recommender).
    Click model :  users click in proportion to how relevant the shown set is
                   to their taste (cosine similarity) + a little popularity pull.

Run from the project root:
    .\\venv\\Scripts\\python.exe src\\ab_test_simulation.py
"""
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.preprocessing import normalize
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "lego.db"
MODEL_DIR = ROOT / "models"

# Click-model coefficients (the hidden "ground truth"). Tuned for realistic CTRs.
BASE_CTR = 0.03      # baseline click propensity
W_SIM = 0.18         # how much relevance (similarity) drives clicks
W_POP = 0.04         # small extra pull from sheer popularity
CLIP = (0.01, 0.95)


def load_world():
    """Load the recommender + popularity so we can simulate a 'world'."""
    b = joblib.load(MODEL_DIR / "recommender.joblib")
    nn, pre, meta = b["nn"], b["transformer"], b["meta"]
    meta = meta.reset_index(drop=True)
    X = pre.transform(meta[b["features"]])

    owned = duckdb.connect(str(DB_PATH), read_only=True).execute(
        "SELECT set_num, owned_by FROM raw_brickset"
    ).df()
    meta = meta.merge(owned, on="set_num", how="left")
    meta["owned_by"] = meta["owned_by"].fillna(0)

    # Popularity normalised 0..1 on a log scale (for the click model).
    logpop = np.log1p(meta["owned_by"].to_numpy())
    meta["pop_norm"] = (logpop - logpop.min()) / (logpop.max() - logpop.min() + 1e-9)

    # Most-popular set per theme = the control recommender's lookup table.
    top_by_theme = meta.loc[meta.groupby("theme")["owned_by"].idxmax()]
    theme_to_top = dict(zip(top_by_theme["theme"], top_by_theme.index))

    Xn = normalize(X)        # L2-normalised rows => cosine = dot product
    return meta, Xn, nn, theme_to_top


def simulate(n_users=20000, seed=42):
    meta, Xn, nn, theme_to_top = load_world()
    rng = np.random.default_rng(seed)

    # Each user has a favourite set — popular sets are more common entry points.
    weights = (meta["owned_by"].to_numpy() + 1.0)
    weights = weights / weights.sum()
    user_seed = rng.choice(len(meta), size=n_users, p=weights)

    # Randomly assign each user to A (control) or B (treatment).
    group = rng.integers(0, 2, size=n_users)        # 0 = A, 1 = B

    # B's recommendation: nearest content neighbour (exclude the seed itself).
    _, nbrs = nn.kneighbors(Xn[user_seed], n_neighbors=2)
    content_rec = np.where(nbrs[:, 0] == user_seed, nbrs[:, 1], nbrs[:, 0])

    # A's recommendation: most-popular set in the user's seed theme.
    seed_theme = meta["theme"].to_numpy()[user_seed]
    pop_rec = np.array([theme_to_top[t] for t in seed_theme])

    shown = np.where(group == 1, content_rec, pop_rec)

    # Relevance = cosine(seed, shown); click prob from the hidden model.
    sim = np.asarray(Xn[user_seed].multiply(Xn[shown]).sum(axis=1)).ravel()
    pop_norm = meta["pop_norm"].to_numpy()[shown]
    p_click = np.clip(BASE_CTR + W_SIM * sim + W_POP * pop_norm, *CLIP)
    clicked = rng.random(n_users) < p_click

    return group, clicked


def analyse(group, clicked):
    a, b = group == 0, group == 1
    n_a, n_b = a.sum(), b.sum()
    c_a, c_b = clicked[a].sum(), clicked[b].sum()
    ctr_a, ctr_b = c_a / n_a, c_b / n_b

    # Two-proportion z-test (is B's CTR different from A's?).
    z, p = proportions_ztest([c_b, c_a], [n_b, n_a])

    ci_a = proportion_confint(c_a, n_a, alpha=0.05, method="wilson")
    ci_b = proportion_confint(c_b, n_b, alpha=0.05, method="wilson")

    abs_lift = ctr_b - ctr_a
    rel_lift = abs_lift / ctr_a

    # Power: how many users PER ARM to reliably detect this effect (80% power)?
    eff = proportion_effectsize(ctr_b, ctr_a)
    n_needed = NormalIndPower().solve_power(
        effect_size=eff, alpha=0.05, power=0.80, alternative="two-sided")

    print("================  A/B TEST RESULT  ================")
    print(f"Control  A (popularity): {c_a:>5}/{n_a:<6} clicks  CTR={ctr_a:.2%} "
          f"95% CI [{ci_a[0]:.2%}, {ci_a[1]:.2%}]")
    print(f"Treat    B (content)   : {c_b:>5}/{n_b:<6} clicks  CTR={ctr_b:.2%} "
          f"95% CI [{ci_b[0]:.2%}, {ci_b[1]:.2%}]")
    print(f"\nAbsolute lift : {abs_lift:+.2%}")
    print(f"Relative lift : {rel_lift:+.1%}")
    print(f"z = {z:.2f},  p-value = {p:.2e}")
    verdict = "SIGNIFICANT (reject H0)" if p < 0.05 else "not significant"
    print(f"Decision (alpha=0.05): {verdict}")
    print(f"\nSample size needed per arm to detect this effect at 80% power: "
          f"{int(np.ceil(n_needed)):,}")
    print("===================================================")


if __name__ == "__main__":
    group, clicked = simulate()
    analyse(group, clicked)
