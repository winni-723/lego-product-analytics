# Machine Learning Models

This document explains the two ML models in the project, so the assistant can
answer questions about how they work.

## 1. Theme Survival Predictor (classification)

**Business question:** Given a LEGO product line's structural characteristics,
is it still Active or has it been Retired?

- **Unit of analysis:** top-level themes (product lines), 149 of them, with sets
  rolled up over the whole sub-theme tree.
- **Label:** Active (released a set in the last year of data) vs Retired.
- **Features:** first_year, n_sets, avg_parts, median_parts, max_parts,
  num_sub_themes.
- **Deliberately excluded to avoid data leakage:** last_year and lifespan, which
  are computed from the same recency that defines the label.
- **Model:** Random Forest classifier.
- **Performance:** ROC-AUC about 0.88 on a held-out test set.
- **Key driver:** scale (n_sets) is the strongest predictor — product lines that
  shipped many sets and have flagship-scale sets are more likely to survive.
- **Scoring:** themes are scored with out-of-fold predictions so the at-risk
  watchlist is honest (not over-optimistic in-sample numbers).

## 2. Set Popularity Predictor (regression)

**Business question:** Before a set ships, how many collectors will end up
owning it?

- **Label:** log(owned_by) from Brickset. Log-transformed because popularity is
  heavily right-skewed (a few mega-hits, a long tail of niche sets).
- **Features known at design time:** year, num_parts, minifigs, theme.
- **Deliberately excluded as post-release leakage:** rating, review_count,
  wanted_by — these only exist after a set is released.
- **Model:** Random Forest regressor.
- **Performance:** R-squared about 0.82, mean absolute error about 784 owners.
- **Key drivers (in order):** theme, number of minifigures, part count, year.
  Popular licensed themes plus more minifigures plus larger size drive ownership.

## Shared lesson: data leakage

Both models illustrate avoiding data leakage — never feeding the model a feature
that encodes the answer or that is only knowable after the outcome. This is what
keeps the reported metrics honest and the models useful for real prediction.
