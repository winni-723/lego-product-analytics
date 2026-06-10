# A/B Testing & Data Sources

## A/B Test: Content vs Popularity Recommender

**Goal:** decide whether the content-based recommender beats a popularity
baseline at driving clicks.

- **Control (A):** shows the most-owned set in the user's favourite theme
  ("theme-popularity" baseline).
- **Treatment (B):** shows the most content-similar set (cosine similarity over
  theme, size, era, minifigures).
- **Method:** users are randomly split 50/50. A synthetic click model (the
  "ground truth") decides clicks based on how relevant the shown set is.
- **Analysis:** two-proportion z-test on click-through rate (CTR), Wilson
  confidence intervals, absolute and relative lift, and a power analysis for the
  required sample size.
- **Funnel:** a two-stage user funnel tracks impression -> click -> conversion
  for each arm.
- **Typical result:** Treatment B lifts CTR by roughly 12% relative over
  Control A, statistically significant (p well below 0.05).

### Key statistics terms
- **CTR (click-through rate):** clicks divided by impressions.
- **p-value:** probability of seeing a difference this large if the two arms
  were truly equal. Below 0.05 is the usual "significant" threshold.
- **Confidence interval:** a range that likely contains the true CTR.
- **Statistical power:** the chance of detecting a real effect; 80% is the
  industry target and drives the required sample size.
- **Lift:** how much better Treatment is than Control (absolute = B minus A;
  relative = that difference divided by A).

## Data Sources

- **Rebrickable** — the catalogue: set numbers, names, release years, themes,
  and part counts (about 27,000 sets). No sales, price, or rating data.
- **Brickset** — popularity signals: how many users own (owned_by) and want
  (wanted_by) each set, plus ratings and minifigure counts (about 23,000 sets).
- The two sources are joined on set number. The popularity model trains on the
  roughly 20,000 sets present in both.

### Important caveat
All catalogue figures reflect **sets released** (product breadth), not units
sold or revenue. Popularity is measured via Brickset collector ownership, which
is a proxy for demand, not actual sales. Trends should be read as product
**strategy** signals, not direct demand signals.

## Pipeline / architecture
Rebrickable + Brickset APIs -> Parquet files -> DuckDB -> (dashboard, ML models,
A/B testing, and this AI assistant all read from DuckDB).
