# 🧱 LEGO Product Analytics

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-analytics-FFF000?logo=duckdb&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

An end-to-end **product-analytics portfolio project** built on LEGO's full product catalogue
(~27,000 sets across ~494 themes). It pairs business-facing analytics with a complete
machine-learning and experimentation stack, all surfaced through a single, LEGO-branded
Streamlit dashboard.

The project spans the full product-analyst toolkit: **descriptive analytics → predictive
ML → a content recommender → an A/B-test simulation → a local AI assistant.**

> **Data note:** Rebrickable is a *catalogue* source — sets, parts, themes, years, part
> counts. It carries **no sales, price, or rating data**, so there is no native popularity
> signal. Popularity labels are joined in from **Brickset** (owner counts). All catalogue
> trends should be read as *portfolio-strategy* signals (what LEGO chooses to release),
> **not** demand or revenue signals.

---

## ✨ What it demonstrates

| # | Component | Highlights |
|---|-----------|-----------|
| 1 | **Statistical analytics & business insight** | Portfolio volume/complexity trends, theme lifecycle, growth & concentration (Pareto), computed-from-data takeaways |
| 2 | **Streamlit dashboard** | 8-page LEGO-themed app with interactive Plotly charts |
| 3 | **Machine learning** | Theme-survival classifier · set-popularity regressor · content-based recommender |
| 4 | **A/B testing** | Synthetic experiment comparing two recommendation strategies, analysed with proper stats |
| 5 | **AI assistant** | Hybrid local agent: RAG over methodology docs **+** Text-to-SQL over the analytics DB |

### Model results
- **Theme-survival classifier** (RandomForest) — predicts whether a product line stays active or is retired · **ROC-AUC ≈ 0.88**
- **Set-popularity regressor** (RandomForest) — predicts collector ownership from era/parts/minifigs/theme · **R² ≈ 0.82**, MAE ≈ 784 owners
- **Content recommender** (cosine `NearestNeighbors`) — "more sets like this one" · **theme-coherence@10 ≈ 96%**
- **A/B simulation** — content recommender vs. popularity baseline: **+12.6% relative CTR lift** (21.8% → 24.5%, p ≈ 4.5e-6), ~3,727 users/arm for 80% power

---

## 🖥️ Dashboard pages

**Analytics**
1. **Executive Overview** — KPIs + portfolio volume/complexity over time
2. **Theme Lifecycle** — 3D launch-year × lifespan × complexity matrix, new-vs-returning themes, portfolio depth
3. **Growth & Concentration** — year-over-year theme growth, Pareto concentration curve

**Machine learning**
4. **Survival Predictor** — performance, feature importance, at-risk/comeback watchlists, what-if scorer
5. **Popularity Predictor** — drivers, predicted-vs-actual, hidden-hits tables, what-if forecaster
6. **Recommender** — pick a set, get the top-N most similar sets with thumbnails

**Experiment & AI**
7. **A/B Test** — interactive simulation with CTR + confidence intervals, conversion funnel, power curve, ship/no-ship verdict
8. **AI Assistant** — chat that routes between a docs RAG tool and a read-only Text-to-SQL tool over DuckDB

---

## 🛠️ Tech stack

- **App / viz:** Streamlit, Plotly
- **Data:** pandas, DuckDB (local analytics warehouse), Snowflake (raw landing zone)
- **ML / stats:** scikit-learn, statsmodels, scipy, NumPy, joblib
- **AI agent:** LangChain · Ollama (local, free) with `llama3.2` (routing/RAG) + `qwen2.5-coder:1.5b` (Text-to-SQL) + `nomic-embed-text` (embeddings) · Chroma vector store
- **Sources:** Rebrickable API (catalogue) · Brickset API (popularity labels)

---

## 📁 Project structure

```
lego-product-analytics/
├── dashboard/
│   └── app.py                 # The 8-page Streamlit dashboard (LEGO-themed)
├── src/
│   ├── config.py              # Loads secrets from .env (no keys in source)
│   ├── ingest_rebrickable.py  # Pull catalogue (sets, themes) → parquet → DuckDB
│   ├── ingest_brickset.py     # Pull Brickset popularity labels → DuckDB
│   ├── ingest_sets.py / ingest_themes.py
│   ├── train_survival_model.py   # Theme active/retired classifier
│   ├── train_popularity_model.py # Set popularity regressor
│   ├── recommender.py            # Content-based recommender
│   ├── ab_test_simulation.py     # Synthetic A/B experiment + analysis
│   ├── rag_assistant.py          # RAG over docs/knowledge (Chroma + Ollama)
│   └── agent.py                  # Hybrid agent: RAG + Text-to-SQL router
├── docs/knowledge/            # Methodology docs = the RAG corpus
├── data/                      # Aggregated CSVs used by the dashboard
├── models/                    # Trained model artifacts (gitignored — regenerable)
├── .streamlit/config.toml     # Light theme + LEGO primary colour
├── .env.example               # Template for required API keys / Snowflake creds
└── requirements.txt
```

> `models/*.joblib`, `lego.db`, and the Chroma index are **gitignored** because they're
> large and reproducible — see [Rebuilding](#-rebuilding-data--models) below.

---

## 🚀 Getting started

### 1. Clone & create a virtual environment
```bash
git clone <your-repo-url>
cd lego-product-analytics
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure secrets
```bash
cp .env.example .env      # then fill in your own values
```
`.env` holds your Rebrickable / Brickset API keys and Snowflake connection details.
It is **gitignored and must never be committed.**

### 4. Run the dashboard
```bash
streamlit run dashboard/app.py
```

> The **AI Assistant** page also needs a local [Ollama](https://ollama.com) running with
> `llama3.2`, `qwen2.5-coder:1.5b`, and `nomic-embed-text` pulled. The other 7 pages work
> without it.

---

## 🔄 Rebuilding data & models

A fresh clone ships with the aggregated CSVs in `data/` but not the DuckDB file or trained
models (both gitignored). To regenerate them:

```bash
python src/ingest_rebrickable.py     # catalogue → DuckDB (lego.db)
python src/ingest_brickset.py        # popularity labels → DuckDB
python src/train_survival_model.py   # → models/survival_model.joblib
python src/train_popularity_model.py # → models/popularity_model.joblib
python src/recommender.py            # → models/recommender.joblib
python src/rag_assistant.py build    # → Chroma index for the AI assistant
```

---

## ⚠️ Disclaimer

This is an independent portfolio project and is **not affiliated with or endorsed by the
LEGO Group**. "LEGO" is a trademark of the LEGO Group. Catalogue data is sourced from
[Rebrickable](https://rebrickable.com) and [Brickset](https://brickset.com) under their
respective terms; this project visualises it for educational and demonstrative purposes.

## 📄 License

Released under the MIT License.
