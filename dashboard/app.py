import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.preprocessing import normalize
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)

st.set_page_config(
    page_title="LEGO Product Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Anchor paths to the project root so the app runs from any directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "lego.db"


def _reindex_years(df, count_cols):
    """Fill missing years so trend lines have no x-gaps.

    Count columns are filled with 0 (a year with no releases really is 0),
    but rate/average columns are left as NaN — filling those with 0 would
    draw a misleading crash-to-zero dip instead of a gap.
    """
    df = (
        df.set_index("RELEASE_YEAR")
        .reindex(range(int(df["RELEASE_YEAR"].min()), int(df["RELEASE_YEAR"].max()) + 1))
        .reset_index()
    )
    df[count_cols] = df[count_cols].fillna(0)
    return df


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    yearly        = pd.read_csv(DATA_DIR / "lego_project_YEARLY_PORTFOLIO.csv")
    yearly        = _reindex_years(yearly, count_cols=["TOTAL_SETS"])
    complexity    = pd.read_csv(DATA_DIR / "lego_project_COMPLEXITY_TREND.csv")
    complexity    = _reindex_years(complexity, count_cols=[])
    size_seg      = pd.read_csv(DATA_DIR / "lego_project_SET_SIZE_SEGMENT.csv")
    new_vs_ret    = pd.read_csv(DATA_DIR / "lego_project_NEW_VS_RETURNING_THEMES.csv")
    new_vs_ret    = _reindex_years(new_vs_ret, count_cols=["NEW_THEMES", "RETURNING_THEMES"])
    lifecycle     = pd.read_csv(DATA_DIR / "lego_project_THEME_LIFECYCLE.csv")
    theme_depth   = pd.read_csv(DATA_DIR / "lego_project_THEME_DEPTH.csv")
    yoy_growth    = pd.read_csv(DATA_DIR / "lego_project_THEME_YEARLY_GROWTH.csv")
    concentration = pd.read_csv(DATA_DIR / "lego_project_PORTFOLIO_CONCENTRATION.csv")
    size_mix      = pd.read_csv(DATA_DIR / "lego_project_THEME_SIZE_MIX.csv")
    return (
        yearly, complexity, size_seg, new_vs_ret,
        lifecycle, theme_depth, yoy_growth, concentration, size_mix,
    )

(
    yearly, complexity, size_seg, new_vs_ret,
    lifecycle, theme_depth, yoy_growth, concentration, size_mix,
) = load_all()


# ── ML artifacts (theme survival model) ────────────────────────────────────────
@st.cache_data
def load_survival_data():
    """Metrics JSON + the scored theme_features table from DuckDB."""
    metrics = json.loads((MODEL_DIR / "survival_metrics.json").read_text())
    feats = duckdb.connect(str(DB_PATH), read_only=True).execute(
        "SELECT * FROM theme_features"
    ).df()
    return metrics, feats


@st.cache_resource
def load_survival_model():
    """The fitted sklearn model (cached as a resource, not data)."""
    return joblib.load(MODEL_DIR / "survival_model.joblib")


@st.cache_data
def load_popularity_data():
    """Metrics JSON + scored sets (joined to set names) for the popularity model."""
    metrics = json.loads((MODEL_DIR / "popularity_metrics.json").read_text())
    scored = duckdb.connect(str(DB_PATH), read_only=True).execute(
        """
        SELECT p.*, s.name
        FROM popularity_scored p
        LEFT JOIN raw_sets s ON p.set_num = s.set_num
        """
    ).df()
    return metrics, scored


@st.cache_resource
def load_popularity_model():
    return joblib.load(MODEL_DIR / "popularity_model.joblib")


@st.cache_resource
def load_recommender():
    """Recommender bundle + the re-transformed feature matrix for querying."""
    b = joblib.load(MODEL_DIR / "recommender.joblib")
    nn, pre, meta, features = b["nn"], b["transformer"], b["meta"], b["features"]
    X = pre.transform(meta[features])
    return nn, meta, X


@st.cache_data
def load_set_images():
    """set_num -> image URL, for showing thumbnails in the recommender."""
    df = duckdb.connect(str(DB_PATH), read_only=True).execute(
        "SELECT set_num, set_img_url FROM raw_sets"
    ).df()
    return dict(zip(df["set_num"], df["set_img_url"]))


@st.cache_resource
def load_ab_world():
    """Pre-compute everything the A/B simulation needs so each run is instant.

    For every set we precompute its top content-neighbour and the most-popular
    set in its theme — then a simulation is just fast numpy indexing.
    """
    nn, meta, X = load_recommender()
    meta = meta.reset_index(drop=True)
    n = len(meta)
    Xn = normalize(X)

    _, nbrs = nn.kneighbors(Xn, n_neighbors=2)            # top neighbour per set
    arange = np.arange(n)
    content_top = np.where(nbrs[:, 0] == arange, nbrs[:, 1], nbrs[:, 0])

    owned = duckdb.connect(str(DB_PATH), read_only=True).execute(
        "SELECT set_num, owned_by FROM raw_brickset"
    ).df()
    meta = meta.merge(owned, on="set_num", how="left")
    meta["owned_by"] = meta["owned_by"].fillna(0)

    logpop = np.log1p(meta["owned_by"].to_numpy())
    pop_norm = (logpop - logpop.min()) / (logpop.max() - logpop.min() + 1e-9)

    top_by_theme = meta.loc[meta.groupby("theme")["owned_by"].idxmax()]
    theme_to_top = dict(zip(top_by_theme["theme"], top_by_theme.index))
    theme_top_arr = meta["theme"].map(theme_to_top).to_numpy()

    weights = meta["owned_by"].to_numpy() + 1.0
    weights = weights / weights.sum()
    return Xn, content_top, theme_top_arr, pop_norm, weights


@st.cache_data
def run_ab_sim(n_users, w_sim, seed):
    """Simulate a randomized A/B test + 2-stage funnel. Returns group/click/convert."""
    Xn, content_top, theme_top_arr, pop_norm, weights = load_ab_world()
    rng = np.random.default_rng(seed)

    user_seed = rng.choice(len(weights), size=n_users, p=weights)
    group = rng.integers(0, 2, size=n_users)              # 0 = A control, 1 = B
    shown = np.where(group == 1, content_top[user_seed], theme_top_arr[user_seed])

    sim = np.asarray(Xn[user_seed].multiply(Xn[shown]).sum(axis=1)).ravel()
    p_click = np.clip(0.03 + w_sim * sim + 0.04 * pop_norm[shown], 0.01, 0.95)
    clicked = rng.random(n_users) < p_click
    # Stage 2: of those who clicked, who converts (relevance helps here too).
    p_conv = np.clip(0.15 + 0.25 * sim, 0.01, 0.95)
    converted = clicked & (rng.random(n_users) < p_conv)
    return group, clicked, converted


LEGO_RED    = "#D01012"
LEGO_YELLOW = "#FFCF00"
LEGO_BLUE   = "#006CB7"
LEGO_GREEN  = "#00963A"
LEGO_INK    = "#1B1B1B"


# ── LEGO brand theme ───────────────────────────────────────────────────────────
def apply_lego_theme():
    """Inject the LEGO look: chunky font, brick metric cards, yellow sidebar,
    and a Plotly template so every chart inherits the brand colours + font."""
    # Plotly: layer our colourway/font ON TOP of plotly_white so we keep its
    # clean gridlines + white plot area. Charts that set explicit colours win;
    # charts that don't will pick up the LEGO palette automatically.
    pio.templates["lego"] = go.layout.Template(
        layout=dict(
            colorway=[LEGO_RED, LEGO_YELLOW, LEGO_BLUE, LEGO_GREEN, LEGO_INK],
            font=dict(family="Fredoka, sans-serif", color=LEGO_INK),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
    )
    pio.templates.default = "plotly_white+lego"

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Luckiest+Guy&display=swap');

        /* Chunky rounded font everywhere + warm "instruction-paper" page bg */
        html, body, [class*="css"], .stApp { font-family: 'Fredoka', sans-serif; }
        [data-testid="stAppViewContainer"] { background: #FAF8F2; }

        /* Section headers in LEGO red */
        h1, h2, h3 { color: #D01012 !important; font-weight: 700 !important; }

        /* Sidebar = one big yellow brick */
        section[data-testid="stSidebar"] {
            background: #FFCF00;
            border-right: 4px solid #1B1B1B;
        }
        section[data-testid="stSidebar"] * { color: #1B1B1B; }
        section[data-testid="stSidebar"] h1 { color: #1B1B1B !important; }

        /* Metric cards = LEGO bricks: thick outline, hard shadow, studs on top */
        div[data-testid="stMetric"] {
            background: #fff;
            border: 3px solid #1B1B1B;
            border-radius: 14px;
            padding: 18px 16px 14px;
            box-shadow: 4px 4px 0 #1B1B1B;
            position: relative;
        }
        div[data-testid="stMetric"]::before {
            content: "";
            position: absolute; top: -9px; left: 18px;
            width: 15px; height: 15px; border-radius: 50%;
            background: #D01012;
            box-shadow: 24px 0 0 #D01012;   /* second stud */
        }
        div[data-testid="stMetricValue"] { color: #1B1B1B; font-weight: 700; }

        /* Top banner: white Luckiest-Guy title with a black outline on red */
        .lego-banner {
            display: flex; align-items: center; gap: 12px;
            background: #D01012; border: 3px solid #1B1B1B; border-radius: 12px;
            padding: 14px 18px; margin-bottom: 18px;
        }
        .lego-studs { display: flex; gap: 6px; }
        .lego-studs span { width: 12px; height: 12px; border-radius: 50%; background: #FFCF00; }
        .lego-title {
            font-family: 'Luckiest Guy', cursive; font-size: 26px; letter-spacing: 1px;
            color: #fff; -webkit-text-stroke: 1.4px #1B1B1B; line-height: 1.2;
        }
        .lego-sub { color: #1B1B1B; font-size: 12px; font-weight: 500; }

        /* ── Sidebar nav = LEGO brick buttons (route B) ───────────────────── */
        .nav-label { font-size: 11px; font-weight: 700; color: #9a7d00;
            text-transform: uppercase; letter-spacing: .6px; margin-bottom: 2px; }
        section[data-testid="stSidebar"] button[kind] {
            position: relative; width: 100%; justify-content: flex-start; text-align: left;
            overflow: visible; font-family: 'Fredoka', sans-serif; font-weight: 700; font-size: 14px;
            color: #fff; text-shadow: 0 1px 0 rgba(0,0,0,.35);
            background: #FFC400; border: 3px solid #1B1B1B; border-radius: 9px;
            padding: 15px 14px 13px; margin-top: 18px;
            box-shadow: inset 0 5px 0 rgba(255,255,255,.35), inset 0 -8px 0 rgba(0,0,0,.16), 4px 6px 0 #1B1B1B;
            transition: transform .12s ease, background .12s ease; }
        /* the row of studs on top, drawn as a repeating radial-gradient */
        section[data-testid="stSidebar"] button[kind]::before {
            content: ""; position: absolute; top: -9px; left: 14px; right: 14px; height: 15px;
            background-image: radial-gradient(circle at center, #fff 0 5px, #1B1B1B 5px 7px, transparent 7px);
            background-size: 46px 15px; background-repeat: repeat-x; background-position: center; }
        section[data-testid="stSidebar"] button[kind]:hover {
            transform: translate(-1px,-1px); color: #fff; border-color: #1B1B1B; }
        section[data-testid="stSidebar"] button[kind]:active {
            transform: translateY(4px);
            box-shadow: inset 0 5px 0 rgba(255,255,255,.35), inset 0 -8px 0 rgba(0,0,0,.18), 1px 1px 0 #1B1B1B; }
        section[data-testid="stSidebar"] button[kind]:focus:not(:active) {
            color: #fff; border-color: #1B1B1B; outline: none;
            box-shadow: inset 0 5px 0 rgba(255,255,255,.35), inset 0 -8px 0 rgba(0,0,0,.16), 4px 6px 0 #1B1B1B; }
        /* active page = primary: brighter gold + red studs */
        section[data-testid="stSidebar"] button[kind="primary"] { background: #FFD11A; }
        section[data-testid="stSidebar"] button[kind="primary"]::before {
            background-image: radial-gradient(circle at center, #E11C1C 0 5px, #1B1B1B 5px 7px, transparent 7px); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def lego_banner():
    """The red title banner shown once at the top of the main area."""
    st.markdown(
        """
        <div class="lego-banner">
          <div class="lego-studs"><span></span><span></span></div>
          <div>
            <div class="lego-title">LEGO Product Analytics</div>
            <div class="lego-sub">Portfolio intelligence for product analysts</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


apply_lego_theme()
lego_banner()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("LEGO Product Analytics")
st.sidebar.caption("A portfolio intelligence dashboard for product analysts.")
st.sidebar.markdown("---")
PAGES = [
    "Executive Overview", "Theme Lifecycle", "Growth & Concentration",
    "Survival Predictor (ML)", "Popularity Predictor (ML)",
    "Recommender (ML)", "A/B Test (Experiment)", "AI Assistant (Agent)",
]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

# Navigation as LEGO-brick buttons. The active page is rendered as a "primary"
# button (red studs); the rest are "secondary" (white studs). Styling lives in
# apply_lego_theme(); the press feel is a CSS :active effect.
st.sidebar.markdown("<div class='nav-label'>Pages</div>", unsafe_allow_html=True)
for _p in PAGES:
    if st.sidebar.button(
        _p,
        key=f"nav_{_p}",
        use_container_width=True,
        type="primary" if st.session_state.page == _p else "secondary",
    ):
        st.session_state.page = _p
        st.rerun()
page = st.session_state.page
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Data coverage:** {int(yearly['RELEASE_YEAR'].min())} – {int(yearly['RELEASE_YEAR'].max())}  \n"
    f"**Themes tracked:** {lifecycle['THEME_NAME'].nunique()}  \n"
    f"**Total sets:** {size_seg['SET_COUNT'].sum():,}"
)
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Data source & method"):
    st.markdown(
        "Source: **Rebrickable** catalogue data, ingested via Snowflake and "
        "aggregated in DuckDB. Figures reflect **sets released** (catalogue "
        "breadth), not units sold or revenue — Rebrickable does not carry "
        "sales, price, or rating data. Trends should be read as portfolio "
        "*strategy* signals, not demand signals."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Executive Overview":
    st.title("Executive Overview")
    st.markdown(
        "High-level portfolio health metrics across LEGO's entire product history."
    )

    # ── KPI Cards ────────────────────────────────────────────────────────────
    latest       = yearly.sort_values("RELEASE_YEAR").iloc[-1]
    prev         = yearly.sort_values("RELEASE_YEAR").iloc[-2]
    active_count = (lifecycle["STATUS"] == "Active").sum()
    retired_count= (lifecycle["STATUS"] == "Retired").sum()
    flagship_row = complexity.sort_values("RELEASE_YEAR").iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Sets Released (Latest Year)",
        int(latest["TOTAL_SETS"]),
        delta=int(latest["TOTAL_SETS"] - prev["TOTAL_SETS"]),
    )
    col2.metric(
        "Avg Parts per Set",
        f"{latest['AVG_PARTS']:.0f}",
        delta=f"{latest['AVG_PARTS'] - prev['AVG_PARTS']:.0f}",
    )
    col3.metric("Active Themes", active_count)
    col4.metric(
        "Flagship Sets % (Latest Year)",
        f"{flagship_row['PCT_OVER_1000']:.1f}%",
    )

    # ── Insight callout (computed from the data, not hardcoded) ──────────────
    parts_series = yearly.dropna(subset=["AVG_PARTS"]).sort_values("RELEASE_YEAR")
    first_parts  = parts_series.iloc[0]
    last_parts   = parts_series.iloc[-1]
    parts_mult   = last_parts["AVG_PARTS"] / first_parts["AVG_PARTS"]
    st.info(
        f"**Takeaway:** Average parts per set has grown "
        f"**{parts_mult:.1f}×** since {int(first_parts['RELEASE_YEAR'])} "
        f"({first_parts['AVG_PARTS']:.0f} → {last_parts['AVG_PARTS']:.0f} parts), "
        f"while flagship sets (1,000+ parts) now make up "
        f"**{flagship_row['PCT_OVER_1000']:.1f}%** of releases. The catalogue is "
        "shifting toward larger, premium builds — consistent with LEGO's push "
        "into the adult (18+) market."
    )

    st.markdown("---")

    # ── Row 2: Volume + Complexity trend ─────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Portfolio Volume & Complexity Over Time")
        st.caption(
            "Left axis: total sets released per year. "
            "Right axis: average number of parts per set."
        )
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(
            x=yearly["RELEASE_YEAR"], y=yearly["TOTAL_SETS"],
            name="Total Sets",
            line=dict(color=LEGO_RED, width=2),
            fill="tozeroy", fillcolor="rgba(227,0,11,0.08)",
        ))
        fig_dual.add_trace(go.Scatter(
            x=yearly["RELEASE_YEAR"], y=yearly["AVG_PARTS"],
            name="Avg Parts",
            yaxis="y2",
            line=dict(color=LEGO_BLUE, width=2, dash="dash"),
            fill="tozeroy", fillcolor="rgba(0,123,255,0.2)"
        ))
        fig_dual.update_layout(
            yaxis=dict(title="Total Sets Released"),
            yaxis2=dict(title="Avg Parts per Set", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
            height=360,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_dual, use_container_width=True, theme=None)

    with col_b:
        st.subheader("Flagship Product Trend")
        st.caption(
            "Percentage of sets with more than 1,000 parts — "
            "a proxy for premium product investment."
        )
        fig_flag = go.Figure()
        fig_flag.add_trace(go.Scatter(
            x=complexity["RELEASE_YEAR"], y=complexity["PCT_OVER_1000"],
            name="% Flagship",
            fill="tozeroy",
            line=dict(color=LEGO_YELLOW, width=2),
            fillcolor="rgba(245,197,24,0.2)",
        ))
        fig_flag.update_layout(
            yaxis=dict(title="% Sets with > 1,000 Parts"),
            hovermode="x unified",
            height=360,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_flag, use_container_width=True, theme=None)

    st.markdown("---")

    # ── Row 3: Size segmentation + scatter ───────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Product Tier Distribution")
        st.caption(
            "How LEGO's catalogue splits across size tiers "
            "(Small < 100 parts, Medium 100–499, Large 500–999, Flagship 1,000+)."
        )
        fig_pie = px.pie(
            size_seg,
            names="SIZE_CATEGORY",
            values="SET_COUNT",
            hole=0.5,
            color_discrete_sequence=[LEGO_RED, LEGO_BLUE, LEGO_YELLOW, "#2ECC71"],
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(height=360, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

    with col_d:
        st.subheader("Volume vs. Complexity by Year")
        st.caption(
            "Each dot is a year. Darker = more recent. "
            "Reveals whether high-volume years also had complex sets."
        )
        fig_scat = px.scatter(
            yearly,
            x="TOTAL_SETS",
            y="AVG_PARTS",
            color="RELEASE_YEAR",
            size="TOTAL_SETS",
            hover_data=["RELEASE_YEAR"],
            color_continuous_scale="RdBu",
            labels={
                "TOTAL_SETS": "Total Sets Released",
                "AVG_PARTS": "Avg Parts per Set",
                "RELEASE_YEAR": "Year",
            },
        )
        fig_scat.update_layout(height=360, margin=dict(t=10))
        st.plotly_chart(fig_scat, use_container_width=True, theme=None)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — THEME LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Theme Lifecycle":
    st.title("Theme Lifecycle Analysis")
    st.markdown(
        "Understand how product lines are born, mature, and retire — "
        "the classic product lifecycle applied to LEGO themes."
    )

    # ── 3D Portfolio Matrix ───────────────────────────────────────────────────
    st.subheader("3D Portfolio Matrix — Launch Year × Lifespan × Complexity")
    st.caption(
        "Each bubble is a theme. **X** = year it launched, **Y** = how many years it ran, "
        "**Z** = average parts per set (complexity). **Size** = total sets ever released. "
        "**Color** = current status."
    )

    lc = lifecycle.dropna(subset=["FIRST_YEAR", "LIFESPAN_YEARS", "AVG_PARTS", "TOTAL_SETS"])

    fig_3d = px.scatter_3d(
        lc,
        x="FIRST_YEAR",
        y="LIFESPAN_YEARS",
        z="AVG_PARTS",
        color="STATUS",
        size="TOTAL_SETS",
        hover_name="THEME_NAME",
        hover_data={
            "PARENT_THEME": True,
            "TOTAL_SETS": True,
            "FIRST_YEAR": True,
            "LATEST_YEAR": True,
            "STATUS": False,
        },
        color_discrete_map={"Active": "#2ECC71", "Retired": LEGO_RED},
        labels={
            "FIRST_YEAR": "Launch Year",
            "LIFESPAN_YEARS": "Lifespan (years)",
            "AVG_PARTS": "Avg Parts (Complexity)",
        },
        height=600,
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_3d, use_container_width=True, theme=None)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("New vs. Returning Themes Per Year")
        st.caption(
            "Innovation rate: how many new themes launched each year "
            "vs. continuing themes."
        )
        yr_min = int(new_vs_ret["RELEASE_YEAR"].min())
        yr_max = int(new_vs_ret["RELEASE_YEAR"].max())
        nvr_range = st.slider(
            "Year range", yr_min, yr_max, (yr_min, yr_max), key="nvr_slider"
        )
        nvr = new_vs_ret[
            new_vs_ret["RELEASE_YEAR"].between(nvr_range[0], nvr_range[1])
        ]
        fig_nvr = go.Figure()
        fig_nvr.add_trace(go.Bar(
            x=nvr["RELEASE_YEAR"],
            y=nvr["NEW_THEMES"],
            name="New Themes",
            marker_color=LEGO_RED,
        ))
        fig_nvr.add_trace(go.Bar(
            x=nvr["RELEASE_YEAR"],
            y=nvr["RETURNING_THEMES"],
            name="Returning Themes",
            marker_color=LEGO_BLUE,
        ))
        fig_nvr.add_trace(go.Scatter(
            x=nvr["RELEASE_YEAR"],
            y=nvr["NEW_THEME_PCT"],
            name="% New (right)",
            yaxis="y2",
            line=dict(color=LEGO_YELLOW, width=2),
        ))
        fig_nvr.update_layout(
            barmode="stack",
            yaxis=dict(title="Number of Themes"),
            yaxis2=dict(title="% New Themes", overlaying="y", side="right"),
            hovermode="x unified",
            height=380,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_nvr, use_container_width=True, theme=None)

    with col_b:
        st.subheader("Theme Portfolio Depth (Top 15 Parent Themes)")
        st.caption(
            "How broad is each parent theme's sub-theme tree? "
            "Bar length = total sets; color = number of sub-themes."
        )
        # Spacer to match the year-range slider in the left column, so this
        # chart's top lines up with the New-vs-Returning chart beside it.
        st.markdown("<div style='height:74px'></div>", unsafe_allow_html=True)
        depth_top = (
            theme_depth
            .sort_values("TOTAL_SETS_IN_TREE", ascending=True)
            .tail(15)
        )
        fig_depth = px.bar(
            depth_top,
            x="TOTAL_SETS_IN_TREE",
            y="PARENT_THEME",
            orientation="h",
            color="NUM_SUB_THEMES",
            color_continuous_scale="Blues",
            labels={
                "TOTAL_SETS_IN_TREE": "Total Sets in Theme Tree",
                "PARENT_THEME": "Parent Theme",
                "NUM_SUB_THEMES": "# Sub-themes",
            },
            height=380,
        )
        fig_depth.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_depth, use_container_width=True, theme=None)

    # ── Active vs Retired summary ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Active vs. Retired Theme Summary")
    col_c, col_d, col_e = st.columns(3)
    active_lc  = lifecycle[lifecycle["STATUS"] == "Active"]
    retired_lc = lifecycle[lifecycle["STATUS"] == "Retired"]

    col_c.metric("Active Themes", len(active_lc))
    col_c.metric("Avg Lifespan (Active)", f"{active_lc['LIFESPAN_YEARS'].mean():.1f} yrs")
    col_d.metric("Retired Themes", len(retired_lc))
    col_d.metric("Avg Lifespan (Retired)", f"{retired_lc['LIFESPAN_YEARS'].mean():.1f} yrs")
    col_e.metric(
        "Longest-Running Active",
        active_lc.sort_values("LIFESPAN_YEARS", ascending=False).iloc[0]["THEME_NAME"],
    )
    col_e.metric(
        "Most Complex Active (Avg Parts)",
        active_lc.sort_values("AVG_PARTS", ascending=False).iloc[0]["THEME_NAME"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — GROWTH & CONCENTRATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Growth & Concentration":
    st.title("Theme Growth & Portfolio Concentration")
    st.markdown(
        "Track which themes are growing and how concentrated the portfolio is — "
        "key inputs for product prioritisation decisions."
    )

    # ── YoY Growth ───────────────────────────────────────────────────────────
    st.subheader("Year-over-Year Growth by Theme")
    st.caption(
        "Select one or more themes to compare their annual set-release growth rate. "
        "Dashed line = 0% (flat). Points above = growth, below = decline."
    )
    all_themes = sorted(yoy_growth["THEME_NAME"].dropna().unique())
    default_themes = all_themes[:6]
    selected = st.multiselect("Choose themes to compare", all_themes, default=default_themes)

    if selected:
        filtered = yoy_growth[yoy_growth["THEME_NAME"].isin(selected)]
        fig_yoy = px.line(
            filtered,
            x="RELEASE_YEAR",
            y="YOY_PCT_CHANGE",
            color="THEME_NAME",
            markers=True,
            labels={
                "YOY_PCT_CHANGE": "YoY % Change in Sets",
                "RELEASE_YEAR": "Year",
                "THEME_NAME": "Theme",
            },
            height=380,
        )
        fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_yoy.update_layout(hovermode="x unified", margin=dict(t=10))
        st.plotly_chart(fig_yoy, use_container_width=True, theme=None)
    else:
        st.info("Select at least one theme above.")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Portfolio Concentration by Year")
        st.caption(
            "Which themes dominated in a given year? "
            "Color intensity = share of that year's total releases."
        )
        years = sorted(concentration["RELEASE_YEAR"].unique(), reverse=True)
        sel_year = st.selectbox("Select a year", years)
        year_data = (
            concentration[concentration["RELEASE_YEAR"] == sel_year]
            .sort_values("SETS_RELEASED")
        )
        fig_conc = px.bar(
            year_data,
            x="SETS_RELEASED",
            y="THEME_NAME",
            orientation="h",
            color="PCT_OF_YEAR",
            color_continuous_scale="Reds",
            labels={
                "SETS_RELEASED": "Sets Released",
                "THEME_NAME": "Theme",
                "PCT_OF_YEAR": "% of Year's Total",
            },
            height=420,
        )
        fig_conc.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_conc, use_container_width=True, theme=None)

    with col_b:
        st.subheader("Product Size Mix by Theme")
        st.caption(
            "What is the SKU mix within a theme? "
            "A theme heavy in Small sets has a different margin profile than one heavy in Flagship."
        )
        theme_list = sorted(size_mix["THEME_NAME"].dropna().unique())
        sel_theme = st.selectbox("Select a theme", theme_list)
        theme_data = size_mix[size_mix["THEME_NAME"] == sel_theme]
        fig_mix = px.pie(
            theme_data,
            names="SIZE_CATEGORY",
            values="PCT_OF_THEME",
            hole=0.45,
            color_discrete_sequence=[LEGO_RED, LEGO_BLUE, LEGO_YELLOW, "#2ECC71"],
        )
        fig_mix.update_traces(textposition="outside", textinfo="percent+label")
        fig_mix.update_layout(height=420, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_mix, use_container_width=True, theme=None)

    # ── Cumulative concentration curve ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Portfolio Concentration Curve (Selected Year)")
    st.caption(
        "Pareto-style view: what share of themes accounts for 80% of releases? "
        "Steeper = more concentrated portfolio."
    )
    conc_year = concentration[concentration["RELEASE_YEAR"] == sel_year].sort_values(
        "THEME_RANK"
    )
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(
        x=conc_year["THEME_RANK"],
        y=conc_year["CUMULATIVE_PCT"],
        mode="lines+markers",
        line=dict(color=LEGO_RED, width=2),
        name="Cumulative %",
    ))
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="gray",
                         annotation_text="80% threshold")
    fig_pareto.update_layout(
        xaxis_title="Theme Rank (1 = largest)",
        yaxis_title="Cumulative % of Year's Sets",
        height=340,
        margin=dict(t=10),
    )
    st.plotly_chart(fig_pareto, use_container_width=True, theme=None)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SURVIVAL PREDICTOR (ML)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Survival Predictor (ML)":
    metrics, feats = load_survival_data()
    bundle = load_survival_model()
    model, feature_cols = bundle["model"], bundle["features"]

    FEATURE_LABELS = {
        "first_year": "Launch year",
        "n_sets": "Total sets ever released",
        "avg_parts": "Avg parts per set",
        "median_parts": "Median parts per set",
        "max_parts": "Largest set (parts)",
        "num_sub_themes": "# Sub-themes",
    }

    st.title("Theme Survival Predictor 🤖")
    st.markdown(
        "A machine-learning model that predicts whether a LEGO **product line** "
        "(top-level theme, rolled up over its whole sub-tree) is still **Active** "
        "or has been **Retired**, from structural features (scale, complexity, age). "
        "Useful for spotting **at-risk product lines** early."
    )
    st.caption(
        "Model: Random Forest. To avoid **data leakage**, features that directly "
        "encode recency (last release year, lifespan) are deliberately excluded — "
        "so the score reflects genuine signal, not the answer in disguise."
    )

    # ── Model performance ────────────────────────────────────────────────────
    st.subheader("Model Performance (held-out test set)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{metrics['test_roc_auc']:.2f}",
              help="Ability to rank Active above Retired. 0.5 = random, 1.0 = perfect.")
    m2.metric("Cross-val ROC-AUC", f"{metrics['cv_roc_auc_mean']:.2f}",
              help="5-fold CV — more robust than a single split.")
    m3.metric("Accuracy", f"{metrics['accuracy']*100:.0f}%")
    m4.metric("Active Recall", f"{metrics['active_recall']*100:.0f}%",
              help="Of truly Active themes, how many the model catches.")

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("What Drives Theme Survival?")
        st.caption("Random Forest feature importance — higher = more predictive.")
        imp = (
            pd.Series(metrics["feature_importances"])
            .rename(index=FEATURE_LABELS)
            .sort_values()
        )
        fig_imp = px.bar(
            imp, orientation="h",
            color=imp.values, color_continuous_scale="Reds",
            labels={"value": "Importance", "index": ""},
        )
        fig_imp.update_layout(height=360, showlegend=False, margin=dict(t=10),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True, theme=None)

    with col_b:
        st.subheader("Confusion Matrix")
        st.caption("Test-set predictions vs. reality.")
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Pred Retired", "Pred Active"],
            y=["Actual Retired", "Actual Active"],
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=360, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True, theme=None)

    # ── Watchlists: at-risk + comeback candidates ─────────────────────────────
    st.markdown("---")
    st.subheader("Business Watchlists")
    st.caption(
        "Where the model and current status disagree — the actionable part. "
        "**At-risk:** currently Active but the model thinks it looks like a retiree. "
        "**Comeback-shaped:** Retired but structurally resembles survivors."
    )
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**⚠️ At-risk active themes** (lowest predicted survival)")
        at_risk = (
            feats[feats["is_active"] == 1]
            .nsmallest(10, "pred_active_proba")[["name", "pred_active_proba", "n_sets"]]
            .rename(columns={"name": "Theme", "pred_active_proba": "Survival prob",
                             "n_sets": "Total sets"})
        )
        st.dataframe(
            at_risk.style.format({"Survival prob": "{:.0%}"}),
            hide_index=True, use_container_width=True,
        )

    with col_d:
        st.markdown("**🔁 Comeback-shaped retired themes** (highest predicted survival)")
        comeback = (
            feats[feats["is_active"] == 0]
            .nlargest(10, "pred_active_proba")[["name", "pred_active_proba", "n_sets"]]
            .rename(columns={"name": "Theme", "pred_active_proba": "Survival prob",
                             "n_sets": "Total sets"})
        )
        st.dataframe(
            comeback.style.format({"Survival prob": "{:.0%}"}),
            hide_index=True, use_container_width=True,
        )

    # ── Interactive what-if predictor ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("What-If: Score a Hypothetical Theme")
    st.caption("Move the sliders to describe a theme; the model predicts its survival probability live.")

    inputs = {}
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        inputs["first_year"] = st.slider("Launch year", 1949, int(metrics["latest_year"]), 2010)
        inputs["n_sets"] = st.slider("Total sets ever released (whole sub-tree)", 1, 1200, 80)
    with wc2:
        inputs["avg_parts"] = st.slider("Avg parts per set", 0, 2000, 250)
        inputs["median_parts"] = st.slider("Median parts per set", 0, 2000, 150)
    with wc3:
        inputs["max_parts"] = st.slider("Largest set (parts)", 0, 12000, 1500)
        inputs["num_sub_themes"] = st.slider("# Sub-themes", 0, 100, 5)

    row = pd.DataFrame([[inputs[c] for c in feature_cols]], columns=feature_cols)
    proba = float(model.predict_proba(row)[:, 1][0])
    verdict = "🟢 Likely ACTIVE" if proba >= 0.5 else "🔴 Likely RETIRED"
    # Constrain the result block to the same width as the Launch-year slider
    # (the first of the three input columns above).
    out1, _, _ = st.columns(3)
    with out1:
        st.metric("Predicted survival probability", f"{proba:.0%}", help=verdict)
        st.progress(proba)
        st.markdown(f"### {verdict}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — POPULARITY PREDICTOR (ML)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Popularity Predictor (ML)":
    metrics, scored = load_popularity_data()
    bundle = load_popularity_model()
    model, feature_cols = bundle["model"], bundle["features"]

    FEATURE_LABELS = {
        "theme": "Theme",
        "minifigs": "# Minifigures",
        "num_parts": "# Parts",
        "year": "Release year",
    }

    st.title("Set Popularity Predictor 🤖")
    st.markdown(
        "Predicts how many collectors will end up **owning** a set "
        "(Brickset `ownedBy`), from attributes known at **design time**. "
        "Useful for forecasting demand and prioritising the product pipeline."
    )
    st.caption(
        "Model: Random Forest regressor on log(owned_by). Post-release signals "
        "(rating, reviews, wanted-by) are **excluded** to avoid data leakage — "
        "the model only sees what's knowable before a set ships."
    )

    # ── Performance ────────────────────────────────────────────────────────────
    st.subheader("Model Performance (held-out test set)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (log scale)", f"{metrics['r2']:.2f}",
              help="Share of popularity variance explained. 1.0 = perfect.")
    m2.metric("Cross-val R²", f"{metrics['cv_r2_mean']:.2f}")
    m3.metric("MAE", f"{metrics['mae_real_owners']:,}", help="Avg error in real owner counts.")
    m4.metric("Sets trained on", f"{metrics['n_rows']:,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("What Drives Popularity?")
        st.caption("Random Forest feature importance.")
        imp = (
            pd.Series(metrics["feature_importances"])
            .rename(index=FEATURE_LABELS)
            .sort_values()
        )
        fig_imp = px.bar(
            imp, orientation="h",
            color=imp.values, color_continuous_scale="Reds",
            labels={"value": "Importance", "index": ""},
        )
        fig_imp.update_layout(height=320, showlegend=False, margin=dict(t=10),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True, theme=None)

    with col_b:
        st.subheader("Predicted vs. Actual Ownership")
        st.caption("Each dot is a set. Closer to the diagonal = better prediction.")
        fig_pa = px.scatter(
            scored, x="pred_owned_by", y="owned_by",
            opacity=0.35, hover_name="name",
            labels={"pred_owned_by": "Predicted owners", "owned_by": "Actual owners"},
        )
        lim = int(max(scored["owned_by"].max(), scored["pred_owned_by"].max()))
        fig_pa.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                         line=dict(color="gray", dash="dash"))
        fig_pa.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig_pa, use_container_width=True, theme=None)

    # ── Surprise hits vs over-expected ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Surprises: Model vs. Reality")
    st.caption(
        "Where actual ownership diverges most from the prediction. "
        "**Hidden hits:** far more popular than their attributes suggest. "
        "**Over-expected:** the model expected more love than they got."
    )
    sc = scored.copy()
    sc["residual"] = sc["owned_by"] - sc["pred_owned_by"]
    cols = ["name", "theme", "year", "owned_by", "pred_owned_by"]
    rename = {"name": "Set", "theme": "Theme", "year": "Year",
              "owned_by": "Actual owners", "pred_owned_by": "Predicted"}
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**💎 Hidden hits** (actual ≫ predicted)")
        st.dataframe(sc.nlargest(10, "residual")[cols].rename(columns=rename),
                     hide_index=True, use_container_width=True)
    with col_d:
        st.markdown("**📉 Over-expected** (predicted ≫ actual)")
        st.dataframe(sc.nsmallest(10, "residual")[cols].rename(columns=rename),
                     hide_index=True, use_container_width=True)

    # ── What-if predictor ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("What-If: Forecast a Hypothetical Set's Popularity")
    st.caption("Describe a set; the model predicts how many collectors would own it.")

    themes = sorted(scored["theme"].dropna().unique())
    default_theme = themes.index("Star Wars") if "Star Wars" in themes else 0
    wc1, wc2 = st.columns(2)
    with wc1:
        in_theme = st.selectbox("Theme", themes, index=default_theme)
        in_year = st.slider("Release year", 1949, 2027, 2024)
    with wc2:
        in_parts = st.slider("# Parts", 0, 12000, 500)
        in_minifigs = st.slider("# Minifigures", 0, 20, 3)

    row = pd.DataFrame(
        [{"year": in_year, "num_parts": in_parts, "minifigs": in_minifigs, "theme": in_theme}]
    )[feature_cols]
    pred = int(np.expm1(model.predict(row))[0])
    # Match the Release-year slider width (first of the two input columns).
    out1, _ = st.columns(2)
    with out1:
        st.metric("Predicted collectors who would own this set", f"{pred:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RECOMMENDER (ML)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Recommender (ML)":
    nn, meta, X = load_recommender()
    images = load_set_images()

    st.title("Set Recommender 🤖")
    st.markdown(
        "Content-based recommendations: **\"if you like this set, you'll like these.\"** "
        "Each set is a feature vector (theme, size, era, minifigures); we return its "
        "nearest neighbours by **cosine similarity**."
    )
    st.caption(
        "No user-behaviour data exists, so this is content-based (not collaborative "
        "filtering). Built on a NearestNeighbors index queried on demand — no giant "
        "similarity matrix stored."
    )

    # ── Pick a seed set (theme filter -> set) ──────────────────────────────────
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        themes = sorted(meta["theme"].dropna().unique())
        default_t = themes.index("Star Wars") if "Star Wars" in themes else 0
        sel_theme = st.selectbox("1. Pick a theme", themes, index=default_t)
    with c2:
        pool = meta[meta["theme"] == sel_theme].sort_values("year")
        labels = pool["name"] + " (" + pool["year"].astype(int).astype(str) + ")"
        label_to_num = dict(zip(labels, pool["set_num"]))
        sel_label = st.selectbox("2. Pick a set you like", labels.tolist())
        seed = label_to_num[sel_label]
    with c3:
        n_rec = st.slider("How many recs", 3, 12, 6)

    seed_row = meta[meta["set_num"] == seed].iloc[0]

    # ── Seed set ───────────────────────────────────────────────────────────────
    st.markdown("---")
    s1, s2 = st.columns([1, 3])
    with s1:
        if images.get(seed):
            st.image(images[seed], width=160)
    with s2:
        st.markdown(f"### {seed_row['name']}")
        st.markdown(
            f"**Theme:** {seed_row['theme']}  |  **Year:** {int(seed_row['year'])}  |  "
            f"**Parts:** {int(seed_row['num_parts'])}  |  **Minifigs:** {int(seed_row['minifigs'])}"
        )

    # ── Compute recommendations ────────────────────────────────────────────────
    idx = meta.index[meta["set_num"] == seed][0]
    dist, nbrs = nn.kneighbors(X[idx], n_neighbors=n_rec + 1)
    recs = meta.iloc[nbrs[0]].copy()
    recs["similarity"] = 1 - dist[0]
    recs = recs[recs["set_num"] != seed].head(n_rec)

    st.markdown(f"#### Recommended because you like *{seed_row['name']}*")
    cols = st.columns(min(n_rec, 6))
    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % len(cols)]:
            if images.get(r["set_num"]):
                st.image(images[r["set_num"]], use_container_width=True)
            st.caption(
                f"**{r['name']}**  \n{r['theme']} · {int(r['year'])}  \n"
                f"Similarity: {r['similarity']:.0%}"
            )

    with st.expander("See recommendation details (table)"):
        st.dataframe(
            recs[["name", "theme", "year", "num_parts", "minifigs", "similarity"]]
            .rename(columns={"name": "Set", "theme": "Theme", "year": "Year",
                             "num_parts": "Parts", "minifigs": "Minifigs",
                             "similarity": "Similarity"})
            .style.format({"Similarity": "{:.0%}"}),
            hide_index=True, use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — A/B TEST (EXPERIMENT)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "A/B Test (Experiment)":
    st.title("A/B Test: Content vs. Popularity Recommender 🧪")
    st.markdown(
        "A simulated online experiment. Users are randomly split: **Control (A)** "
        "sees a *theme-popularity* recommendation, **Treatment (B)** sees our "
        "*content-based* recommender. We then test whether B lifts click-through."
    )

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        n_users = st.slider("Total users in experiment", 2_000, 40_000, 20_000, step=2_000)
    with cfg2:
        w_sim = st.slider("Relevance weight (true effect size)", 0.05, 0.35, 0.18, step=0.01,
                          help="How strongly relevance drives clicks. Higher = bigger true B−A gap.")
    with cfg3:
        seed = st.slider("Random seed (re-roll the experiment)", 1, 100, 42)

    group, clicked, converted = run_ab_sim(n_users, w_sim, seed)
    a, b = group == 0, group == 1
    n_a, n_b = int(a.sum()), int(b.sum())
    clk_a, clk_b = int(clicked[a].sum()), int(clicked[b].sum())
    cnv_a, cnv_b = int(converted[a].sum()), int(converted[b].sum())
    ctr_a, ctr_b = clk_a / n_a, clk_b / n_b

    z, p = proportions_ztest([clk_b, clk_a], [n_b, n_a])
    ci_a = proportion_confint(clk_a, n_a, method="wilson")
    ci_b = proportion_confint(clk_b, n_b, method="wilson")
    abs_lift, rel_lift = ctr_b - ctr_a, (ctr_b - ctr_a) / ctr_a

    # ── Result KPIs ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Control A — CTR", f"{ctr_a:.2%}", help=f"95% CI [{ci_a[0]:.2%}, {ci_a[1]:.2%}]")
    k2.metric("Treatment B — CTR", f"{ctr_b:.2%}", f"{abs_lift:+.2%}",
              help=f"95% CI [{ci_b[0]:.2%}, {ci_b[1]:.2%}]")
    k3.metric("Relative lift", f"{rel_lift:+.1%}")
    sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
    k4.metric("p-value", f"{p:.1e}", sig, delta_color="off")

    st.markdown("---")
    col_f, col_p = st.columns(2)

    # ── User funnel ────────────────────────────────────────────────────────────
    with col_f:
        st.subheader("User Behaviour Funnel")
        st.caption("How each arm's users flow from impression → click → conversion.")
        funnel = pd.DataFrame({
            "Stage": ["Shown rec (impression)", "Clicked", "Converted"] * 2,
            "Users": [n_a, clk_a, cnv_a, n_b, clk_b, cnv_b],
            "Variant": ["A · Popularity"] * 3 + ["B · Content"] * 3,
        })
        fig_fun = px.funnel(
            funnel, x="Users", y="Stage", color="Variant",
            color_discrete_map={"A · Popularity": LEGO_BLUE, "B · Content": LEGO_RED},
        )
        fig_fun.update_layout(height=380, margin=dict(t=10),
                              legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_fun, use_container_width=True, theme=None)
        conv_a = cnv_a / n_a if n_a else 0
        conv_b = cnv_b / n_b if n_b else 0
        st.caption(f"End-to-end conversion: A = {conv_a:.2%}  ·  B = {conv_b:.2%}")

    # ── Power curve ────────────────────────────────────────────────────────────
    with col_p:
        st.subheader("Statistical Power vs. Sample Size")
        st.caption(
            "Given this effect size, how likely are we to detect it (reject H₀) "
            "at different per-arm sample sizes? Industry target = 80%."
        )
        eff = abs(proportion_effectsize(ctr_b, ctr_a))
        ns = np.linspace(100, max(8_000, n_b * 1.5), 60)
        powers = NormalIndPower().power(effect_size=eff, nobs1=ns,
                                        alpha=0.05, alternative="two-sided")
        try:
            n_needed = NormalIndPower().solve_power(
                effect_size=eff, alpha=0.05, power=0.80, alternative="two-sided")
        except Exception:
            n_needed = float("nan")
        fig_pw = go.Figure()
        fig_pw.add_trace(go.Scatter(x=ns, y=powers, line=dict(color=LEGO_RED, width=2),
                                    name="Power"))
        fig_pw.add_hline(y=0.80, line_dash="dash", line_color="gray",
                         annotation_text="80% target")
        fig_pw.add_vline(x=n_b, line_dash="dot", line_color=LEGO_BLUE,
                         annotation_text="Your test (per arm)")
        fig_pw.update_layout(height=380, margin=dict(t=10),
                             xaxis_title="Sample size per arm", yaxis_title="Power")
        st.plotly_chart(fig_pw, use_container_width=True, theme=None)
        if np.isfinite(n_needed):
            st.caption(f"Minimum sample size per arm for 80% power: **{int(np.ceil(n_needed)):,}** "
                       f"(you have {n_b:,}).")

    # ── Interpretation ─────────────────────────────────────────────────────────
    st.markdown("---")
    if p < 0.05 and abs_lift > 0:
        st.success(
            f"**Decision: ship Treatment B.** Content recommendations lifted CTR by "
            f"a relative **{rel_lift:.1%}** ({ctr_a:.2%} → {ctr_b:.2%}), and the result "
            f"is statistically significant (p = {p:.1e} < 0.05). The two confidence "
            f"intervals barely overlap, reinforcing the conclusion."
        )
    else:
        st.warning(
            f"**Decision: do not ship (yet).** The observed lift is {rel_lift:+.1%} but "
            f"p = {p:.2f} ≥ 0.05 — we can't rule out random noise. Either the true effect "
            f"is small or the sample is underpowered. Try increasing users or effect size."
        )
    st.caption(
        "⚠️ Pitfalls a real test must avoid: peeking/stopping early (inflates false "
        "positives), testing many metrics without correction, and unequal/again-and-again "
        "re-randomisation. Decide sample size up front from a power analysis."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — AI ASSISTANT (RAG)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "AI Assistant (Agent)":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from rag_assistant import CHROMA_DIR, build_index
    from agent import agent_answer

    st.title("LEGO Analytics AI Assistant 🤖")
    st.caption(
        "A **hybrid agent** running on a local model (Ollama · llama3.2) — no "
        "cloud, no cost. It routes each question to the right tool: **docs** "
        "(RAG over the knowledge files) for concepts, or **SQL** (Text-to-SQL "
        "over DuckDB) for precise numbers."
    )

    if not CHROMA_DIR.exists():
        st.warning("Vector index not built yet.")
        if st.button("Build index now"):
            with st.spinner("Indexing knowledge docs…"):
                build_index()
            st.success("Index built — ask away!")

    with st.expander("ℹ️ How it works & sample questions"):
        st.markdown(
            "**The agent routes each question to a tool:**\n"
            "- **docs (RAG):** retrieves from `docs/knowledge/` and answers from "
            "those passages — for concepts & methodology.\n"
            "- **SQL (Text-to-SQL):** writes a read-only query against DuckDB, runs "
            "it, and explains the result — for precise numbers. Self-corrects on error.\n\n"
            "**Concept questions (→ docs):**\n"
            "- What is a flagship set?\n"
            "- How do you avoid data leakage?\n\n"
            "**Data questions (→ SQL):**\n"
            "- How many themes are currently active?\n"
            "- Which theme has the most sets?\n\n"
            "*Note: the local model is small (3B), so complex SQL can occasionally "
            "be wrong — the tool used and the exact query are shown for transparency.*"
        )

    col_clear, _ = st.columns([1, 4])
    if col_clear.button("Clear chat"):
        st.session_state.rag_messages = []

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for m in st.session_state.rag_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about the LEGO analytics project…"):
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking (local model)…"):
                try:
                    answer, meta = agent_answer(prompt)
                    if meta.get("tool") == "sql":
                        reply = answer + "\n\n*🛠️ Answered via the SQL tool*"
                        rows = meta.get("rows")
                        if rows is not None and not rows.empty:
                            reply += ("\n\n**Query result (ground truth):**\n```\n"
                                      f"{rows.head(20).to_string(index=False)}\n```")
                        if meta.get("sql"):
                            reply += f"\n```sql\n{meta['sql']}\n```"
                    else:
                        srcs = ", ".join(meta.get("sources", []))
                        reply = answer + f"\n\n*📚 Answered via the docs tool — sources: {srcs}*"
                except Exception as e:
                    reply = (
                        f"⚠️ Something went wrong: `{e}`\n\n"
                        "Check that **Ollama is running** and the index is built "
                        "(`python src/rag_assistant.py build`)."
                    )
            st.markdown(reply)
        st.session_state.rag_messages.append({"role": "assistant", "content": reply})
