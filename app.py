"""
Failure Forecasters — AI-Driven Water Infrastructure Upgrade Planner
A decision-support tool for water utilities, municipalities, and local governments
to prioritize infrastructure asset replacements based on risk, cost, and sustainability.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Failure Forecasters | Water Infrastructure Planner",
    page_icon="\U0001F4A7",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium, minimal, water/infrastructure palette
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Global ---- */
.block-container {padding-top:1.2rem;padding-bottom:2rem;max-width:1100px;}
html, body, [class*="css"] {font-family:'Inter','Segoe UI',system-ui,sans-serif;}

/* ---- Hero ---- */
.hero{background:linear-gradient(135deg,#0b3d54 0%,#146b8a 55%,#1a9d8f 100%);
  border-radius:14px;padding:1.3rem 2rem 1.1rem;color:#fff;margin-bottom:1.4rem;}
.hero h1{font-size:1.55rem;font-weight:700;margin:0 0 .2rem;letter-spacing:-.4px;}
.hero p{font-size:.85rem;opacity:.88;margin:0;line-height:1.45;max-width:680px;}

/* ---- Step labels ---- */
.step-label{font-size:.7rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:#64748b;margin-top:1.5rem;margin-bottom:.55rem;
  display:flex;align-items:center;gap:.45rem;}
.step-num{background:#0b3d54;color:#fff;border-radius:5px;
  padding:.1rem .45rem;font-size:.65rem;letter-spacing:.04em;}

/* ---- Metric cards (HTML-based for subtitle support) ---- */
.kpi-row{display:flex;gap:.75rem;margin-bottom:1rem;}
.kpi{flex:1;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:.85rem 1rem .7rem;min-width:0;}
.kpi .kpi-label{font-size:.65rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:#64748b;margin-bottom:.15rem;}
.kpi .kpi-value{font-size:1.35rem;font-weight:700;color:#0f172a;line-height:1.2;}
.kpi .kpi-sub{font-size:.72rem;color:#94a3b8;margin-top:.15rem;line-height:1.3;}

/* ---- Recommendation panel ---- */
.rec-panel{background:linear-gradient(135deg,#f0fdf4 0%,#ecfdf5 50%,#f0fdfa 100%);
  border:1px solid #a7f3d0;border-radius:12px;padding:1.3rem 1.6rem 1.1rem;
  margin-bottom:.8rem;position:relative;overflow:hidden;}
.rec-panel::before{content:'';position:absolute;top:0;left:0;width:4px;
  height:100%;background:linear-gradient(180deg,#059669,#0d9488);border-radius:4px 0 0 4px;}
.rec-panel h3{color:#065f46;font-size:1.05rem;font-weight:700;margin:0 0 .1rem .1rem;}
.rec-panel .rec-sub{font-size:.8rem;color:#047857;margin:0 0 .8rem .1rem;font-weight:400;}
.rec-asset{display:flex;align-items:flex-start;gap:.7rem;
  padding:.6rem .8rem;background:rgba(255,255,255,.65);border-radius:8px;
  margin-bottom:.45rem;border:1px solid #d1fae5;}
.rec-asset .rec-rank{background:#059669;color:#fff;border-radius:6px;
  min-width:1.6rem;height:1.6rem;display:flex;align-items:center;
  justify-content:center;font-size:.75rem;font-weight:700;flex-shrink:0;margin-top:.1rem;}
.rec-asset .rec-info{flex:1;min-width:0;}
.rec-asset .rec-name{font-weight:600;font-size:.88rem;color:#1e293b;}
.rec-asset .rec-rationale{font-size:.78rem;color:#4b5563;line-height:1.4;margin-top:.1rem;}
.rec-asset .rec-meta{font-size:.72rem;color:#6b7280;margin-top:.2rem;}
.rec-asset .rec-meta span{margin-right:.8rem;}

/* ---- Portfolio strip ---- */
.port-strip{display:flex;gap:.6rem;margin-bottom:1.2rem;flex-wrap:wrap;}
.port-chip{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;
  padding:.4rem .8rem;font-size:.75rem;color:#475569;font-weight:500;}
.port-chip b{color:#0f172a;}

/* ---- No-budget ---- */
.no-budget{background:#fef2f2;border:1px solid #fecaca;border-radius:10px;
  padding:1rem 1.3rem;margin-bottom:.8rem;}
.no-budget p{color:#991b1b;font-size:.9rem;margin:0;}

/* ---- Table section label ---- */
.table-label{font-size:.8rem;font-weight:600;color:#475569;margin-bottom:.4rem;}

/* ---- Charts ---- */
.stPlotlyChart{border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;}

/* ---- Footer ---- */
.footer{font-size:.75rem;color:#94a3b8;text-align:center;
  padding-top:.8rem;line-height:1.55;}

/* ---- Streamlit chrome ---- */
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
[data-testid="stDataFrame"]{border-radius:8px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CARBON_FACTORS: dict[str, float] = {
    "Ductile Iron": 1.80, "Steel": 2.10, "Concrete": 0.15,
    "HDPE": 2.50, "PVC": 3.00, "Cast Iron": 1.90,
    "Copper": 3.80, "FRP": 8.00,
}
MASS_PER_K_DOLLAR = 120
ASSET_TYPES = ["Pipe", "Pump", "Tank", "Clarifier", "Valve", "Hydrant"]
MATERIAL_OPTIONS = list(CARBON_FACTORS.keys())
W_AGE, W_COND, W_CONSEQ = 0.30, 0.40, 0.30


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_default_assets() -> pd.DataFrame:
    return pd.DataFrame({
        "Asset Name":          ["Main St Trunk Line", "Elm St Distribution", "North Pump Station",
                                "Central Reservoir",  "WWTP Clarifier #2",   "Oak Ave Service Line",
                                "South Booster Pump", "River Rd Main"],
        "Asset Type":          ["Pipe","Pipe","Pump","Tank","Clarifier","Pipe","Pump","Pipe"],
        "Age (yrs)":           [65, 42, 28, 55, 35, 78, 15, 50],
        "Condition (1-5)":     [4,  3,  2,  4,  3,  5,  1,  3],
        "Failure Consequence": [5,  3,  4,  5,  4,  2,  3,  4],
        "Upgrade Cost ($K)":   [1200, 450, 380, 2100, 780, 180, 290, 620],
        "Material":            ["Cast Iron","Ductile Iron","Steel","Concrete",
                                "Concrete","Cast Iron","Steel","HDPE"],
    })


def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    max_age = out["Age (yrs)"].max()
    out["age_norm"]    = out["Age (yrs)"] / max_age if max_age > 0 else 0
    out["cond_norm"]   = (out["Condition (1-5)"] - 1) / 4
    out["conseq_norm"] = (out["Failure Consequence"] - 1) / 4
    tw = W_AGE + W_COND + W_CONSEQ
    out["Priority Score"] = (
        (W_AGE * out["age_norm"] + W_COND * out["cond_norm"] + W_CONSEQ * out["conseq_norm"]) / tw * 100
    ).round(1)
    return out


def estimate_carbon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Carbon (t CO2e)"] = out.apply(
        lambda r: round(CARBON_FACTORS.get(r["Material"], 2.0) * r["Upgrade Cost ($K)"] * MASS_PER_K_DOLLAR / 1000, 1),
        axis=1,
    )
    return out


def select_within_budget(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Greedy knapsack: iterate by priority, pick each asset that still fits."""
    ranked = df.sort_values("Priority Score", ascending=False).copy()
    remaining = budget
    selected = []
    for idx, row in ranked.iterrows():
        if row["Upgrade Cost ($K)"] <= remaining:
            selected.append(True)
            remaining -= row["Upgrade Cost ($K)"]
        else:
            selected.append(False)
    ranked["Selected"] = selected
    return ranked


def generate_rationale(row, all_df: pd.DataFrame) -> str:
    """Short, decision-ready rationale for why an asset was selected."""
    reasons = []
    # Highest consequence check
    if row["Failure Consequence"] >= 4:
        reasons.append("high failure consequence")
    # Poor condition
    if row["Condition (1-5)"] >= 4:
        reasons.append("poor asset condition")
    # Old asset
    age_pct = row["Age (yrs)"] / all_df["Age (yrs)"].max() if all_df["Age (yrs)"].max() > 0 else 0
    if age_pct >= 0.7:
        reasons.append("aging infrastructure")
    # Cost efficiency
    if row["Priority Score"] > 0:
        efficiency = row["Priority Score"] / (row["Upgrade Cost ($K)"] / 100)
        median_eff = all_df.apply(
            lambda r: r["Priority Score"] / (r["Upgrade Cost ($K)"] / 100) if r["Priority Score"] > 0 else 0, axis=1
        ).median()
        if efficiency >= median_eff * 1.2:
            reasons.append("strong risk reduction per dollar")
    if not reasons:
        reasons.append("fits current budget efficiently")
    return reasons[0][0].upper() + reasons[0][1:] + ("; " + "; ".join(reasons[1:]) if len(reasons) > 1 else "")


# ======================= APP LAYOUT ========================================

# ---- Hero ----
st.markdown("""
<div class="hero">
    <h1>\U0001F4A7 Failure Forecasters</h1>
    <p>Prioritize water infrastructure upgrades by failure risk, cost, and embodied carbon
       \u2014 so every capital dollar goes where it matters most.</p>
</div>
""", unsafe_allow_html=True)

# ---- Step 1: Asset Inventory ----
st.markdown('<div class="step-label"><span class="step-num">1</span> Asset Inventory</div>',
            unsafe_allow_html=True)

if "assets" not in st.session_state:
    st.session_state.assets = get_default_assets()

edited_df = st.data_editor(
    st.session_state.assets, num_rows="dynamic", use_container_width=True,
    column_config={
        "Asset Name":          st.column_config.TextColumn("Asset Name", width="medium"),
        "Asset Type":          st.column_config.SelectboxColumn("Type", options=ASSET_TYPES, width="small"),
        "Age (yrs)":           st.column_config.NumberColumn("Age (yrs)", min_value=0, max_value=200, step=1, width="small"),
        "Condition (1-5)":     st.column_config.NumberColumn("Condition", min_value=1, max_value=5, step=1,
                                                             help="1 = Excellent \u00b7 5 = Failed", width="small"),
        "Failure Consequence": st.column_config.NumberColumn("Consequence", min_value=1, max_value=5, step=1,
                                                             help="1 = Minimal \u00b7 5 = Catastrophic", width="small"),
        "Upgrade Cost ($K)":   st.column_config.NumberColumn("Cost ($K)", min_value=0, step=10, format="$%d", width="small"),
        "Material":            st.column_config.SelectboxColumn("Material", options=MATERIAL_OPTIONS, width="small"),
    },
    key="asset_editor",
)
st.session_state.assets = edited_df

valid_df = edited_df.dropna(subset=["Asset Name","Asset Type","Age (yrs)",
                                     "Condition (1-5)","Failure Consequence",
                                     "Upgrade Cost ($K)","Material"])
if valid_df.empty:
    st.warning("Add at least one complete asset row to see results.")
    st.stop()


# ---- Step 2: Planning Constraints ----
st.markdown('<div class="step-label"><span class="step-num">2</span> Planning Constraints</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.2, 1, 0.8])
with c1:
    budget = st.number_input("Capital Budget ($K)", min_value=0, value=3000, step=100,
                             help="Maximum budget for this planning cycle.")
with c2:
    carbon_weight = st.slider("Sustainability Weight", 0.0, 1.0, 0.2, 0.05,
                              help="Higher values favor lower-carbon materials in ranking.")
with c3:
    favor_low_carbon = st.toggle("Favor low carbon", value=False,
                                 help="Penalize high-carbon material choices.")


# ---- Compute scores ----
scored_df = compute_priority(valid_df)
scored_df = estimate_carbon(scored_df)

if favor_low_carbon or carbon_weight > 0:
    max_c = scored_df["Carbon (t CO2e)"].max()
    if max_c > 0:
        penalty = (scored_df["Carbon (t CO2e)"] / max_c) * carbon_weight * 15
        scored_df["Priority Score"] = (scored_df["Priority Score"] - penalty).round(1)

ranked_df = select_within_budget(scored_df, budget)

# Labels
def label_rec(row):
    if row["Selected"]:
        return "Recommended This Cycle" if row["Priority Score"] >= 40 else "Near-Term Candidate"
    return "Defer to Future Cycle"

ranked_df["Recommendation"] = ranked_df.apply(label_rec, axis=1)
selected_df = ranked_df[ranked_df["Selected"]].copy()
deferred_df = ranked_df[~ranked_df["Selected"]].copy()


# ---- Step 3: Recommendations (CENTERPIECE) ----
st.markdown('<div class="step-label"><span class="step-num">3</span> Recommended Upgrade Plan</div>',
            unsafe_allow_html=True)

total_cost = selected_df["Upgrade Cost ($K)"].sum()
remaining  = budget - total_cost
total_carb = selected_df["Carbon (t CO2e)"].sum()
pct_used   = (total_cost / budget * 100) if budget > 0 else 0
high_risk  = len(selected_df[selected_df["Failure Consequence"] >= 4])

# KPI row (HTML for subtitle support)
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Assets Selected</div>
    <div class="kpi-value">{len(selected_df)} <span style="font-size:.85rem;font-weight:400;color:#64748b;">of {len(ranked_df)}</span></div>
    <div class="kpi-sub">{high_risk} high-risk asset{'s' if high_risk != 1 else ''} addressed</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Estimated Investment</div>
    <div class="kpi-value">${total_cost:,.0f}K</div>
    <div class="kpi-sub">{pct_used:.0f}% of budget allocated</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Budget Remaining</div>
    <div class="kpi-value">${remaining:,.0f}K</div>
    <div class="kpi-sub">Available for future cycles</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Embodied Carbon</div>
    <div class="kpi-value">{total_carb:,.1f} <span style="font-size:.85rem;">t CO\u2082e</span></div>
    <div class="kpi-sub">Across all selected upgrades</div>
  </div>
</div>
""", unsafe_allow_html=True)


# Recommendation panel with per-asset rationale
if not selected_df.empty:
    asset_cards = ""
    for rank_i, (_, row) in enumerate(selected_df.iterrows(), 1):
        rationale = generate_rationale(row, scored_df)
        asset_cards += f"""
        <div class="rec-asset">
            <div class="rec-rank">{rank_i}</div>
            <div class="rec-info">
                <div class="rec-name">{row['Asset Name']}</div>
                <div class="rec-rationale">{rationale}</div>
                <div class="rec-meta">
                    <span>{row['Asset Type']}</span>
                    <span>{int(row['Age (yrs)'])} yrs old</span>
                    <span>Condition {int(row['Condition (1-5)'])}/5</span>
                    <span>Score {row['Priority Score']:.0f}/100</span>
                    <span>${row['Upgrade Cost ($K)']:,.0f}K</span>
                    <span>{row['Carbon (t CO2e)']:.1f} t CO\u2082e</span>
                </div>
            </div>
        </div>"""

    st.markdown(f"""
    <div class="rec-panel">
        <h3>Recommended This Cycle</h3>
        <div class="rec-sub">{len(selected_df)} upgrade{'s' if len(selected_df) != 1 else ''} selected by priority within ${budget:,.0f}K budget</div>
        {asset_cards}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="no-budget">
        <p>No assets fit within the current budget. Increase the capital budget to see recommendations.</p>
    </div>
    """, unsafe_allow_html=True)


# Portfolio summary strip
if not selected_df.empty:
    avg_score = selected_df["Priority Score"].mean()
    st.markdown(f"""
    <div class="port-strip">
        <div class="port-chip"><b>{len(selected_df)}</b> assets selected</div>
        <div class="port-chip"><b>{pct_used:.0f}%</b> of budget used</div>
        <div class="port-chip"><b>{high_risk}</b> high-risk assets addressed</div>
        <div class="port-chip">Avg priority <b>{avg_score:.0f}</b>/100</div>
        <div class="port-chip"><b>{total_carb:,.1f}</b> t CO\u2082e total carbon</div>
    </div>
    """, unsafe_allow_html=True)


# ---- Full ranking table ----
st.markdown('<div class="table-label">Full Asset Ranking</div>', unsafe_allow_html=True)

display_cols = ["Asset Name", "Asset Type", "Priority Score",
                "Upgrade Cost ($K)", "Carbon (t CO2e)", "Recommendation"]

def style_rec(val):
    m = {
        "Recommended This Cycle": "background-color:#dcfce7;color:#166534;font-weight:600;",
        "Near-Term Candidate":    "background-color:#fef9c3;color:#854d0e;",
        "Defer to Future Cycle":  "background-color:#f1f5f9;color:#64748b;",
    }
    return m.get(val, "")

def style_row(row):
    if row["Recommendation"] == "Defer to Future Cycle":
        return ["opacity:0.55;"] * len(row)
    return [""] * len(row)

styled = (
    ranked_df[display_cols]
    .style
    .apply(style_row, axis=1)
    .map(style_rec, subset=["Recommendation"])
    .format({"Priority Score": "{:.1f}", "Upgrade Cost ($K)": "${:,.0f}", "Carbon (t CO2e)": "{:.1f}"})
)
st.dataframe(styled, use_container_width=True, hide_index=True)


# ---- Charts ----
st.markdown('<div class="step-label" style="margin-top:1.2rem;">Supporting Analysis</div>',
            unsafe_allow_html=True)

ch1, ch2 = st.columns(2)

REC_COLORS = {
    "Recommended This Cycle": "#059669",
    "Near-Term Candidate":    "#d97706",
    "Defer to Future Cycle":  "#cbd5e1",
}

# Chart 1 — Risk vs Cost with clear selected / deferred split
with ch1:
    fig1 = px.scatter(
        ranked_df, x="Upgrade Cost ($K)", y="Priority Score",
        size="Carbon (t CO2e)", color="Recommendation",
        hover_name="Asset Name",
        hover_data={"Priority Score": ":.1f", "Upgrade Cost ($K)": ":$,.0f",
                    "Carbon (t CO2e)": ":.1f t", "Recommendation": True},
        color_discrete_map=REC_COLORS,
        size_max=42,
        category_orders={"Recommendation": list(REC_COLORS.keys())},
    )
    # Budget line
    if budget > 0:
        fig1.add_vline(x=budget, line_dash="dot", line_color="#94a3b8", line_width=1,
                       annotation_text=f"Budget ${budget:,.0f}K",
                       annotation_font_size=10, annotation_font_color="#94a3b8")
    fig1.update_layout(
        title=dict(text="Risk vs. Cost", font=dict(size=14, color="#334155")),
        plot_bgcolor="#fafcfe", paper_bgcolor="#ffffff",
        font=dict(family="Inter,sans-serif", size=11),
        margin=dict(l=44, r=16, t=48, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30,
                    xanchor="center", x=0.5, font_size=9,
                    title_text=""),
        height=360,
    )
    fig1.update_xaxes(gridcolor="#edf2f7", title_text="Upgrade Cost ($K)", title_font_size=11)
    fig1.update_yaxes(gridcolor="#edf2f7", title_text="Priority Score", title_font_size=11)
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2 — Horizontal bar: selected vs deferred, color = recommendation
with ch2:
    bar_df = ranked_df.sort_values("Priority Score", ascending=True).copy()
    fig2 = go.Figure()
    for rec_label, color in REC_COLORS.items():
        subset = bar_df[bar_df["Recommendation"] == rec_label]
        if subset.empty:
            continue
        fig2.add_trace(go.Bar(
            y=subset["Asset Name"], x=subset["Priority Score"],
            orientation="h", name=rec_label,
            marker_color=color,
            text=subset["Priority Score"].apply(lambda v: f"{v:.0f}"),
            textposition="outside", textfont_size=10,
            hovertemplate="%{y}<br>Score: %{x:.1f}<br>Cost: $%{customdata[0]:,.0f}K<br>Carbon: %{customdata[1]:.1f} t CO₂e<extra></extra>",
            customdata=subset[["Upgrade Cost ($K)", "Carbon (t CO2e)"]].values,
        ))
    fig2.update_layout(
        title=dict(text="Upgrade Priority Ranking", font=dict(size=14, color="#334155")),
        barmode="stack",
        plot_bgcolor="#fafcfe", paper_bgcolor="#ffffff",
        font=dict(family="Inter,sans-serif", size=11),
        margin=dict(l=8, r=40, t=48, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30,
                    xanchor="center", x=0.5, font_size=9,
                    title_text=""),
        height=360,
    )
    fig2.update_xaxes(gridcolor="#edf2f7", title_text="Priority Score", title_font_size=11)
    fig2.update_yaxes(title_text="")
    st.plotly_chart(fig2, use_container_width=True)


# ---- Methodology (collapsed) ----
with st.expander("Scoring Methodology"):
    st.markdown(f"""
The priority score is a **transparent weighted index** — no black-box AI.

| Factor | Weight | Scale |
|--------|--------|-------|
| Asset Age | {W_AGE:.0%} | Normalized 0\u20131 by oldest asset |
| Condition | {W_COND:.0%} | 1 (Excellent) to 5 (Failed) |
| Failure Consequence | {W_CONSEQ:.0%} | 1 (Minimal) to 5 (Catastrophic) |

**Budget selection** uses a greedy knapsack approach: assets are ranked by priority and
selected in order while they fit the remaining budget. If an asset is too expensive, the
algorithm skips it and continues to the next, maximizing the number of high-priority
upgrades funded.

Carbon estimates use planning-level factors (kg CO\u2082e/kg) scaled by upgrade cost.
These are order-of-magnitude estimates for early capital planning, not detailed LCA values.
""")


# ---- Footer ----
st.markdown("---")
st.markdown("""
<div class="footer">
    <b>Built for</b> water utilities \u00b7 municipalities \u00b7 city planners \u00b7 public works departments<br>
    Supports capital improvement planning, deferred maintenance decisions, and sustainability-aware infrastructure management.<br><br>
    Failure Forecasters \u2014 CIVE 580c4 Final Project &nbsp;\u00b7&nbsp; Built with Streamlit & Plotly &nbsp;\u00b7&nbsp; Sample data for demonstration only
</div>
""", unsafe_allow_html=True)
