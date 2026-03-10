"""
Failure Forecasters -- AI-Driven Water Infrastructure Upgrade Planner
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
# CSS -- minimal, only for elements that need it (hero, KPIs, footer)
# Native Streamlit components handle everything else.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Global */
.block-container {padding-top:1.2rem;padding-bottom:2rem;max-width:1100px;}

/* Hero banner */
.hero{background:linear-gradient(135deg,#0b3d54 0%,#146b8a 55%,#1a9d8f 100%);
  border-radius:14px;padding:1rem 2rem .9rem;color:#fff;margin-bottom:1.2rem;}
.hero h1{font-size:1.45rem;font-weight:700;margin:0 0 .15rem;letter-spacing:-.4px;}
.hero p{font-size:.82rem;opacity:.88;margin:0;line-height:1.4;max-width:700px;}

/* Step labels */
.step-label{font-size:.7rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:#64748b;margin-top:1.5rem;margin-bottom:.55rem;
  display:flex;align-items:center;gap:.45rem;}
.step-num{background:#0b3d54;color:#fff;border-radius:5px;
  padding:.1rem .45rem;font-size:.65rem;letter-spacing:.04em;}

/* KPI row */
.kpi-row{display:flex;gap:.75rem;margin-bottom:.6rem;}
.kpi{flex:1;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:.85rem 1rem .7rem;min-width:0;}
.kpi .kpi-label{font-size:.65rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:#64748b;margin-bottom:.15rem;}
.kpi .kpi-value{font-size:1.3rem;font-weight:700;color:#0f172a;line-height:1.2;}
.kpi .kpi-sub{font-size:.72rem;color:#94a3b8;margin-top:.15rem;line-height:1.3;}

/* Plotly chart containers */
.stPlotlyChart{border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;}

/* Footer */
.footer{font-size:.75rem;color:#94a3b8;text-align:center;
  padding-top:.8rem;line-height:1.55;}

/* Streamlit chrome */
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
    for _, row in ranked.iterrows():
        if row["Upgrade Cost ($K)"] <= remaining:
            selected.append(True)
            remaining -= row["Upgrade Cost ($K)"]
        else:
            selected.append(False)
    ranked["Selected"] = selected
    return ranked


def generate_rationale(row, all_df: pd.DataFrame) -> str:
    """Professional, decision-ready rationale for why an asset was selected."""
    phrases = []
    age_pct = row["Age (yrs)"] / all_df["Age (yrs)"].max() if all_df["Age (yrs)"].max() > 0 else 0

    # Build natural-language fragments
    if row["Condition (1-5)"] >= 4 and row["Failure Consequence"] >= 4:
        phrases.append(f"Critical condition (rated {int(row['Condition (1-5)'])}/5) with high failure consequence")
    elif row["Failure Consequence"] >= 4:
        phrases.append("High failure consequence requiring proactive replacement")
    elif row["Condition (1-5)"] >= 4:
        phrases.append(f"Poor condition (rated {int(row['Condition (1-5)'])}/5) indicating near-term failure risk")

    if age_pct >= 0.7 and not phrases:
        phrases.append(f"Aging asset at {int(row['Age (yrs)'])} years approaching end of useful life")
    elif age_pct >= 0.7:
        phrases.append(f"nearing end of useful life at {int(row['Age (yrs)'])} years")

    if row["Priority Score"] > 0:
        efficiency = row["Priority Score"] / (row["Upgrade Cost ($K)"] / 100)
        median_eff = all_df.apply(
            lambda r: r["Priority Score"] / (r["Upgrade Cost ($K)"] / 100) if r["Priority Score"] > 0 else 0, axis=1
        ).median()
        if efficiency >= median_eff * 1.2:
            phrases.append("cost-effective risk reduction")

    if not phrases:
        phrases.append("Fits within current budget with favorable risk profile")
        return phrases[0]

    # Join: capitalize first phrase, lowercase joiners
    result = phrases[0]
    if len(phrases) > 1:
        result += "; " + "; ".join(phrases[1:])
    return result


# ======================= APP LAYOUT ========================================

# ---- Hero ----
st.markdown("""
<div class="hero">
    <h1>\U0001F4A7 Failure Forecasters</h1>
    <p>Prioritize water infrastructure upgrades by failure risk, cost, and embodied
       carbon -- so every capital dollar goes where it matters most.</p>
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
                                                             help="1 = Excellent, 5 = Failed", width="small"),
        "Failure Consequence": st.column_config.NumberColumn("Consequence", min_value=1, max_value=5, step=1,
                                                             help="1 = Minimal, 5 = Catastrophic", width="small"),
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

def label_rec(row):
    if row["Selected"]:
        return "Recommended This Cycle" if row["Priority Score"] >= 40 else "Near-Term Candidate"
    return "Defer to Future Cycle"

ranked_df["Recommendation"] = ranked_df.apply(label_rec, axis=1)
selected_df = ranked_df[ranked_df["Selected"]].copy()
deferred_df = ranked_df[~ranked_df["Selected"]].copy()


# ---- Step 3: Recommendations ----
st.markdown('<div class="step-label"><span class="step-num">3</span> Recommended Upgrade Plan</div>',
            unsafe_allow_html=True)

total_cost = selected_df["Upgrade Cost ($K)"].sum()
remaining  = budget - total_cost
total_carb = selected_df["Carbon (t CO2e)"].sum()
pct_used   = (total_cost / budget * 100) if budget > 0 else 0
high_risk  = len(selected_df[selected_df["Failure Consequence"] >= 4])

# KPI row
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
    <div class="kpi-value">{total_carb:,.1f} t CO2e</div>
    <div class="kpi-sub">Across all selected upgrades</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---- Portfolio summary sentence ----
if not selected_df.empty:
    avg_score = selected_df["Priority Score"].mean()
    st.markdown(
        f"This plan directs **${total_cost:,.0f}K** ({pct_used:.0f}% of budget) toward "
        f"**{len(selected_df)} assets** averaging a priority score of **{avg_score:.0f}/100**, "
        f"addressing **{high_risk} high-risk** asset{'s' if high_risk != 1 else ''} "
        f"with an estimated **{total_carb:,.1f} t CO2e** of embodied carbon."
    )

# ---- Recommended portfolio (native Streamlit) ----
if not selected_df.empty:
    st.markdown(f"#### Recommended This Cycle")

    # Render asset cards in a 2-column grid with green accent
    rows = list(selected_df.iterrows())
    for i in range(0, len(rows), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(rows):
                break
            _, row = rows[i + j]
            rank_num = i + j + 1
            rationale = generate_rationale(row, scored_df)
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<span style="background:#059669;color:#fff;padding:2px 8px;'
                        f'border-radius:4px;font-size:.7rem;font-weight:600;'
                        f'letter-spacing:.03em;">PRIORITY #{rank_num}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{row['Asset Name']}**")
                    st.markdown(
                        f"<span style='color:#4b5563;font-size:.88rem;'>{rationale}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"{row['Asset Type']}  &middot;  "
                        f"{int(row['Age (yrs)'])} yrs  &middot;  "
                        f"Condition {int(row['Condition (1-5)'])}/5  &middot;  "
                        f"Score **{row['Priority Score']:.0f}**/100  &middot;  "
                        f"**${row['Upgrade Cost ($K)']:,.0f}K**  &middot;  "
                        f"{row['Carbon (t CO2e)']:.1f} t CO2e"
                    )
else:
    st.info("No assets fit within the current budget. Increase the capital budget to see recommendations.")


# ---- Full ranking table ----
st.markdown("**Full Asset Ranking**")

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

# Chart 1 -- Risk vs Cost scatter (improved marker clarity)
with ch1:
    fig1 = px.scatter(
        ranked_df, x="Upgrade Cost ($K)", y="Priority Score",
        size="Carbon (t CO2e)", color="Recommendation",
        hover_name="Asset Name",
        hover_data={"Priority Score": ":.1f", "Upgrade Cost ($K)": ":$,.0f",
                    "Carbon (t CO2e)": ":.1f", "Recommendation": False},
        color_discrete_map=REC_COLORS,
        size_max=45,
        category_orders={"Recommendation": list(REC_COLORS.keys())},
    )
    # Add marker borders for clarity
    fig1.update_traces(marker=dict(line=dict(width=1.5, color="white"), opacity=0.9))
    fig1.update_layout(
        title=dict(text="Risk vs. Cost", font=dict(size=14, color="#334155")),
        plot_bgcolor="#fafcfe", paper_bgcolor="#ffffff",
        font=dict(family="Inter,system-ui,sans-serif", size=11),
        margin=dict(l=44, r=16, t=48, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30,
                    xanchor="center", x=0.5, font_size=9, title_text=""),
        height=370,
    )
    fig1.update_xaxes(gridcolor="#edf2f7", title_text="Upgrade Cost ($K)", title_font_size=11)
    fig1.update_yaxes(gridcolor="#edf2f7", title_text="Priority Score", title_font_size=11)
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2 -- Investment by Asset (sorted: recommended first by priority, then deferred)
with ch2:
    # Sort: selected assets by priority descending, then deferred by priority descending
    # Reverse for horizontal bar (bottom = highest priority)
    ordered = pd.concat([
        deferred_df.sort_values("Priority Score", ascending=False),
        selected_df.sort_values("Priority Score", ascending=False),
    ])
    bar_colors = [REC_COLORS.get(r, "#cbd5e1") for r in ordered["Recommendation"]]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=ordered["Asset Name"],
        x=ordered["Upgrade Cost ($K)"],
        orientation="h",
        marker_color=bar_colors,
        marker_line=dict(width=0),
        text=ordered.apply(
            lambda r: f"${r['Upgrade Cost ($K)']:,.0f}K  (Score: {r['Priority Score']:.0f})", axis=1
        ),
        textposition="outside",
        textfont=dict(size=9, color="#475569"),
        hovertemplate="%{y}<br>Cost: $%{x:,.0f}K<extra></extra>",
        showlegend=False,
    ))

    fig2.update_layout(
        title=dict(text="Investment by Asset", font=dict(size=14, color="#334155")),
        plot_bgcolor="#fafcfe", paper_bgcolor="#ffffff",
        font=dict(family="Inter,system-ui,sans-serif", size=11),
        margin=dict(l=8, r=70, t=48, b=32),
        height=370,
    )
    fig2.update_xaxes(gridcolor="#edf2f7", title_text="Upgrade Cost ($K)", title_font_size=11)
    fig2.update_yaxes(title_text="")
    st.plotly_chart(fig2, use_container_width=True)


# ---- Methodology (collapsed) ----
with st.expander("Scoring Methodology"):
    st.markdown(f"""
The priority score is a **transparent weighted index** -- no black-box AI.

| Factor | Weight | Scale |
|--------|--------|-------|
| Asset Age | {W_AGE:.0%} | Normalized 0-1 by oldest asset |
| Condition | {W_COND:.0%} | 1 (Excellent) to 5 (Failed) |
| Failure Consequence | {W_CONSEQ:.0%} | 1 (Minimal) to 5 (Catastrophic) |

**Budget selection** uses a greedy knapsack approach: assets are ranked by priority and
selected in order while they fit the remaining budget. If an asset is too expensive, the
algorithm skips it and continues to the next, maximizing the number of high-priority
upgrades funded.

Carbon estimates use planning-level factors (kg CO2e/kg) scaled by upgrade cost.
These are order-of-magnitude estimates for early capital planning, not detailed LCA values.
""")


# ---- Footer ----
st.markdown("---")
st.markdown("""
<div class="footer">
    <b>Built for</b> water utilities &middot; municipalities &middot; city planners &middot; public works departments<br>
    Supports capital improvement planning, deferred maintenance decisions, and sustainability-aware infrastructure management.<br><br>
    Failure Forecasters -- CIVE 580c4 Final Project &nbsp;&middot;&nbsp; Built with Streamlit & Plotly &nbsp;&middot;&nbsp; Sample data for demonstration only
</div>
""", unsafe_allow_html=True)
