"""
Failure Forecasters — AI-Driven Water Infrastructure Upgrade Planner
A decision-support tool for water utilities, municipalities, and local governments
to prioritize infrastructure asset replacements based on risk, cost, and sustainability.

CIVE 580c4 Final Project
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from fpdf import FPDF

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Failure Forecasters | Water Infrastructure Planner",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — Modern dashboard aesthetic
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Reset & Global ── */
.block-container{padding-top:.8rem;padding-bottom:2rem;max-width:1200px;}
html, body, [class*="css"]{font-family:'Inter',system-ui,-apple-system,sans-serif;}

/* ── Hero banner ── */
.hero{
  background:linear-gradient(135deg,#0c2d48 0%,#145374 40%,#0d9488 100%);
  border-radius:16px;padding:1.1rem 2rem 1rem;color:#fff;
  margin-bottom:.6rem;display:flex;align-items:center;gap:1.4rem;
  box-shadow:0 4px 24px rgba(12,45,72,.25);position:relative;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;top:-40%;right:-10%;width:300px;height:300px;
  background:radial-gradient(circle,rgba(255,255,255,.06) 0%,transparent 70%);
  border-radius:50%;
}
.hero h1{font-size:1.45rem;font-weight:800;margin:0;letter-spacing:-.4px;
  white-space:nowrap;position:relative;}
.hero p{font-size:.82rem;opacity:.88;margin:0;line-height:1.4;position:relative;}
.hero-badge{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.2);
  border-radius:20px;padding:.15rem .7rem;font-size:.62rem;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;display:inline-block;margin-top:.25rem;}

/* ── Section headers ── */
.section-hdr{font-size:.72rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.12em;color:#64748b;margin:1.2rem 0 .5rem;
  display:flex;align-items:center;gap:.5rem;}
.section-hdr .num{background:linear-gradient(135deg,#0c2d48,#145374);
  color:#fff;border-radius:6px;padding:.12rem .5rem;font-size:.64rem;}

/* ── KPI cards ── */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:.65rem;margin-bottom:.8rem;}
.kpi{background:#fff;border:1px solid #e8ecf1;border-radius:12px;
  padding:.8rem 1rem .65rem;transition:box-shadow .2s,transform .2s;}
.kpi:hover{box-shadow:0 4px 16px rgba(0,0,0,.06);transform:translateY(-1px);}
.kpi-label{font-size:.62rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:#8896a6;margin-bottom:.1rem;}
.kpi-value{font-size:1.35rem;font-weight:800;color:#0f172a;line-height:1.2;}
.kpi-sub{font-size:.7rem;color:#94a3b8;margin-top:.12rem;line-height:1.3;}
.kpi-accent{border-top:3px solid #0d9488;}

/* ── Asset cards ── */
.asset-card{border-left:4px solid #059669;padding-left:.75rem;margin-bottom:.15rem;}
.asset-rank{background:linear-gradient(135deg,#059669,#0d9488);color:#fff;
  padding:1px 8px;border-radius:4px;font-size:.6rem;font-weight:700;
  letter-spacing:.05em;display:inline-block;}
.asset-name{font-size:1rem;font-weight:700;color:#0f172a;margin-top:.15rem;}
.defer-card{border-left-color:#d97706;}
.defer-rank{background:linear-gradient(135deg,#d97706,#f59e0b);}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:2px solid #e8ecf1;}
.stTabs [data-baseweb="tab"]{font-size:.78rem;font-weight:600;
  letter-spacing:.02em;padding:.5rem 1.2rem;border-radius:8px 8px 0 0;}

/* ── Charts ── */
.stPlotlyChart{border:1px solid #e8ecf1;border-radius:12px;overflow:hidden;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:#f8fafc;border-right:1px solid #e8ecf1;}
section[data-testid="stSidebar"] .stMarkdown h3{font-size:.85rem;font-weight:700;
  color:#334155;letter-spacing:-.01em;}

/* ── Data table ── */
[data-testid="stDataFrame"]{border-radius:10px;}

/* ── Footer ── */
.footer{font-size:.75rem;color:#94a3b8;text-align:center;
  padding-top:.8rem;line-height:1.6;}

/* ── Chrome ── */
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════════

# Embodied carbon factors (kg CO2e per kg of material)
CARBON_FACTORS: dict[str, float] = {
    "Ductile Iron": 1.80, "Steel": 2.10, "Concrete": 0.15,
    "HDPE": 2.50, "PVC": 3.00, "Cast Iron": 1.90,
    "Copper": 3.80, "FRP": 8.00,
}

MASS_PER_K_DOLLAR = 120  # kg of material per $1K of upgrade cost (planning estimate)

ASSET_TYPES = ["Pipe", "Pump", "Tank", "Clarifier", "Valve", "Hydrant"]
MATERIAL_OPTIONS = list(CARBON_FACTORS.keys())

# Default priority weights
DEFAULT_W_AGE = 0.25
DEFAULT_W_COND = 0.35
DEFAULT_W_CONSEQ = 0.25
DEFAULT_W_RUL = 0.15

# Expected service life (years) by (Asset Type, Material)
# Sources: AWWA, EPA infrastructure guidelines
EXPECTED_SERVICE_LIFE: dict[tuple[str, str], int] = {
    ("Pipe", "Cast Iron"): 90, ("Pipe", "Ductile Iron"): 90,
    ("Pipe", "Concrete"): 85, ("Pipe", "HDPE"): 75,
    ("Pipe", "PVC"): 80, ("Pipe", "Steel"): 65,
    ("Pipe", "Copper"): 70, ("Pipe", "FRP"): 40,
    ("Pump", "Steel"): 20, ("Pump", "Cast Iron"): 25,
    ("Pump", "Ductile Iron"): 22, ("Pump", "Copper"): 20,
    ("Tank", "Concrete"): 65, ("Tank", "Steel"): 50,
    ("Tank", "FRP"): 35, ("Tank", "HDPE"): 40,
    ("Clarifier", "Concrete"): 50, ("Clarifier", "Steel"): 40,
    ("Valve", "Cast Iron"): 45, ("Valve", "Ductile Iron"): 50,
    ("Valve", "Steel"): 40,
    ("Hydrant", "Cast Iron"): 55, ("Hydrant", "Ductile Iron"): 60,
}

# Fallback expected life by asset type
DEFAULT_SERVICE_LIFE: dict[str, int] = {
    "Pipe": 75, "Pump": 20, "Tank": 60, "Clarifier": 45, "Valve": 40, "Hydrant": 50,
}

# Emergency/failure cost multiplier by consequence level
FAILURE_COST_MULTIPLIER = {1: 1.5, 2: 2.0, 3: 3.0, 4: 4.5, 5: 6.0}

# Default cost ranges and common materials by asset type
ASSET_DEFAULTS: dict[str, dict] = {
    "Pipe":      {"cost": 500,  "materials": ["Ductile Iron", "HDPE", "PVC", "Cast Iron", "Concrete", "Steel", "Copper"]},
    "Pump":      {"cost": 350,  "materials": ["Steel", "Cast Iron", "Ductile Iron"]},
    "Tank":      {"cost": 1500, "materials": ["Concrete", "Steel", "FRP", "HDPE"]},
    "Clarifier": {"cost": 800,  "materials": ["Concrete", "Steel"]},
    "Valve":     {"cost": 120,  "materials": ["Ductile Iron", "Cast Iron", "Steel"]},
    "Hydrant":   {"cost": 80,   "materials": ["Cast Iron", "Ductile Iron"]},
}

# Material alternatives: for each material, suggest lower-carbon options by asset type
MATERIAL_ALTERNATIVES: dict[str, list[dict]] = {
    "Cast Iron":    [{"alt": "Ductile Iron", "note": "Similar strength, ~5% less carbon"},
                     {"alt": "HDPE", "note": "~32% less carbon, flexible, corrosion-free"}],
    "Steel":        [{"alt": "Ductile Iron", "note": "~14% less carbon, better corrosion resistance"},
                     {"alt": "HDPE", "note": "~19% more carbon but no corrosion"}],
    "PVC":          [{"alt": "HDPE", "note": "~17% less carbon, more flexible"}],
    "Copper":       [{"alt": "HDPE", "note": "~34% less carbon, lower cost"},
                     {"alt": "PVC", "note": "~21% less carbon"}],
    "FRP":          [{"alt": "HDPE", "note": "~69% less carbon"},
                     {"alt": "Concrete", "note": "~98% less carbon for large structures"}],
    "Concrete":     [],
    "HDPE":         [{"alt": "Concrete", "note": "~94% less carbon for large structures"}],
    "Ductile Iron": [{"alt": "HDPE", "note": "~39% more carbon but corrosion-free"}],
}

# Recommendation colors
REC_COLORS = {
    "Recommended This Cycle": "#059669",
    "Near-Term Candidate":    "#d97706",
    "Defer to Future Cycle":  "#94a3b8",
}

# Plotly chart defaults
CHART_LAYOUT = dict(
    plot_bgcolor="#fafcfe", paper_bgcolor="#ffffff",
    font=dict(family="Inter,system-ui,sans-serif", size=11),
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_expected_life(asset_type: str, material: str) -> int:
    return EXPECTED_SERVICE_LIFE.get(
        (asset_type, material),
        DEFAULT_SERVICE_LIFE.get(asset_type, 50),
    )


def get_default_assets() -> pd.DataFrame:
    # 5 demo assets — each tells a different story for presentation:
    #  1. Transmission Main A     — old cast iron, failing, catastrophic consequence (obvious #1)
    #  2. Booster Pump Station 1  — critical pump at 95% of 20-yr life, triggers RUL warning
    #  3. Storage Reservoir B     — big expensive tank, condition 4, high consequence
    #  4. WWTP Clarifier #2      — steel clarifier at 95% of 40-yr life, moderate consequence
    #  5. Distribution Line C    — younger copper line, low consequence, shows deferral logic
    # Data tuned so positions 2-4 swap under different weight profiles (sensitivity chart)
    return pd.DataFrame({
        "Asset Name":          ["Transmission Main A", "Booster Pump Station 1",
                                "Storage Reservoir B",  "WWTP Clarifier #2",
                                "Distribution Line C"],
        "Asset Type":          ["Pipe", "Pump", "Tank", "Clarifier", "Pipe"],
        "Age (yrs)":           [72, 19, 45, 38, 25],
        "Condition (1-5)":     [4,  3,  4,  3,  2],
        "Failure Consequence": [5,  4,  5,  3,  2],
        "Upgrade Cost ($K)":   [1200, 380, 2100, 780, 180],
        "Material":            ["Cast Iron", "Steel", "Concrete", "Steel", "Copper"],
        "Critical Asset":      [True, True, True, False, False],
    })


def compute_scores(df: pd.DataFrame, w_age: float, w_cond: float,
                   w_conseq: float, w_rul: float, carbon_weight: float,
                   favor_low_carbon: bool) -> pd.DataFrame:
    """Compute all derived scores: RUL, priority, risk, carbon, failure cost."""
    out = df.copy()

    # ── Remaining Useful Life ──
    out["Expected Life (yrs)"] = out.apply(
        lambda r: get_expected_life(r["Asset Type"], r["Material"]), axis=1)
    out["RUL (yrs)"] = (out["Expected Life (yrs)"] - out["Age (yrs)"]).clip(lower=0)
    out["Life Used (%)"] = ((out["Age (yrs)"] / out["Expected Life (yrs)"]) * 100).round(1)

    # ── Normalized factors (0-1) ──
    max_age = out["Age (yrs)"].max()
    out["age_norm"]    = out["Age (yrs)"] / max_age if max_age > 0 else 0
    out["cond_norm"]   = (out["Condition (1-5)"] - 1) / 4
    out["conseq_norm"] = (out["Failure Consequence"] - 1) / 4
    max_rul = out["RUL (yrs)"].max()
    out["rul_norm"] = 1 - (out["RUL (yrs)"] / max_rul) if max_rul > 0 else 0

    # ── Criticality boost ──
    crit_boost = out["Critical Asset"].apply(lambda c: 0.10 if c else 0)

    # ── Priority Score ──
    tw = w_age + w_cond + w_conseq + w_rul
    if tw == 0:
        tw = 1
    raw_priority = (
        (w_age * out["age_norm"] + w_cond * out["cond_norm"]
         + w_conseq * out["conseq_norm"] + w_rul * out["rul_norm"]) / tw
        + crit_boost
    ) * 100
    out["Priority Score"] = raw_priority.clip(upper=100).round(1)

    # ── Risk Score ──
    out["Risk Score"] = ((out["Condition (1-5)"] / 5) * (out["Failure Consequence"] / 5) * 100).round(1)

    # ── Carbon ──
    out["Carbon (t CO2e)"] = out.apply(
        lambda r: round(CARBON_FACTORS.get(r["Material"], 2.0)
                        * r["Upgrade Cost ($K)"] * MASS_PER_K_DOLLAR / 1000, 1),
        axis=1,
    )

    # ── Carbon penalty ──
    if favor_low_carbon or carbon_weight > 0:
        max_c = out["Carbon (t CO2e)"].max()
        if max_c > 0:
            penalty = (out["Carbon (t CO2e)"] / max_c) * carbon_weight * 15
            out["Priority Score"] = (out["Priority Score"] - penalty).clip(lower=0).round(1)

    # ── Failure cost (cost of NOT upgrading) ──
    out["Failure Cost ($K)"] = out.apply(
        lambda r: round(r["Upgrade Cost ($K)"] * FAILURE_COST_MULTIPLIER.get(
            int(r["Failure Consequence"]), 3.0), 0),
        axis=1,
    )
    out["Deferral Risk ($K)"] = out["Failure Cost ($K)"] - out["Upgrade Cost ($K)"]

    # ── Cost efficiency ──
    out["Risk Reduction per $100K"] = (
        out["Risk Score"] / (out["Upgrade Cost ($K)"] / 100)
    ).round(2)

    return out


def select_within_budget(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Greedy knapsack: pick highest-priority assets that fit the budget."""
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


def label_recommendation(row) -> str:
    if row["Selected"]:
        return "Recommended This Cycle"
    # Not selected: near-term if score is decent, otherwise defer
    return "Near-Term Candidate" if row["Priority Score"] >= 40 else "Defer to Future Cycle"


def generate_rationale(row, all_df: pd.DataFrame) -> str:
    """Decision-ready rationale for asset prioritization."""
    age = int(row["Age (yrs)"])
    cond = int(row["Condition (1-5)"])
    conseq = int(row["Failure Consequence"])
    rul = int(row.get("RUL (yrs)", 0))
    life_pct = row.get("Life Used (%)", 0)
    is_critical = row.get("Critical Asset", False)

    parts = []

    # Primary driver
    if cond >= 4 and conseq >= 4:
        parts.append(f"Condition rated {cond}/5 with catastrophic failure consequence")
    elif life_pct >= 100:
        parts.append(f"Past expected service life ({age} yrs vs {int(row.get('Expected Life (yrs)', age))} yr design life)")
    elif conseq >= 4:
        parts.append("High failure consequence warrants proactive replacement")
    elif cond >= 4:
        parts.append(f"Deteriorated condition ({cond}/5) signals near-term failure risk")
    elif life_pct >= 80:
        parts.append(f"{int(life_pct)}% of expected service life consumed ({rul} yrs remaining)")

    if is_critical and "Critical" not in " ".join(parts):
        parts.append("critical single-point-of-failure asset")

    # Cost efficiency context — only mention if the asset actually has meaningful risk
    if row["Priority Score"] > 0 and row["Upgrade Cost ($K)"] > 0 and row["Risk Score"] >= 20:
        eff = row["Risk Reduction per $100K"]
        med = all_df["Risk Reduction per $100K"].median()
        if eff >= med * 1.3:
            parts.append("strong cost-to-risk-reduction ratio")

    # Low-risk, low-cost assets: be honest about why they were selected
    if not parts:
        if conseq <= 2 and cond <= 2:
            return "Low-risk asset funded with remaining budget capacity"
        return "Favorable risk profile within available budget"
    return "; ".join(parts[:3])


def get_material_alternatives(row, all_df: pd.DataFrame) -> list[dict]:
    """Suggest lower-carbon material alternatives for an asset."""
    current = row["Material"]
    asset_type = row["Asset Type"]
    cost = row["Upgrade Cost ($K)"]
    alts = MATERIAL_ALTERNATIVES.get(current, [])
    compatible = ASSET_DEFAULTS.get(asset_type, {}).get("materials", [])

    suggestions = []
    current_carbon = CARBON_FACTORS.get(current, 2.0) * cost * MASS_PER_K_DOLLAR / 1000
    for a in alts:
        if a["alt"] in compatible:
            alt_carbon = CARBON_FACTORS.get(a["alt"], 2.0) * cost * MASS_PER_K_DOLLAR / 1000
            savings = current_carbon - alt_carbon
            suggestions.append({
                "material": a["alt"],
                "note": a["note"],
                "carbon_savings": round(savings, 1),
                "new_carbon": round(alt_carbon, 1),
            })
    return suggestions


def multi_year_plan(df: pd.DataFrame, num_years: int, yearly_budget: float,
                    w_age: float, w_cond: float, w_conseq: float,
                    w_rul: float, carbon_weight: float,
                    favor_low_carbon: bool) -> list[pd.DataFrame]:
    """Allocate assets across multiple budget years with condition degradation."""
    remaining = df.copy()
    year_plans = []

    for year in range(1, num_years + 1):
        if remaining.empty:
            year_plans.append(pd.DataFrame())
            continue

        # Re-score with degraded conditions
        scored = compute_scores(remaining, w_age, w_cond, w_conseq, w_rul,
                                carbon_weight, favor_low_carbon)
        selected = select_within_budget(scored, yearly_budget)

        year_selected = selected[selected["Selected"]].copy()
        year_selected["Plan Year"] = year
        year_plans.append(year_selected)

        # Remove selected assets from the pool
        selected_names = set(year_selected["Asset Name"])
        remaining = remaining[~remaining["Asset Name"].isin(selected_names)].copy()

        # Degrade condition of deferred assets (closer to end of life = faster degradation)
        if not remaining.empty:
            remaining["Age (yrs)"] = remaining["Age (yrs)"] + 1
            life_frac = remaining.apply(
                lambda r: r["Age (yrs)"] / get_expected_life(r["Asset Type"], r["Material"]),
                axis=1)
            degrade = life_frac.apply(lambda f: 0.5 if f >= 1.0 else (0.3 if f >= 0.8 else 0.1))
            remaining["Condition (1-5)"] = (remaining["Condition (1-5)"] + degrade).clip(upper=5).round(1)

    return year_plans


def compute_system_health(df: pd.DataFrame) -> float:
    """Weighted system health score (0-100, higher = healthier).
    Weight by consequence so critical assets matter more."""
    if df.empty:
        return 100.0
    weights = df["Failure Consequence"]
    conditions = df["Condition (1-5)"]
    # Convert condition (1=best, 5=worst) to health (100=best, 0=worst)
    health_per_asset = (1 - (conditions - 1) / 4) * 100
    return round(np.average(health_per_asset, weights=weights), 1)


def generate_csv_template() -> str:
    """Generate a CSV template for asset upload."""
    example = pd.DataFrame({
        "Asset Name": ["Example Pipe 1", "Example Pump 1"],
        "Asset Type": ["Pipe", "Pump"],
        "Age (yrs)": [50, 15],
        "Condition (1-5)": [3, 2],
        "Failure Consequence": [4, 3],
        "Upgrade Cost ($K)": [500, 300],
        "Material": ["Ductile Iron", "Steel"],
        "Critical Asset": [True, False],
    })
    return example.to_csv(index=False)


def generate_pdf(ranked_df: pd.DataFrame, selected_df: pd.DataFrame,
                  deferred_df: pd.DataFrame, budget: float,
                  scored_df: pd.DataFrame, health_before: float,
                  health_after: float) -> bytes:
    """Generate a PDF summary report."""
    total_cost = selected_df["Upgrade Cost ($K)"].sum()
    total_carbon = selected_df["Carbon (t CO2e)"].sum()
    risk_reduced = selected_df["Risk Score"].sum() if not selected_df.empty else 0
    total_risk = ranked_df["Risk Score"].sum()
    risk_pct = (risk_reduced / total_risk * 100) if total_risk > 0 else 0
    avoided_cost = selected_df["Deferral Risk ($K)"].sum() if not selected_df.empty else 0

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(12, 45, 72)
    pdf.cell(0, 12, "Failure Forecasters", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, "Capital Improvement Plan Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%B %d, %Y')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── KPI Summary ──
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "Plan Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)

    kpi_data = [
        ("Capital Budget", f"${budget:,.0f}K"),
        ("Planned Investment", f"${total_cost:,.0f}K ({total_cost / budget * 100 if budget > 0 else 0:.0f}% of budget)"),
        ("Budget Remaining", f"${budget - total_cost:,.0f}K"),
        ("Assets Selected", f"{len(selected_df)} of {len(ranked_df)}"),
        ("System Risk Reduction", f"{risk_pct:.0f}%"),
        ("Failure Cost Avoided", f"${avoided_cost:,.0f}K"),
        ("System Health", f"{health_before:.0f}/100 -> {health_after:.0f}/100 (+{health_after - health_before:.0f} pts)"),
        ("Embodied Carbon", f"{total_carbon:,.1f} t CO2e"),
    ]
    for label, value in kpi_data:
        pdf.cell(65, 6, f"  {label}:", new_x="RIGHT")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

    pdf.ln(4)

    # ── Recommended Upgrades Table ──
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "Recommended Upgrades", new_x="LMARGIN", new_y="NEXT")

    # Table header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(51, 65, 85)
    col_widths = [44, 18, 14, 12, 14, 18, 18, 18, 34]
    headers = ["Asset", "Type", "Age", "Cond", "Conseq", "Cost($K)", "Risk", "CO2e(t)", "Recommendation"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    for _, row in ranked_df.iterrows():
        rec = row.get("Recommendation", "")
        if rec == "Recommended This Cycle":
            pdf.set_fill_color(220, 252, 231)
        elif rec == "Near-Term Candidate":
            pdf.set_fill_color(254, 249, 195)
        else:
            pdf.set_fill_color(241, 245, 249)

        vals = [
            str(row["Asset Name"])[:22],
            str(row["Asset Type"]),
            str(int(row["Age (yrs)"])),
            str(int(row["Condition (1-5)"])),
            str(int(row["Failure Consequence"])),
            f"${row['Upgrade Cost ($K)']:,.0f}",
            f"{row['Risk Score']:.0f}",
            f"{row['Carbon (t CO2e)']:.1f}",
            rec[:18],
        ]
        for i, v in enumerate(vals):
            align = "L" if i == 0 else "C"
            pdf.cell(col_widths[i], 5.5, v, border=1, fill=True, align=align)
        pdf.ln()

    pdf.ln(4)

    # ── Rationales ──
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "Prioritization Rationale", new_x="LMARGIN", new_y="NEXT")

    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        rationale = generate_rationale(row, scored_df)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(5, 150, 105)
        pdf.cell(0, 6, f"  #{i}  {row['Asset Name']}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(0, 5, f"      {rationale}")
        pdf.ln(1)

    if not deferred_df.empty:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, "Deferred Assets", new_x="LMARGIN", new_y="NEXT")
        for _, row in deferred_df.iterrows():
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(217, 119, 6)
            pdf.cell(0, 6, f"  {row['Asset Name']}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(71, 85, 105)
            pdf.multi_cell(0, 5,
                f"      Cost: ${row['Upgrade Cost ($K)']:,.0f}K  |  "
                f"Failure cost if deferred: ${row['Failure Cost ($K)']:,.0f}K  |  "
                f"Priority: {row['Priority Score']:.0f}/100")
            pdf.ln(1)

    # ── Footer ──
    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, "Failure Forecasters - CIVE 580c4 Final Project", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "This report is for planning purposes. Carbon estimates are order-of-magnitude.", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def generate_csv_export(ranked_df: pd.DataFrame) -> str:
    """Generate a comprehensive CSV export of all asset data and scores."""
    export_cols = [c for c in [
        "Asset Name", "Asset Type", "Age (yrs)", "Condition (1-5)",
        "Failure Consequence", "Upgrade Cost ($K)", "Material", "Critical Asset",
        "Expected Life (yrs)", "RUL (yrs)", "Life Used (%)",
        "Priority Score", "Risk Score", "Carbon (t CO2e)",
        "Failure Cost ($K)", "Deferral Risk ($K)",
        "Risk Reduction per $100K", "Recommendation",
    ] if c in ranked_df.columns]
    return ranked_df[export_cols].to_csv(index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Upload & Settings
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📁 Data Import")

    st.download_button(
        "📥 Download CSV Template",
        data=generate_csv_template(),
        file_name="asset_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader(
        "Upload asset inventory", type=["csv", "xlsx"],
        help="Upload a CSV or Excel file matching the template format.",
    )

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".xlsx"):
                upload_df = pd.read_excel(uploaded)
            else:
                upload_df = pd.read_csv(uploaded)

            required = {"Asset Name", "Asset Type", "Age (yrs)", "Condition (1-5)",
                        "Failure Consequence", "Upgrade Cost ($K)", "Material"}
            missing = required - set(upload_df.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                if "Critical Asset" not in upload_df.columns:
                    upload_df["Critical Asset"] = False
                st.session_state.assets = upload_df
                st.success(f"Loaded {len(upload_df)} assets")
        except Exception as e:
            st.error(f"Upload error: {e}")

    st.markdown("---")
    st.markdown("### ⚖️ Priority Weights")
    st.caption("Adjust how factors influence the priority ranking.")
    w_age = st.slider("Age", 0.0, 1.0, DEFAULT_W_AGE, 0.05, key="w_age")
    w_cond = st.slider("Condition", 0.0, 1.0, DEFAULT_W_COND, 0.05, key="w_cond")
    w_conseq = st.slider("Consequence", 0.0, 1.0, DEFAULT_W_CONSEQ, 0.05, key="w_conseq")
    w_rul = st.slider("Remaining Life", 0.0, 1.0, DEFAULT_W_RUL, 0.05, key="w_rul")

    st.markdown("---")
    st.markdown("### 🌱 Sustainability")
    carbon_weight = st.slider("Carbon penalty", 0.0, 1.0, 0.15, 0.05,
                              help="Higher = penalize high-carbon materials more.")
    favor_low_carbon = st.toggle("Favor low-carbon materials", value=False)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div>
        <h1>💧 Failure Forecasters</h1>
        <p>Prioritize water infrastructure upgrades by failure risk, cost, and sustainability.</p>
        <span class="hero-badge">Capital Improvement Planner</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Asset Inventory
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-hdr"><span class="num">1</span> Asset Inventory</div>',
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
                                                             help="1 = Excellent … 5 = Failed", width="small"),
        "Failure Consequence": st.column_config.NumberColumn("Consequence", min_value=1, max_value=5, step=1,
                                                             help="1 = Minimal … 5 = Catastrophic", width="small"),
        "Upgrade Cost ($K)":   st.column_config.NumberColumn("Cost ($K)", min_value=0, step=10, format="$%d", width="small"),
        "Material":            st.column_config.SelectboxColumn("Material", options=MATERIAL_OPTIONS, width="small"),
        "Critical Asset":      st.column_config.CheckboxColumn("Critical?", width="small",
                                                                help="Single point of failure / no redundancy"),
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


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Planning Constraints
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-hdr"><span class="num">2</span> Planning Constraints</div>',
            unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns([1.2, 0.8, 0.8])
with pc1:
    budget = st.number_input("Capital Budget ($K)", min_value=0, value=3000, step=100,
                             help="Maximum budget for this planning cycle.")
with pc2:
    plan_years = st.number_input("Planning Horizon (yrs)", min_value=1, max_value=10, value=5, step=1,
                                 help="Number of years for multi-year CIP.")
with pc3:
    # Default annual budget: at least enough to fund the single most expensive asset,
    # so every asset can be scheduled in some year
    total_asset_cost = valid_df["Upgrade Cost ($K)"].sum() if "Upgrade Cost ($K)" in valid_df.columns else budget
    smart_default = max(
        int(total_asset_cost / max(plan_years, 1)),  # spread evenly
        int(valid_df["Upgrade Cost ($K)"].max()) if "Upgrade Cost ($K)" in valid_df.columns else 0,  # cover largest asset
    )
    yearly_budget = st.number_input("Annual Budget ($K)", min_value=0,
                                    value=smart_default, step=100,
                                    help="Per-year budget for multi-year planning.")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE
# ═══════════════════════════════════════════════════════════════════════════════

scored_df = compute_scores(valid_df, w_age, w_cond, w_conseq, w_rul,
                           carbon_weight, favor_low_carbon)
ranked_df = select_within_budget(scored_df, budget)
ranked_df["Recommendation"] = ranked_df.apply(label_recommendation, axis=1)

selected_df = ranked_df[ranked_df["Selected"]].copy()
deferred_df = ranked_df[~ranked_df["Selected"]].copy()

# ── System Health (needed by Overview + PDF export) ──
health_before = compute_system_health(scored_df)
after_df = scored_df.copy()
after_df.loc[after_df["Asset Name"].isin(selected_df["Asset Name"]), "Condition (1-5)"] = 1
health_after = compute_system_health(after_df)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Results Dashboard (Tabbed)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-hdr"><span class="num">3</span> Results</div>',
            unsafe_allow_html=True)

tab_overview, tab_analysis, tab_multiyear, tab_sensitivity = st.tabs([
    "📋 Overview", "📊 Analysis", "📅 Multi-Year CIP", "🔬 Scenarios & Sensitivity"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    total_cost   = selected_df["Upgrade Cost ($K)"].sum()
    remaining    = budget - total_cost
    total_carbon = selected_df["Carbon (t CO2e)"].sum()
    pct_used     = (total_cost / budget * 100) if budget > 0 else 0
    high_risk    = len(selected_df[selected_df["Failure Consequence"] >= 4])
    total_risk   = ranked_df["Risk Score"].sum()
    risk_reduced = selected_df["Risk Score"].sum() if not selected_df.empty else 0
    risk_pct     = (risk_reduced / total_risk * 100) if total_risk > 0 else 0
    avoided_cost = selected_df["Deferral Risk ($K)"].sum() if not selected_df.empty else 0

    # KPI row
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Assets Selected</div>
        <div class="kpi-value">{len(selected_df)} <span style="font-size:.8rem;font-weight:400;color:#64748b;">of {len(ranked_df)}</span></div>
        <div class="kpi-sub">{high_risk} high-consequence asset{'s' if high_risk != 1 else ''}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Planned Investment</div>
        <div class="kpi-value">${total_cost:,.0f}K</div>
        <div class="kpi-sub">{pct_used:.0f}% of ${budget:,.0f}K budget</div>
      </div>
      <div class="kpi kpi-accent">
        <div class="kpi-label">System Risk Reduction</div>
        <div class="kpi-value">{risk_pct:.0f}%</div>
        <div class="kpi-sub">{risk_reduced:.0f} of {total_risk:.0f} risk points</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Failure Cost Avoided</div>
        <div class="kpi-value">${avoided_cost:,.0f}K</div>
        <div class="kpi-sub">Emergency repair savings</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Budget Remaining</div>
        <div class="kpi-value">${remaining:,.0f}K</div>
        <div class="kpi-sub">Available for future cycles</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Embodied Carbon</div>
        <div class="kpi-value">{total_carbon:,.1f} t</div>
        <div class="kpi-sub">CO₂e ≈ {total_carbon / 4.6:.0f} households' annual emissions</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Before / After System Health ──

    hc1, hc2 = st.columns(2)
    with hc1:
        fig_hb = go.Figure(go.Indicator(
            mode="gauge+number", value=health_before,
            title={"text": "Current System Health", "font": {"size": 13}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#64748b"},
                   "steps": [{"range": [0, 40], "color": "#fee2e2"},
                             {"range": [40, 70], "color": "#fef3c7"},
                             {"range": [70, 100], "color": "#dcfce7"}],
                   "threshold": {"line": {"color": "#0f172a", "width": 2}, "value": health_before}},
            number={"suffix": "/100", "font": {"size": 28}},
        ))
        fig_hb.update_layout(height=200, margin=dict(l=30, r=30, t=50, b=10),
                             paper_bgcolor="#ffffff")
        st.plotly_chart(fig_hb, use_container_width=True)
    with hc2:
        fig_ha = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=health_after,
            delta={"reference": health_before, "increasing": {"color": "#059669"},
                   "suffix": " pts", "font": {"size": 14}},
            title={"text": "After Planned Upgrades", "font": {"size": 13}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#059669"},
                   "steps": [{"range": [0, 40], "color": "#fee2e2"},
                             {"range": [40, 70], "color": "#fef3c7"},
                             {"range": [70, 100], "color": "#dcfce7"}],
                   "threshold": {"line": {"color": "#0f172a", "width": 2}, "value": health_after}},
            number={"suffix": "/100", "font": {"size": 28}},
        ))
        fig_ha.update_layout(height=200, margin=dict(l=30, r=30, t=50, b=10),
                             paper_bgcolor="#ffffff")
        st.plotly_chart(fig_ha, use_container_width=True)

    # ── Recommended asset cards ──
    if not selected_df.empty:
        st.markdown("#### Recommended This Cycle")
        rows = list(selected_df.iterrows())
        for i in range(0, len(rows), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(rows):
                    break
                _, row = rows[i + j]
                rank_num = i + j + 1
                rationale = generate_rationale(row, scored_df)
                alts = get_material_alternatives(row, scored_df)

                with col:
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="asset-card">'
                            f'<span class="asset-rank">PRIORITY #{rank_num}</span>'
                            f'<div class="asset-name">{row["Asset Name"]}</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"_{rationale}_")
                        st.caption(
                            f"{row['Asset Type']}  ·  "
                            f"{int(row['Age (yrs)'])} yrs ({int(row.get('Life Used (%)', 0))}% of life)  ·  "
                            f"Condition {int(row['Condition (1-5)'])}/5  ·  "
                            f"Score **{row['Priority Score']:.0f}**/100  ·  "
                            f"**${row['Upgrade Cost ($K)']:,.0f}K**  ·  "
                            f"{row['Carbon (t CO2e)']:.1f} t CO₂e"
                        )
                        if row.get("RUL (yrs)", 0) <= 5:
                            st.warning(f"⚠ Only {int(row['RUL (yrs)'])} years of remaining useful life", icon="⚠️")
                        if alts:
                            with st.expander("🌱 Lower-carbon alternatives"):
                                for alt in alts:
                                    direction = "saves" if alt["carbon_savings"] > 0 else "adds"
                                    st.caption(
                                        f"**{alt['material']}** — {direction} "
                                        f"{abs(alt['carbon_savings']):.1f} t CO₂e → "
                                        f"{alt['new_carbon']:.1f} t total  \n"
                                        f"_{alt['note']}_"
                                    )
    else:
        st.info("No assets fit within the current budget. Increase the capital budget or adjust priorities.")

    # ── Deferred assets ──
    if not deferred_df.empty:
        st.markdown("#### Deferred — Future Cycles")
        total_deferred_cost = deferred_df["Upgrade Cost ($K)"].sum()
        total_deferred_risk = deferred_df["Failure Cost ($K)"].sum()

        for _, row in deferred_df.iterrows():
            with st.container(border=True):
                rationale_d = generate_rationale(row, scored_df)
                st.markdown(
                    f'<div class="asset-card defer-card">'
                    f'<span class="asset-rank defer-rank">DEFERRED</span>'
                    f'<div class="asset-name">{row["Asset Name"]}</div></div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"{row['Asset Type']}  ·  "
                    f"{int(row['Age (yrs)'])} yrs  ·  "
                    f"Condition {int(row['Condition (1-5)'])}/5  ·  "
                    f"**${row['Upgrade Cost ($K)']:,.0f}K** upgrade  ·  "
                    f"**${row['Failure Cost ($K)']:,.0f}K** if it fails"
                )
                st.markdown(
                    f"_Deferred because upgrade cost (${row['Upgrade Cost ($K)']:,.0f}K) "
                    f"exceeds remaining budget. {rationale_d}_"
                )

    # ── Full ranking table ──
    st.markdown("#### Full Asset Ranking")

    display_cols = ["Asset Name", "Asset Type", "Priority Score", "Risk Score",
                    "Upgrade Cost ($K)", "Failure Cost ($K)", "RUL (yrs)",
                    "Carbon (t CO2e)", "Recommendation"]

    available_cols = [c for c in display_cols if c in ranked_df.columns]

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

    format_dict = {
        "Priority Score": "{:.1f}", "Risk Score": "{:.1f}",
        "Upgrade Cost ($K)": "${:,.0f}", "Failure Cost ($K)": "${:,.0f}",
        "RUL (yrs)": "{:.0f}", "Carbon (t CO2e)": "{:.1f}",
    }
    format_dict = {k: v for k, v in format_dict.items() if k in available_cols}

    styled = (
        ranked_df[available_cols].style
        .apply(style_row, axis=1)
        .map(style_rec, subset=["Recommendation"])
        .format(format_dict)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_analysis:

    # ── Row 1: Bubble chart + Cumulative risk curve ──
    ac1, ac2 = st.columns(2)

    with ac1:
        st.markdown("**Risk vs. Cost vs. Carbon**")
        bubble_df = ranked_df.copy()
        bubble_df["Bubble Size"] = bubble_df["Carbon (t CO2e)"].clip(lower=1)
        # Short labels for bubble chart to avoid overlap
        bubble_df["Label"] = bubble_df["Asset Name"].apply(
            lambda n: n.replace("Transmission ", "").replace("Booster ", "")
                       .replace("Storage ", "").replace("Distribution ", "")
                       .replace("Station ", "Stn ")
        )
        fig_bubble = px.scatter(
            bubble_df,
            x="Upgrade Cost ($K)", y="Risk Score",
            size="Bubble Size", color="Recommendation",
            color_discrete_map=REC_COLORS,
            hover_name="Asset Name",
            hover_data={"Upgrade Cost ($K)": ":$,.0f", "Risk Score": ":.1f",
                        "Carbon (t CO2e)": ":.1f", "Priority Score": ":.1f",
                        "Bubble Size": False, "Recommendation": True, "Label": False},
            text="Label",
            size_max=35,
        )
        fig_bubble.update_traces(
            textposition="top center",
            textfont=dict(size=9, color="#334155"),
        )
        fig_bubble.update_layout(
            **CHART_LAYOUT, height=460, margin=dict(l=50, r=30, t=30, b=80),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                        font=dict(size=10), title_text=""),
            xaxis_title="Upgrade Cost ($K)", yaxis_title="Risk Score",
        )
        fig_bubble.update_xaxes(gridcolor="#edf2f7")
        fig_bubble.update_yaxes(gridcolor="#edf2f7", range=[0, bubble_df["Risk Score"].max() * 1.2])
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption("Bubble size = embodied carbon. Top-left = high risk, low cost (best value).")

    with ac2:
        st.markdown("**Cumulative Risk Reduction vs. Spend**")
        cum_df = ranked_df.sort_values("Priority Score", ascending=False).copy()
        cum_df["Cumulative Cost ($K)"] = cum_df["Upgrade Cost ($K)"].cumsum()
        cum_df["Cumulative Risk Reduced"] = cum_df["Risk Score"].cumsum()
        total_risk_for_pct = ranked_df["Risk Score"].sum()
        cum_df["Cumulative Risk Reduction (%)"] = (
            cum_df["Cumulative Risk Reduced"] / total_risk_for_pct * 100
        ).round(1) if total_risk_for_pct > 0 else 0

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum_df["Cumulative Cost ($K)"],
            y=cum_df["Cumulative Risk Reduction (%)"],
            mode="lines+markers",
            marker=dict(size=8, color="#0d9488"),
            line=dict(color="#0d9488", width=2.5),
            text=cum_df["Asset Name"],
            hovertemplate="<b>%{text}</b><br>Cumulative spend: $%{x:,.0f}K<br>"
                          "Risk reduction: %{y:.1f}%<extra></extra>",
        ))
        # Budget line
        fig_cum.add_vline(x=budget, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"Budget: ${budget:,.0f}K",
                          annotation_position="top left",
                          annotation_font=dict(size=10, color="#ef4444"))
        fig_cum.update_layout(
            **CHART_LAYOUT, height=460, margin=dict(l=50, r=30, t=30, b=80),
            xaxis_title="Cumulative Investment ($K)",
            yaxis_title="Cumulative Risk Reduction (%)",
            showlegend=False,
        )
        fig_cum.update_xaxes(gridcolor="#edf2f7")
        fig_cum.update_yaxes(gridcolor="#edf2f7", range=[0, 105])
        st.plotly_chart(fig_cum, use_container_width=True)
        st.caption("Shows diminishing returns — the 'knee' helps justify the optimal budget level.")

    # ── Row 2: Priority bars + Cost efficiency ──
    bc1, bc2 = st.columns(2)

    # Sort for chart display
    chart_order = ranked_df.sort_values("Priority Score", ascending=True)
    chart_colors = [REC_COLORS.get(r, "#94a3b8") for r in chart_order["Recommendation"]]

    with bc1:
        st.markdown("**Priority Score by Asset**")
        fig_pri = go.Figure()
        fig_pri.add_trace(go.Bar(
            y=chart_order["Asset Name"], x=chart_order["Priority Score"],
            orientation="h", marker_color=chart_colors,
            text=chart_order["Priority Score"].apply(lambda v: f"{v:.0f}"),
            textposition="outside", textfont=dict(size=9, color="#475569"),
            hovertemplate="<b>%{y}</b><br>Priority: %{x:.1f}/100<extra></extra>",
            showlegend=False,
        ))
        fig_pri.update_layout(**CHART_LAYOUT, height=370, margin=dict(l=8, r=30, t=44, b=28),
                              title=dict(text="Priority Score", font=dict(size=13, color="#334155")))
        fig_pri.update_xaxes(gridcolor="#edf2f7", title_text="Score (0–100)", title_font_size=10)
        fig_pri.update_yaxes(title_text="")
        st.plotly_chart(fig_pri, use_container_width=True)

    with bc2:
        st.markdown("**Cost Efficiency (Risk Reduction per $100K)**")
        eff_order = ranked_df.sort_values("Risk Reduction per $100K", ascending=True)
        eff_colors = [REC_COLORS.get(r, "#94a3b8") for r in eff_order["Recommendation"]]

        fig_eff = go.Figure()
        fig_eff.add_trace(go.Bar(
            y=eff_order["Asset Name"], x=eff_order["Risk Reduction per $100K"],
            orientation="h", marker_color=eff_colors,
            text=eff_order["Risk Reduction per $100K"].apply(lambda v: f"{v:.1f}"),
            textposition="outside", textfont=dict(size=9, color="#475569"),
            hovertemplate="<b>%{y}</b><br>Efficiency: %{x:.1f} per $100K<br>"
                          "Cost: $%{customdata[0]:,.0f}K<extra></extra>",
            customdata=eff_order[["Upgrade Cost ($K)"]].values,
            showlegend=False,
        ))
        fig_eff.update_layout(**CHART_LAYOUT, height=370, margin=dict(l=8, r=30, t=44, b=28),
                              title=dict(text="Cost Efficiency", font=dict(size=13, color="#334155")))
        fig_eff.update_xaxes(gridcolor="#edf2f7", title_text="Risk per $100K", title_font_size=10)
        fig_eff.update_yaxes(title_text="")
        st.plotly_chart(fig_eff, use_container_width=True)

    # Shared legend
    st.markdown(
        '<div style="text-align:center;font-size:.75rem;color:#64748b;margin:-.3rem 0 .5rem;">'
        '<span style="color:#059669;">●</span> Recommended &nbsp;&nbsp; '
        '<span style="color:#d97706;">●</span> Near-Term &nbsp;&nbsp; '
        '<span style="color:#94a3b8;">●</span> Deferred'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Row 3: Risk Matrix Heatmap ──
    st.markdown("**Risk Assessment Matrix**")
    st.caption("Classic 5×5 condition vs. consequence matrix. Dots show asset positions.")

    # Build heatmap grid — rows ordered so 5-Failed is at top (conventional risk matrix)
    cond_labels = ["5-Failed", "4-Poor", "3-Fair", "2-Good", "1-Excellent"]
    cons_labels = ["1-Minimal", "2-Low", "3-Moderate", "4-High", "5-Catastrophic"]
    matrix = np.zeros((5, 5))
    for ri, cond_val in enumerate([5, 4, 3, 2, 1]):  # top row = 5-Failed
        for ci, cons_val in enumerate([1, 2, 3, 4, 5]):
            matrix[ri][ci] = (cond_val / 5) * (cons_val / 5) * 100

    fig_matrix = go.Figure()

    # Heatmap background — use numeric coords so dots can be jittered within cells
    fig_matrix.add_trace(go.Heatmap(
        z=matrix, x=list(range(5)), y=list(range(5)),
        colorscale=[[0, "#dcfce7"], [0.3, "#fef9c3"], [0.6, "#fed7aa"], [1, "#fecaca"]],
        showscale=False, hoverinfo="skip",
    ))

    # Asset dots — use short label (e.g. "Main A", "Pump 1") for readability
    def short_label(name: str) -> str:
        """Create a readable short label from the asset name."""
        parts = name.split()
        if len(parts) >= 2:
            # Drop generic leading words like "Transmission", "Booster", "Storage", "Distribution"
            skip = {"Transmission", "Booster", "Storage", "Distribution", "WWTP"}
            filtered = [p for p in parts if p not in skip]
            return " ".join(filtered[:2]) if filtered else " ".join(parts[-2:])
        return name

    # Map numeric values to categorical label strings so dots land on heatmap cells
    cond_val_to_label = {5: "5-Failed", 4: "4-Poor", 3: "3-Fair", 2: "2-Good", 1: "1-Excellent"}
    cons_val_to_label = {1: "1-Minimal", 2: "2-Low", 3: "3-Moderate", 4: "4-High", 5: "5-Catastrophic"}

    # Group assets by cell to apply jitter when dots overlap
    from collections import defaultdict
    cell_counts = defaultdict(int)
    cell_index = {}
    for _, row in ranked_df.iterrows():
        key = (int(row["Condition (1-5)"]), int(row["Failure Consequence"]))
        cell_index[(row["Asset Name"])] = cell_counts[key]
        cell_counts[key] += 1

    for _, row in ranked_df.iterrows():
        cond_val = int(row["Condition (1-5)"])
        cons_val = int(row["Failure Consequence"])
        color = REC_COLORS.get(row["Recommendation"], "#64748b")
        label = short_label(row["Asset Name"])
        key = (cond_val, cons_val)
        n_in_cell = cell_counts[key]
        idx = cell_index[row["Asset Name"]]
        # Apply horizontal offset when multiple assets share a cell
        if n_in_cell > 1:
            spread = 0.35
            x_offset = -spread / 2 + spread * idx / (n_in_cell - 1)
        else:
            x_offset = 0
        # Use numeric coords (0-based index) to allow sub-cell positioning
        cons_idx = cons_val - 1  # 0-4
        cond_idx = 5 - cond_val  # flip: 5->0, 4->1, 3->2, 2->3, 1->4
        fig_matrix.add_trace(go.Scatter(
            x=[cons_idx + x_offset], y=[cond_idx],
            mode="markers+text", marker=dict(size=18, color=color, line=dict(width=2, color="#fff")),
            text=[label],
            textposition="top center", textfont=dict(size=10, color="#1e293b"),
            hovertemplate=f"<b>{row['Asset Name']}</b><br>"
                          f"Condition: {cond_val}/5<br>"
                          f"Consequence: {cons_val}/5<br>"
                          f"Risk Score: {row['Risk Score']:.0f}<extra></extra>",
            showlegend=False,
        ))

    fig_matrix.update_layout(
        **CHART_LAYOUT, height=450,
        xaxis=dict(title="Failure Consequence",
                   tickvals=list(range(5)), ticktext=cons_labels),
        yaxis=dict(title="Condition",
                   tickvals=list(range(5)), ticktext=cond_labels),
        margin=dict(l=10, r=20, t=30, b=50),
    )
    st.plotly_chart(fig_matrix, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: MULTI-YEAR CIP
# ─────────────────────────────────────────────────────────────────────────────
with tab_multiyear:
    st.markdown("**Multi-Year Capital Improvement Plan**")
    st.caption(
        f"Allocates assets across {plan_years} years at ${yearly_budget:,.0f}K/year. "
        "Deferred assets degrade in condition each year."
    )

    year_plans = multi_year_plan(valid_df, plan_years, yearly_budget,
                                 w_age, w_cond, w_conseq, w_rul,
                                 carbon_weight, favor_low_carbon)

    # Summary metrics by year
    year_summary = []
    cumulative_cost = 0
    cumulative_risk = 0
    total_sys_risk = scored_df["Risk Score"].sum()

    for yr, ydf in enumerate(year_plans, 1):
        yr_cost = ydf["Upgrade Cost ($K)"].sum() if not ydf.empty else 0
        yr_risk = ydf["Risk Score"].sum() if not ydf.empty else 0
        cumulative_cost += yr_cost
        cumulative_risk += yr_risk
        year_summary.append({
            "Year": yr,
            "Assets": len(ydf),
            "Investment ($K)": yr_cost,
            "Cumulative ($K)": cumulative_cost,
            "Risk Addressed": yr_risk,
            "Cum. Risk Reduction (%)": round(cumulative_risk / total_sys_risk * 100, 1) if total_sys_risk > 0 else 0,
        })

    summary_df = pd.DataFrame(year_summary)
    st.dataframe(
        summary_df.style.format({
            "Investment ($K)": "${:,.0f}", "Cumulative ($K)": "${:,.0f}",
            "Risk Addressed": "{:.0f}", "Cum. Risk Reduction (%)": "{:.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    # Timeline chart
    if not summary_df.empty:
        mc1, mc2 = st.columns(2)
        with mc1:
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Bar(
                x=summary_df["Year"], y=summary_df["Investment ($K)"],
                marker_color="#145374", name="Annual Investment",
                text=summary_df["Investment ($K)"].apply(lambda v: f"${v:,.0f}K"),
                textposition="outside", textfont=dict(size=9),
            ))
            fig_timeline.update_layout(
                **CHART_LAYOUT, height=320, margin=dict(l=8, r=30, t=44, b=28),
                title=dict(text="Annual Investment", font=dict(size=13)),
                xaxis=dict(title="Year", dtick=1),
                yaxis=dict(title="Investment ($K)"),
                showlegend=False,
            )
            fig_timeline.update_xaxes(gridcolor="#edf2f7")
            fig_timeline.update_yaxes(gridcolor="#edf2f7")
            st.plotly_chart(fig_timeline, use_container_width=True)

        with mc2:
            fig_cumrisk = go.Figure()
            fig_cumrisk.add_trace(go.Scatter(
                x=summary_df["Year"],
                y=summary_df["Cum. Risk Reduction (%)"],
                mode="lines+markers+text",
                marker=dict(size=10, color="#059669"),
                line=dict(color="#059669", width=2.5),
                text=summary_df["Cum. Risk Reduction (%)"].apply(lambda v: f"{v:.0f}%"),
                textposition="top center", textfont=dict(size=9),
            ))
            fig_cumrisk.update_layout(
                **CHART_LAYOUT, height=320, margin=dict(l=8, r=30, t=44, b=28),
                title=dict(text="Cumulative Risk Reduction", font=dict(size=13)),
                xaxis=dict(title="Year", dtick=1),
                yaxis=dict(title="Risk Reduction (%)", range=[0, 105]),
                showlegend=False,
            )
            fig_cumrisk.update_xaxes(gridcolor="#edf2f7")
            fig_cumrisk.update_yaxes(gridcolor="#edf2f7")
            st.plotly_chart(fig_cumrisk, use_container_width=True)

    # Year-by-year detail
    for yr, ydf in enumerate(year_plans, 1):
        if ydf.empty:
            continue
        with st.expander(f"Year {yr} — {len(ydf)} assets, ${ydf['Upgrade Cost ($K)'].sum():,.0f}K"):
            detail_cols = ["Asset Name", "Asset Type", "Priority Score",
                           "Upgrade Cost ($K)", "Risk Score", "Carbon (t CO2e)"]
            available = [c for c in detail_cols if c in ydf.columns]
            st.dataframe(ydf[available], use_container_width=True, hide_index=True)

    # Unplanned assets
    all_planned = set()
    for ydf in year_plans:
        if not ydf.empty:
            all_planned.update(ydf["Asset Name"])
    unplanned = valid_df[~valid_df["Asset Name"].isin(all_planned)]
    if not unplanned.empty:
        st.warning(
            f"**{len(unplanned)} asset(s) not scheduled** within the {plan_years}-year horizon: "
            f"{', '.join(unplanned['Asset Name'].tolist())}. "
            "Consider increasing the annual budget or extending the planning horizon."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB: SCENARIOS & SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
with tab_sensitivity:

    # ── Budget Scenario Comparison ──
    st.markdown("**Budget Scenario Comparison**")
    st.caption("Compare outcomes across different budget levels.")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        budget_a = st.number_input("Constrained ($K)", value=max(int(budget * 0.35), 100), step=100, key="ba")
    with sc2:
        budget_b = st.number_input("Baseline ($K)", value=budget, step=100, key="bb")
    with sc3:
        budget_c = st.number_input("Expanded ($K)", value=int(budget * 1.8), step=100, key="bc")

    scenarios = {"Constrained": budget_a, "Baseline": budget_b, "Expanded": budget_c}
    scenario_results = []
    for label, bgt in scenarios.items():
        sel = select_within_budget(scored_df, bgt)
        sel_df = sel[sel["Selected"]]
        risk_red = sel_df["Risk Score"].sum() / total_risk * 100 if total_risk > 0 else 0
        scenario_results.append({
            "Scenario": f"{label} (${bgt:,.0f}K)",
            "Assets Funded": len(sel_df),
            "Investment ($K)": sel_df["Upgrade Cost ($K)"].sum(),
            "Risk Reduction (%)": round(risk_red, 1),
            "Carbon (t CO2e)": round(sel_df["Carbon (t CO2e)"].sum(), 1),
            "Failure Cost Avoided ($K)": round(sel_df["Deferral Risk ($K)"].sum(), 0) if not sel_df.empty else 0,
        })

    scenario_df = pd.DataFrame(scenario_results)
    st.dataframe(
        scenario_df.style.format({
            "Investment ($K)": "${:,.0f}",
            "Risk Reduction (%)": "{:.1f}%",
            "Carbon (t CO2e)": "{:.1f}",
            "Failure Cost Avoided ($K)": "${:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # Scenario bar chart
    fig_scen = go.Figure()
    fig_scen.add_trace(go.Bar(
        x=[r["Scenario"] for r in scenario_results],
        y=[r["Risk Reduction (%)"] for r in scenario_results],
        marker_color=["#94a3b8", "#059669", "#145374"],
        text=[f"{r['Risk Reduction (%)']:.0f}%" for r in scenario_results],
        textposition="outside",
    ))
    fig_scen.update_layout(
        **CHART_LAYOUT, height=280, margin=dict(l=8, r=30, t=44, b=28),
        title=dict(text="Risk Reduction by Budget Scenario", font=dict(size=13)),
        yaxis=dict(title="Risk Reduction (%)", range=[0, 110]),
        showlegend=False,
    )
    fig_scen.update_xaxes(gridcolor="#edf2f7")
    fig_scen.update_yaxes(gridcolor="#edf2f7")
    st.plotly_chart(fig_scen, use_container_width=True)
    st.caption(
        "Use this to justify budget requests: show stakeholders the risk reduction "
        "gained by increasing the budget, or the consequences of cutting it."
    )

    st.markdown("---")

    # ── Weight Sensitivity Analysis ──
    st.markdown("**Weight Sensitivity Analysis**")
    st.caption("How do rankings change under different weighting philosophies?")

    profiles = {
        "Current":        {"age": w_age, "cond": w_cond, "conseq": w_conseq, "rul": w_rul},
        "Age-Focused":    {"age": 0.80, "cond": 0.05, "conseq": 0.05, "rul": 0.10},
        "Condition-First": {"age": 0.05, "cond": 0.80, "conseq": 0.10, "rul": 0.05},
        "Risk-Averse":    {"age": 0.05, "cond": 0.10, "conseq": 0.80, "rul": 0.05},
        "Life-Based":     {"age": 0.05, "cond": 0.05, "conseq": 0.05, "rul": 0.85},
    }

    sensitivity_data = []
    for profile_name, weights in profiles.items():
        prof_scored = compute_scores(valid_df, weights["age"], weights["cond"],
                                     weights["conseq"], weights["rul"],
                                     carbon_weight, favor_low_carbon)
        prof_ranked = prof_scored.sort_values("Priority Score", ascending=False)
        for rank, (_, row) in enumerate(prof_ranked.iterrows(), 1):
            sensitivity_data.append({
                "Profile": profile_name,
                "Asset": row["Asset Name"],
                "Rank": rank,
                "Score": row["Priority Score"],
            })

    sens_df = pd.DataFrame(sensitivity_data)

    # Bump chart: rank by profile — unique colors per asset so lines are distinguishable
    ASSET_COLORS = ["#0d9488", "#2563eb", "#d97706", "#dc2626", "#7c3aed",
                    "#059669", "#0284c7", "#ca8a04", "#e11d48", "#6d28d9"]
    fig_sens = go.Figure()
    profile_names = list(profiles.keys())
    for i, asset_name in enumerate(valid_df["Asset Name"].unique()):
        asset_sens = sens_df[sens_df["Asset"] == asset_name]
        ranks = [asset_sens[asset_sens["Profile"] == p]["Rank"].values[0]
                 for p in profile_names if len(asset_sens[asset_sens["Profile"] == p]) > 0]
        color = ASSET_COLORS[i % len(ASSET_COLORS)]
        fig_sens.add_trace(go.Scatter(
            x=profile_names, y=ranks,
            mode="lines+markers", name=asset_name,
            line=dict(width=2.5, color=color),
            marker=dict(size=10, color=color),
            hovertemplate=f"<b>{asset_name}</b><br>Profile: %{{x}}<br>Rank: %{{y}}<extra></extra>",
        ))

    fig_sens.update_layout(
        **CHART_LAYOUT, height=400,
        title=dict(text="Ranking Stability Across Weight Profiles", font=dict(size=13)),
        yaxis=dict(title="Rank (1 = highest priority)", autorange="reversed"),
        xaxis=dict(title=""),
        legend=dict(font=dict(size=9), orientation="h", yanchor="bottom", y=-0.35,
                    xanchor="center", x=0.5),
        margin=dict(l=10, r=20, t=44, b=80),
    )
    fig_sens.update_xaxes(gridcolor="#edf2f7")
    fig_sens.update_yaxes(gridcolor="#edf2f7")
    st.plotly_chart(fig_sens, use_container_width=True)
    st.caption("Flat lines = robust ranking regardless of weights. Crossing lines = sensitive to assumptions.")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-hdr"><span class="num">4</span> Export & Reports</div>',
            unsafe_allow_html=True)

ex1, ex2 = st.columns(2)

with ex1:
    pdf_bytes = generate_pdf(ranked_df, selected_df, deferred_df, budget,
                             scored_df, health_before, health_after)
    st.download_button("📄 Download PDF Summary", data=pdf_bytes,
                       file_name=f"cip_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                       mime="application/pdf", use_container_width=True)

with ex2:
    csv_export = generate_csv_export(ranked_df)
    st.download_button("📊 Download Full Data (CSV)", data=csv_export,
                       file_name=f"asset_data_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO USE & METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📘 How to Use This Tool"):
    st.markdown("""
**This tool helps you decide which water infrastructure assets to upgrade first,
given a limited budget.** Here's how to use each part:

---

**Step 1 — Asset Inventory (the table at the top)**

Enter your assets or upload a CSV/Excel file. Each asset needs:
- **Asset Name**: A label to identify it (e.g., "Transmission Main A")
- **Type**: What kind of asset (Pipe, Pump, Tank, Clarifier, Valve, or Hydrant)
- **Age**: How old the asset is in years
- **Condition**: Rate from 1 (excellent, like new) to 5 (failed or near failure)
- **Failure Consequence**: How bad would it be if this asset fails? 1 = minor inconvenience, 5 = catastrophic (e.g., major service outage, public safety risk)
- **Upgrade Cost**: Estimated cost to replace or rehabilitate, in thousands of dollars
- **Material**: What the asset is made of — this affects both its expected lifespan and carbon footprint
- **Critical Asset**: Check this if the asset has no backup or redundancy (single point of failure)

---

**Step 2 — Planning Constraints**

- **Capital Budget**: Your total available budget for this planning cycle. The tool picks the best combination of assets that fit within this amount.
- **Planning Horizon**: How many years to plan ahead (used in the Multi-Year CIP tab)
- **Annual Budget**: Per-year budget for the multi-year plan

---

**Step 3 — Results (four tabs)**

**Overview Tab**
- **KPI cards** at the top show a summary: how many assets are funded, total cost, risk reduction, failure cost avoided, and embodied carbon
- **System Health gauges** show overall system condition before and after the planned upgrades. Higher is healthier.
- **Asset cards** show each funded asset with a plain-English explanation of why it was prioritized
- **Deferred assets** show what didn't make the cut and what it could cost if those assets fail
- **Full Ranking table** shows every asset sorted by priority score with color-coded recommendations

**Analysis Tab**
- **Bubble chart** (Risk vs. Cost vs. Carbon): Each bubble is an asset. Position shows risk (vertical) vs. cost (horizontal). Bubble size shows carbon footprint. Assets in the top-left are high risk and low cost — the best value upgrades.
- **Cumulative Risk Reduction curve**: Shows how much total risk you eliminate as you spend more. The red dashed line is your budget. The "knee" of the curve is where you get the most risk reduction per dollar.
- **Priority Score bars**: Simple ranking of all assets by their overall priority score
- **Cost Efficiency bars**: Which assets give you the most risk reduction per $100K spent
- **Risk Matrix**: A standard 5×5 grid used in asset management. Assets in the lower-right (poor condition + high consequence) are the highest risk.

**Multi-Year CIP Tab**
- Spreads your assets across multiple budget years. Assets that don't get funded in Year 1 are re-evaluated in Year 2 with slightly worse condition (because they've aged another year). This shows how deferring maintenance creates more urgency over time.

**Scenarios & Sensitivity Tab**
- **Budget Scenarios**: Compare three different budget levels side by side to see how much more risk you reduce with a bigger budget. Useful for justifying budget requests.
- **Weight Sensitivity**: Shows whether the rankings change when you use different weighting priorities (e.g., "what if we only cared about condition?" vs. "what if we only cared about remaining life?"). Flat lines mean the ranking is robust. Crossing lines mean the ranking depends on your assumptions.

---

**Sidebar Controls**

- **Priority Weights**: Adjust how much each factor matters. Higher weight = more influence on the ranking. The defaults are balanced, but you can shift them based on your utility's priorities.
- **Sustainability**: Increase the carbon penalty to push high-carbon upgrades lower in the ranking. Turn on "Favor low-carbon materials" for an additional penalty.

---

**Step 4 — Export**

Download your results as CSV files or a formatted summary report that you can attach to budget requests, council presentations, or grant applications.
""")

with st.expander("📖 Scoring Methodology & Assumptions"):
    st.markdown(f"""
### How the Priority Score Works

Each asset gets a score from 0 to 100. Higher = more urgent to upgrade. The score is a
weighted average of four factors, each normalized to a 0–1 scale:

| Factor | Current Weight | What It Measures | How It's Normalized |
|--------|---------------|-------------------|---------------------|
| Age | {w_age:.0%} | How old the asset is | Divided by the oldest asset's age |
| Condition | {w_cond:.0%} | Physical state (1=excellent, 5=failed) | (Condition − 1) ÷ 4 |
| Consequence | {w_conseq:.0%} | Impact if it fails (1=minimal, 5=catastrophic) | (Consequence − 1) ÷ 4 |
| Remaining Life | {w_rul:.0%} | Years left vs. expected service life | 1 − (RUL ÷ max RUL across all assets) |

**Formula**: Priority = (Age × w₁ + Condition × w₂ + Consequence × w₃ + RUL × w₄) ÷ (w₁+w₂+w₃+w₄) × 100

**Critical asset boost**: Assets marked as critical (no redundancy) get +10 points added
to their score. This reflects the higher system-level risk of losing a single point of failure.

**Why these default weights?** Condition (35%) gets the most weight because it's the
strongest predictor of near-term failure — an asset in poor condition can fail regardless
of age. Age (25%) and Consequence (25%) are equal because both matter but neither alone
determines urgency. Remaining Life (15%) is a refinement that catches assets near
end-of-life that might look OK on condition alone.

---

### Risk Score (separate from Priority)

Risk Score = (Condition ÷ 5) × (Consequence ÷ 5) × 100

This is the standard risk = likelihood × impact formula used in asset management.
It ranges from 4 (condition 1, consequence 1) to 100 (condition 5, consequence 5).
This score is used in the Risk Matrix and cost efficiency calculations.

---

### How Budget Selection Works

The tool uses a **greedy knapsack** approach:
1. Rank all assets by Priority Score (highest first)
2. Go down the list. If the next asset fits in the remaining budget, fund it.
3. If it doesn't fit, skip it and try the next one.
4. Continue until the budget is used up or all assets are checked.

This is simpler than full mathematical optimization, but for typical utility portfolios
(5–50 assets), it produces results within ~5% of the theoretical optimum while being
completely transparent and easy to explain to stakeholders.

---

### Remaining Useful Life (RUL)

RUL = Expected Service Life − Current Age (minimum 0)

Expected service life comes from industry reference data based on asset type and material:

| Asset Type | Material | Expected Life (yrs) | Source |
|-----------|----------|--------------------:|--------|
| Pipe | Cast Iron | 90 | AWWA M28 |
| Pipe | Ductile Iron | 90 | AWWA M41 |
| Pipe | HDPE | 75 | PPI TR-3 |
| Pipe | PVC | 80 | Uni-Bell |
| Pipe | Steel | 65 | AWWA M11 |
| Pipe | Copper | 70 | CDA |
| Pump | Steel | 20 | AWWA M14 |
| Tank | Concrete | 65 | AWWA D110 |
| Tank | Steel | 50 | AWWA D100 |
| Clarifier | Concrete | 50 | WEF MOP 8 |
| Clarifier | Steel | 40 | WEF MOP 8 |

These are planning-level estimates. Actual service life varies based on soil conditions,
water quality, maintenance history, and operating environment.

---

### Failure Cost (Cost of Doing Nothing)

If an asset fails before planned replacement, emergency repairs typically cost significantly
more than proactive upgrades. The multipliers used are:

| Consequence Level | Multiplier | Example |
|-------------------|-----------|---------|
| 1 – Minimal | 1.5× | Minor leak repair |
| 2 – Low | 2.0× | Service disruption, small area |
| 3 – Moderate | 3.0× | Neighborhood outage, road damage |
| 4 – High | 4.5× | Major service loss, regulatory violation |
| 5 – Catastrophic | 6.0× | System-wide failure, public safety risk |

These ranges are consistent with findings from AWWA's "Buried No Longer" report and
EPA infrastructure assessments, which document reactive-to-proactive cost ratios of
3–10× depending on asset type and failure severity.

---

### Carbon Estimates

Embodied carbon is estimated using: Carbon (t CO₂e) = Carbon Factor × Cost ($K) × 120 ÷ 1000

- **Carbon Factor**: kg CO₂e per kg of material (from ICE Database v3.0 and EPD data)
- **120**: Estimated kg of material per $1,000 of upgrade cost (planning-level assumption)
- The result is in metric tons of CO₂ equivalent

| Material | Carbon Factor (kg CO₂e/kg) |
|----------|---------------------------:|
| Concrete | 0.15 |
| Ductile Iron | 1.80 |
| Cast Iron | 1.90 |
| Steel | 2.10 |
| HDPE | 2.50 |
| PVC | 3.00 |
| Copper | 3.80 |
| FRP | 8.00 |

**Important**: These are order-of-magnitude estimates for comparing material choices during
early planning. They are not detailed Life Cycle Assessment (LCA) values. The "120 kg per
$1K" assumption is a rough industry average — actual material quantities depend on pipe
diameter, wall thickness, and installation method.

---

### Multi-Year Condition Degradation

When an asset is deferred to a future year, the tool simulates aging:
- **Age increases by 1 year** for each year deferred
- **Condition degrades** at a rate based on how close the asset is to end-of-life:
  - Past expected service life (≥100%): **+0.5** condition points per year (rapid decline)
  - Near end-of-life (≥80%): **+0.3** per year (accelerating)
  - Still within service life (<80%): **+0.1** per year (slow, normal aging)
- Condition is capped at 5 (Failed)

This creates realistic urgency: an asset you defer today becomes a higher priority next
year because its condition has worsened, which is how real infrastructure behaves.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="footer">
    <b>Built for</b> water utilities · municipalities · city planners · public works departments<br>
    Supports capital improvement planning, deferred maintenance decisions, and sustainability-aware infrastructure management.<br><br>
    Failure Forecasters — CIVE 580c4 Final Project &nbsp;·&nbsp; Built with Streamlit & Plotly &nbsp;·&nbsp; Sample data for demonstration
</div>
""", unsafe_allow_html=True)
