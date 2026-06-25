# ============================================================
# EPL TACTICAL INTELLIGENCE LAB — Kaggle-only version
# Season focus: Premier League 2024/25
#
# Place these CSVs inside ./epl_data/
#   1) team_stats.csv      -> Premier league stats file
#   2) fbref_pl.csv        -> fbref_PL dataset content
#   3) squad_stats.csv     -> Squad player stats file
#
# Validated schema source:
# - team_stats columns from provided screenshot/paste
# - fbref_pl columns from provided screenshot/paste
# - squad_stats columns from provided screenshot/paste
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="EPL Tactical Intelligence Lab",
    page_icon="load",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# STYLING
# ------------------------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #0b0d11;
    --surface: #131720;
    --surface-2: #1a1f2b;
    --border: #263042;
    --text: #ecf1f7;
    --muted: #8a96ad;
    --accent: #00d4aa;
    --accent-2: #4e9af1;
    --gold: #f0b429;
    --green: #22c55e;
    --red: #ef4444;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg);
}
[data-testid="stHeader"] {
    background: rgba(11,13,17,0.85);
}
section[data-testid="stSidebar"] > div {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

html, body, [class*="css"] {
    color: var(--text);
    font-family: "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

.kpi-card {
    background: linear-gradient(135deg, #121722 0%, #0f1319 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.28);
    position: relative;
    overflow: hidden;
    min-height: 145px;
}
.kpi-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
}
.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.05;
}
.kpi-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.25rem;
}
.badge {
    display: inline-block;
    margin-top: 0.55rem;
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-elite    { color:#00d4aa; background:rgba(0,212,170,0.10); border:1px solid rgba(0,212,170,0.35); }
.badge-strong   { color:#22c55e; background:rgba(34,197,94,0.10); border:1px solid rgba(34,197,94,0.35); }
.badge-moderate { color:#f0b429; background:rgba(240,180,41,0.10); border:1px solid rgba(240,180,41,0.35); }
.badge-low      { color:#ef4444; background:rgba(239,68,68,0.10); border:1px solid rgba(239,68,68,0.35); }

.section-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1rem 0.4rem 1rem;
    margin-bottom: 1rem;
}
.note-box {
    background: rgba(78,154,241,0.08);
    border: 1px solid rgba(78,154,241,0.25);
    color: #b6c8ea;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    font-size: 0.82rem;
}
.small-muted {
    color: var(--muted);
    font-size: 0.8rem;
}

[data-testid="stTabs"] button {
    color: var(--muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

PLOT_BG = "#131720"
PAPER_BG = "#0b0d11"
FONT_COLOR = "#ecf1f7"
GRID = "#253143"

# ------------------------------------------------------------
# DATA HELPERS
# ------------------------------------------------------------
DATA_DIR = "data"

TEAM_REQUIRED = [
    "Rk","Squad","MP","W","D","L","GF","GA","GD","Pts","Pts/MP",
    "xG","xGA","xGD","xGD/90","Attendance","Top Team Scorer","Goalkeeper","Notes"
]

FBREF_REQUIRED = [
    "Rk","Player","Nation","Pos","Squad","Age","Born","MP","Starts","Min","90s",
    "Gls","Ast","G+A","G-PK","PK","PKatt","CrdY","CrdR",
    "xG","npxG","xAG","npxG+xAG","PrgC","PrgP","PrgR"
]

SQUAD_REQUIRED = [
    "Rk","Player","Nation","Pos","Squad","Age","Born",
    "Playing Time_MP","Playing Time_Starts","Playing Time_Min","Playing Time_90s",
    "Performance_Gls","Performance_Ast","Performance_G+A","Performance_G-PK",
    "Performance_PK","Performance_PKatt","Performance_CrdY","Performance_CrdR",
    "Per 90 Minutes_Gls","Per 90 Minutes_Ast","Per 90 Minutes_G+A",
    "Per 90 Minutes_G-PK","Per 90 Minutes_G+A-PK","Matches"
]

def locate_csv(possible_names):
    for name in possible_names:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None

def validate_columns(df, required, dataset_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

@st.cache_data(show_spinner=True)
def load_data():
    team_path = locate_csv(["team_stats.csv", "premier_league_stats.csv", "Premier league stats file.csv"])
    fbref_path = locate_csv(["fbref_pl.csv", "fbref_PL.csv", "fbref player stats.csv"])
    squad_path = locate_csv(["squad_stats.csv", "squad_player_stats.csv", "Squad player stats file.csv"])

    if not team_path or not fbref_path or not squad_path:
        raise FileNotFoundError(
            "Could not find all three CSVs in ./data/. Expected files like:\n"
            "team_stats.csv, fbref_pl.csv, squad_stats.csv"
        )

    team = pd.read_csv(team_path)
    fbref = pd.read_csv(fbref_path)
    squad = pd.read_csv(squad_path)

    team.columns = [c.strip() for c in team.columns]
    fbref.columns = [c.strip() for c in fbref.columns]
    squad.columns = [c.strip() for c in squad.columns]

    validate_columns(team, TEAM_REQUIRED, "team_stats")
    validate_columns(fbref, FBREF_REQUIRED, "fbref_pl")
    validate_columns(squad, SQUAD_REQUIRED, "squad_stats")

    # numeric conversion
    team_num = ["MP","W","D","L","GF","GA","GD","Pts","Pts/MP","xG","xGA","xGD","xGD/90","Attendance"]
    for c in team_num:
        team[c] = pd.to_numeric(team[c], errors="coerce")

    fbref_num = ["Age","Born","MP","Starts","Min","90s","Gls","Ast","G+A","G-PK","PK","PKatt",
                 "CrdY","CrdR","xG","npxG","xAG","npxG+xAG","PrgC","PrgP","PrgR"]
    for c in fbref_num:
        fbref[c] = pd.to_numeric(fbref[c], errors="coerce")

    squad_num = ["Age","Born","Playing Time_MP","Playing Time_Starts","Playing Time_Min",
                 "Playing Time_90s","Performance_Gls","Performance_Ast","Performance_G+A",
                 "Performance_G-PK","Performance_PK","Performance_PKatt",
                 "Performance_CrdY","Performance_CrdR",
                 "Per 90 Minutes_Gls","Per 90 Minutes_Ast","Per 90 Minutes_G+A",
                 "Per 90 Minutes_G-PK","Per 90 Minutes_G+A-PK"]
    for c in squad_num:
        squad[c] = pd.to_numeric(squad[c], errors="coerce")

    return team, fbref, squad

# ------------------------------------------------------------
# KPI LOGIC
# ------------------------------------------------------------
def add_player_metrics(fbref):
    df = fbref.copy()
    df["xG_90_calc"] = np.where(df["90s"] > 0, df["xG"] / df["90s"], 0)
    df["xAG_90_calc"] = np.where(df["90s"] > 0, df["xAG"] / df["90s"], 0)
    df["ATI"] = 0.6 * df["xG_90_calc"] + 0.4 * df["xAG_90_calc"]
    df["SID_attack_idx"] = 0.7 * df["xG_90_calc"] + 0.3 * df["xAG_90_calc"]
    df["Progression_90"] = np.where(df["90s"] > 0, (df["PrgC"] + df["PrgP"]) / df["90s"], 0)
    return df

def add_team_metrics(team, fbref):
    t = team.copy()

    # SQR
    t["xG_per_match"] = np.where(t["MP"] > 0, t["xG"] / t["MP"], np.nan)
    league_xg_pm = t["xG_per_match"].mean()
    t["SQR"] = t["xG_per_match"] / league_xg_pm

    # DCS proxy adaptation
    # Original DCS requires tackles/interceptions/clearances, which are not present.
    # Proxy = points-rate per xGA per match.
    t["xGA_per_match"] = np.where(t["MP"] > 0, t["xGA"] / t["MP"], np.nan)
    t["def_proxy"] = ((t["W"] + 0.5 * t["D"]) / t["MP"]) * 100
    t["DCS"] = t["def_proxy"] / t["xGA_per_match"]

    # Team ATI from top-3 outfield ATI players
    outfield = fbref[~fbref["Pos"].astype(str).str.contains("GK", na=False)].copy()
    team_ati = (
        outfield.groupby("Squad")["ATI"]
        .apply(lambda s: s.nlargest(min(3, len(s))).mean() if len(s) else 0)
        .reset_index()
        .rename(columns={"ATI": "Team_ATI"})
    )
    t = t.merge(team_ati, on="Squad", how="left")

    # FTPE from player progression fields
    prog = (
        fbref.groupby("Squad")
        .agg(total_prgc=("PrgC", "sum"),
             total_prgp=("PrgP", "sum"),
             total_90s=("90s", "sum"))
        .reset_index()
    )
    prog["FTPE"] = np.where(prog["total_90s"] > 0, (prog["total_prgc"] + prog["total_prgp"]) / prog["total_90s"], 0)
    t = t.merge(prog[["Squad", "FTPE"]], on="Squad", how="left")

    return t

def ati_band(v):
    if v > 0.60: return "Elite attacking contributor", "badge-elite"
    if 0.35 <= v <= 0.60: return "Strong contributor", "badge-strong"
    if 0.15 <= v < 0.35: return "Moderate", "badge-moderate"
    return "Low attacking involvement", "badge-low"

def sqr_band(v):
    if v > 1.3: return "Excellent chance quality", "badge-elite"
    if 1.0 <= v <= 1.3: return "Above average", "badge-strong"
    if 0.8 <= v < 1.0: return "Average", "badge-moderate"
    return "Poor shot quality", "badge-low"

def dcs_band(v):
    if v > 60: return "Excellent defensive organisation", "badge-elite"
    if 40 <= v <= 60: return "Solid", "badge-strong"
    if 20 <= v < 40: return "Moderate", "badge-moderate"
    return "Poor", "badge-low"

def sid_band(v):
    if v > 0.20: return "Strong positive sub", "badge-elite"
    if 0.05 <= v <= 0.20: return "Mild improvement", "badge-strong"
    if -0.05 <= v < 0.05: return "Neutral", "badge-moderate"
    return "Negative", "badge-low"

# ------------------------------------------------------------
# SIMULATION LOGIC
# ------------------------------------------------------------
FORMATION_MULTIPLIERS = {
    "4-3-3":   (1.08, 0.96),
    "4-2-3-1": (1.03, 0.98),
    "3-5-2":   (1.02, 1.02),
    "5-3-2":   (0.94, 1.06),
    "4-4-2":   (1.00, 1.00),
    "3-4-3":   (1.10, 0.95),
    "5-4-1":   (0.90, 1.08),
    "4-1-4-1": (1.01, 1.00),
}

def simulate_poisson(lam_for, lam_against, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    gf = rng.poisson(lam_for, size=n)
    ga = rng.poisson(lam_against, size=n)
    win = (gf > ga).mean()
    draw = (gf == ga).mean()
    loss = (gf < ga).mean()
    return win, draw, loss

def compute_fwpl(team_row, current_formation, new_formation):
    base_for = team_row["xG"] / team_row["MP"]
    base_against = team_row["xGA"] / team_row["MP"]

    att_c, def_c = FORMATION_MULTIPLIERS[current_formation]
    att_n, def_n = FORMATION_MULTIPLIERS[new_formation]

    curr = simulate_poisson(base_for * att_c, base_against * def_c, n=10000, seed=42)
    new = simulate_poisson(base_for * att_n, base_against * def_n, n=10000, seed=42)

    fwpl = new[0] - curr[0]
    return curr, new, fwpl

def compute_sid(remaining_minutes, baseline_xg_pm, baseline_xga_pm,
                atk_out, atk_in, def_out, def_in):
    alpha = 0.4
    beta = 0.4

    delta_atk = atk_in - atk_out
    delta_def = def_in - def_out

    xg_created_base = baseline_xg_pm * remaining_minutes
    xg_conceded_base = baseline_xga_pm * remaining_minutes

    xg_created_new = xg_created_base * (1 + alpha * delta_atk)
    xg_conceded_new = xg_conceded_base * (1 - beta * delta_def)

    sid = (xg_created_new - xg_created_base) + (xg_conceded_base - xg_conceded_new)
    return sid, xg_created_base, xg_created_new, xg_conceded_base, xg_conceded_new

# ------------------------------------------------------------
# UI HELPERS
# ------------------------------------------------------------
def kpi_card(label, value, subtitle, band_text, band_class):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
        <div class="badge {band_class}">{band_text}</div>
    </div>
    """, unsafe_allow_html=True)

def style_plot(fig, title=None, height=420):
    fig.update_layout(
        title=title,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="Segoe UI"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    return fig

def build_pitch():
    fig = go.Figure()

    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor="#1b5e20",
        xaxis=dict(range=[0, 120], visible=False),
        yaxis=dict(range=[0, 80], visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
    )

    shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="white", width=2)),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="white", width=2)),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color="white", width=2)),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="white", width=2)),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="white", width=2)),
        dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="white", width=2)),
        dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="white", width=2)),
    ]
    fig.update_layout(shapes=shapes)
    return fig

def simulated_player_shot_map(df_team):
    rng = np.random.default_rng(7)
    df = df_team.copy()
    df = df[df["90s"] > 0].copy()

    if df.empty:
        return build_pitch()

    # pseudo-spatial adaptation using real xG and progression, because validated schemas lack shot coordinates
    df["pitch_x"] = 70 + np.clip(df["xG_90_calc"] * 60 + rng.normal(0, 8, len(df)), 0, 45)
    df["pitch_y"] = np.clip(40 + rng.normal(0, 14, len(df)), 5, 75)
    df["dot_size"] = np.clip(df["xG_90_calc"] * 65 + 8, 8, 40)
    df["goal_color"] = np.where(df["Gls"] > 0, "#00d4aa", "#ef4444")

    fig = build_pitch()
    fig.add_trace(go.Scatter(
        x=df["pitch_x"],
        y=df["pitch_y"],
        mode="markers+text",
        text=df["Player"],
        textposition="top center",
        marker=dict(
            size=df["dot_size"],
            color=df["goal_color"],
            opacity=0.78,
            line=dict(color="#0b0d11", width=1)
        ),
        customdata=np.stack([df["xG_90_calc"], df["xAG_90_calc"], df["ATI"]], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "xG/90: %{customdata[0]:.2f}<br>"
            "xAG/90: %{customdata[1]:.2f}<br>"
            "ATI: %{customdata[2]:.2f}<extra></extra>"
        ),
        name="Players"
    ))
    fig.update_layout(title="Pseudo-spatial attacking threat map (size = xG/90)")
    return fig

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    st.title("EPL Tactical Intelligence Lab")
    st.caption("Premium 2024/25 Premier League dashboard built from the validated Kaggle schemas you shared.")

    try:
        team_raw, fbref_raw, squad_raw = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    fbref = add_player_metrics(fbref_raw)
    team = add_team_metrics(team_raw, fbref)

    teams = sorted(team["Squad"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown("### Filters")
        selected_team = st.selectbox("Team", teams)
        position_filter = st.multiselect("Position", ["FW", "MF", "DF", "GK"], default=["FW", "MF", "DF"])
        min_90s = st.slider("Minimum 90s", 0.0, float(max(1.0, fbref["90s"].max())), 3.0, 0.5)
        compare_team = st.selectbox("Comparison team", teams, index=min(1, len(teams)-1))

        st.markdown("---")
        st.markdown("### Data files expected")
        st.code(
            "data/\n"
            "├─ team_stats.csv\n"
            "├─ fbref_pl.csv\n"
            "└─ squad_stats.csv",
            language="text"
        )

    filtered_players = fbref[
        (fbref["90s"] >= min_90s) &
        (fbref["Pos"].astype(str).apply(lambda x: any(p in x for p in position_filter)))
    ].copy()

    selected_team_row = team[team["Squad"] == selected_team].iloc[0]
    compare_team_row = team[team["Squad"] == compare_team].iloc[0]

    team_players = filtered_players[filtered_players["Squad"] == selected_team].copy()
    compare_players = filtered_players[filtered_players["Squad"] == compare_team].copy()

    tabs = st.tabs([
        "Executive Overview",
        "Team Analysis",
        "Player Deep Dive",
        "Pitch Visual",
        "What-If Lab",
        "Methodology",
    ])

    # --------------------------------------------------------
    # 1) Executive Overview
    # --------------------------------------------------------
    with tabs[0]:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader(f"{selected_team} — Executive Overview")

        ati_val = float(selected_team_row["Team_ATI"])
        sqr_val = float(selected_team_row["SQR"])
        dcs_val = float(selected_team_row["DCS"])
        ftpe_val = float(selected_team_row["FTPE"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            btxt, bcls = ati_band(ati_val)
            kpi_card("Attacking Threat Index", f"{ati_val:.3f}", "Top-3 player ATI average", btxt, bcls)
        with c2:
            btxt, bcls = sqr_band(sqr_val)
            kpi_card("Shot Quality Ratio", f"{sqr_val:.3f}", "Team xG/match vs league avg", btxt, bcls)
        with c3:
            btxt, bcls = dcs_band(dcs_val)
            kpi_card("Defensive Compactness", f"{dcs_val:.1f}", "Proxy from results-rate and xGA", btxt, bcls)
        with c4:
            kpi_card("Final-Third Penetration", f"{ftpe_val:.2f}", "PrgC + PrgP per 90", "Advanced metric", "badge-strong")

        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            rank_df = team.sort_values("SQR", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rank_df["SQR"],
                y=rank_df["Squad"],
                orientation="h",
                marker_color=["#00d4aa" if x == selected_team else "#4e9af1" for x in rank_df["Squad"]],
                text=rank_df["SQR"].round(3),
                textposition="outside"
            ))
            fig.add_vline(x=1.0, line_dash="dash", line_color="#f0b429")
            style_plot(fig, "League ranking — SQR", 500)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            rank_df = team.sort_values("DCS", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rank_df["DCS"],
                y=rank_df["Squad"],
                orientation="h",
                marker_color=["#00d4aa" if x == selected_team else "#a78bfa" for x in rank_df["Squad"]],
                text=rank_df["DCS"].round(1),
                textposition="outside"
            ))
            style_plot(fig, "League ranking — DCS", 500)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            team, x="xGA", y="xG", text="Squad", size="Pts", color="Pts",
            color_continuous_scale="Tealgrn",
            labels={"xGA":"Season xGA", "xG":"Season xG"}
        )
        max_val = max(team["xG"].max(), team["xGA"].max()) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color="#f0b429", width=1, dash="dot"))
        fig.update_traces(textposition="top center")
        style_plot(fig, "Season profile — xG vs xGA (bubble size = points)", 430)
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # 2) Team Analysis
    # --------------------------------------------------------
    with tabs[1]:
        st.subheader("Team Analysis")

        radar_metrics = ["Team_ATI", "SQR", "DCS", "FTPE", "xGD/90"]
        radar_labels = ["ATI", "SQR", "DCS", "FTPE", "xGD/90"]

        norm_team = team.copy()
        for m in radar_metrics:
            m_min = norm_team[m].min()
            m_max = norm_team[m].max()
            norm_team[m + "_norm"] = (norm_team[m] - m_min) / (m_max - m_min + 1e-9)

        sel_norm = norm_team[norm_team["Squad"] == selected_team].iloc[0]
        cmp_norm = norm_team[norm_team["Squad"] == compare_team].iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[sel_norm[m + "_norm"] for m in radar_metrics] + [sel_norm[radar_metrics[0] + "_norm"]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=selected_team,
            line=dict(color="#00d4aa", width=2),
            fillcolor="rgba(0,212,170,0.10)"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[cmp_norm[m + "_norm"] for m in radar_metrics] + [cmp_norm[radar_metrics[0] + "_norm"]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=compare_team,
            line=dict(color="#4e9af1", width=2),
            fillcolor="rgba(78,154,241,0.08)"
        ))
        fig.update_layout(
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COLOR),
            polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID),
                angularaxis=dict(gridcolor=GRID)
            ),
            height=500,
            title="Normalized team radar"
        )
        st.plotly_chart(fig, use_container_width=True)

        table_cols = ["Rk","Squad","MP","W","D","L","GF","GA","Pts","xG","xGA","xGD/90","SQR","DCS","FTPE","Team_ATI"]
        st.dataframe(
            team[table_cols].sort_values("Pts", ascending=False).reset_index(drop=True).style.format({
                "xG":"{:.1f}", "xGA":"{:.1f}", "xGD/90":"{:.2f}", "SQR":"{:.3f}",
                "DCS":"{:.1f}", "FTPE":"{:.2f}", "Team_ATI":"{:.3f}"
            }),
            use_container_width=True,
            height=420
        )

        fig = px.scatter(
            team, x="xGD", y="Pts", size="FTPE", color="SQR", text="Squad",
            color_continuous_scale="Tealgrn"
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#8a96ad")
        fig.update_traces(textposition="top center")
        style_plot(fig, "Points vs xGD (bubble size = FTPE)", 430)
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # 3) Player Deep Dive
    # --------------------------------------------------------
    with tabs[2]:
        st.subheader("Player Deep Dive")

        if team_players.empty:
            st.warning("No players match the current filters.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                top_players = filtered_players.sort_values("ATI", ascending=False).head(20)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_players["ATI"],
                    y=top_players["Player"],
                    orientation="h",
                    marker_color=["#00d4aa" if s == selected_team else "#4e9af1" for s in top_players["Squad"]],
                    text=top_players["ATI"].round(3),
                    textposition="outside"
                ))
                fig.update_yaxes(autorange="reversed")
                style_plot(fig, "Top 20 players by ATI", 520)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.scatter(
                    filtered_players, x="xG_90_calc", y="xAG_90_calc",
                    color="ATI", size="90s", text="Player",
                    hover_data=["Squad","Pos"],
                    color_continuous_scale="Teal"
                )
                highlight = filtered_players[filtered_players["Squad"] == selected_team]
                fig.add_trace(go.Scatter(
                    x=highlight["xG_90_calc"],
                    y=highlight["xAG_90_calc"],
                    mode="markers",
                    marker=dict(color="#00d4aa", size=12, symbol="diamond", line=dict(color="#0b0d11", width=1.2)),
                    text=highlight["Player"],
                    name=selected_team,
                    hovertemplate="<b>%{text}</b><br>xG/90: %{x:.2f}<br>xAG/90: %{y:.2f}<extra></extra>"
                ))
                style_plot(fig, "xG/90 vs xAG/90", 520)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Team player table")
            view_cols = ["Player","Pos","90s","Gls","Ast","xG","xAG","PrgC","PrgP","ATI"]
            st.dataframe(
                team_players[view_cols].sort_values("ATI", ascending=False).reset_index(drop=True).style.format({
                    "90s":"{:.1f}", "xG":"{:.2f}", "xAG":"{:.2f}", "ATI":"{:.3f}"
                }),
                use_container_width=True,
                height=420
            )

            fig = px.box(
                filtered_players, x="Pos", y="xG_90_calc", color="Pos",
                points="outliers",
                color_discrete_sequence=["#00d4aa","#4e9af1","#f0b429","#ef4444"]
            )
            style_plot(fig, "xG/90 distribution by position", 360)
            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # 4) Pitch Visual
    # --------------------------------------------------------
    with tabs[3]:
        st.subheader("Pitch Visual")

        st.markdown("""
        <div class="note-box">
        This visual is a <b>pseudo-spatial adaptation</b>. The validated Kaggle schemas contain real player attacking
        values (<code>xG</code>, <code>xAG</code>, <code>PrgC</code>, <code>PrgP</code>) but not real shot coordinates,
        so the dashboard maps player threat into realistic attacking zones for presentation quality without fabricating
        the underlying KPI values.
        </div>
        """, unsafe_allow_html=True)

        pitch_players = team_players[~team_players["Pos"].astype(str).str.contains("GK", na=False)].copy()
        fig = simulated_player_shot_map(pitch_players)
        st.plotly_chart(fig, use_container_width=True)

        show_pitch_cols = ["Player","Pos","90s","xG_90_calc","xAG_90_calc","ATI","PrgC","PrgP"]
        st.dataframe(
            pitch_players[show_pitch_cols].sort_values("ATI", ascending=False).reset_index(drop=True).style.format({
                "90s":"{:.1f}", "xG_90_calc":"{:.2f}", "xAG_90_calc":"{:.2f}", "ATI":"{:.3f}"
            }),
            use_container_width=True,
            height=320
        )

    # --------------------------------------------------------
    # 5) What-If Lab
    # --------------------------------------------------------
    with tabs[4]:
        sub_tabs = st.tabs(["Formation Win Probability Lift", "Substitution Impact Delta"])

        with sub_tabs[0]:
            st.subheader("Formation Win Probability Lift (FWPL)")

            c1, c2 = st.columns(2)
            with c1:
                current_formation = st.selectbox("Current formation", list(FORMATION_MULTIPLIERS.keys()), index=0)
            with c2:
                new_formation = st.selectbox("Proposed formation", list(FORMATION_MULTIPLIERS.keys()), index=2)

            current_probs, new_probs, fwpl = compute_fwpl(selected_team_row, current_formation, new_formation)

            result_df = pd.DataFrame({
                "Outcome": ["Win", "Draw", "Loss"],
                current_formation: [current_probs[0], current_probs[1], current_probs[2]],
                new_formation: [new_probs[0], new_probs[1], new_probs[2]]
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=result_df["Outcome"], y=result_df[current_formation],
                name=current_formation, marker_color="#4e9af1"
            ))
            fig.add_trace(go.Bar(
                x=result_df["Outcome"], y=result_df[new_formation],
                name=new_formation, marker_color="#00d4aa"
            ))
            fig.update_layout(barmode="group")
            style_plot(fig, f"Outcome probabilities — {selected_team}", 420)
            st.plotly_chart(fig, use_container_width=True)

            uplift_txt = f"{fwpl:+.3f}"
            badge = "badge-strong" if fwpl > 0 else "badge-low" if fwpl < 0 else "badge-moderate"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">FWPL</div>
                <div class="kpi-value">{uplift_txt}</div>
                <div class="kpi-sub">Simulated win% (new formation) − simulated win% (current formation)</div>
                <div class="badge {badge}">{new_formation} vs {current_formation}</div>
            </div>
            """, unsafe_allow_html=True)

        with sub_tabs[1]:
            st.subheader("Substitution Impact Delta (SID)")

            if team_players.empty or len(team_players) < 2:
                st.warning("Not enough player rows for SID.")
            else:
                player_options = team_players.sort_values("ATI", ascending=False)["Player"].tolist()
                out_player = st.selectbox("Outgoing player", player_options, index=min(1, len(player_options)-1))
                in_player = st.selectbox("Incoming player", player_options, index=0)

                out_row = team_players[team_players["Player"] == out_player].iloc[0]
                in_row = team_players[team_players["Player"] == in_player].iloc[0]

                remaining_minutes = st.slider("Remaining match minutes", 5, 60, 30)

                # attacking index direct from data
                atk_out = float(out_row["SID_attack_idx"])
                atk_in = float(in_row["SID_attack_idx"])

                # defensive index unavailable in validated schemas -> manual scenario input
                c1, c2 = st.columns(2)
                with c1:
                    def_out = st.slider("Outgoing player defensive index", 0.0, 1.5, 0.50, 0.05)
                with c2:
                    def_in = st.slider("Incoming player defensive index", 0.0, 1.5, 0.75, 0.05)

                baseline_xg_pm = float(selected_team_row["xG"]) / (float(selected_team_row["MP"]) * 90)
                baseline_xga_pm = float(selected_team_row["xGA"]) / (float(selected_team_row["MP"]) * 90)

                sid, base_create, new_create, base_concede, new_concede = compute_sid(
                    remaining_minutes,
                    baseline_xg_pm,
                    baseline_xga_pm,
                    atk_out,
                    atk_in,
                    def_out,
                    def_in,
                )

                btxt, bcls = sid_band(sid)
                c1, c2, c3 = st.columns(3)
                with c1:
                    kpi_card("SID", f"{sid:+.3f}", "Net impact on remaining match xG balance", btxt, bcls)
                with c2:
                    kpi_card("Base xG Created", f"{base_create:.2f}", f"Next {remaining_minutes} mins", "Baseline", "badge-moderate")
                with c3:
                    kpi_card("New xG Created", f"{new_create:.2f}", "After substitution scenario", "Scenario", "badge-strong")

                st.markdown("#### SID scenario summary")
                sid_df = pd.DataFrame({
                    "Metric": ["Attack index out", "Attack index in", "Def index out", "Def index in",
                               "Baseline xG created", "New xG created", "Baseline xG conceded", "New xG conceded"],
                    "Value": [atk_out, atk_in, def_out, def_in, base_create, new_create, base_concede, new_concede]
                })
                st.dataframe(sid_df.style.format({"Value":"{:.3f}"}), use_container_width=True)

                st.markdown("""
                <div class="note-box">
                SID uses the validated player attacking fields directly, but the defensive side is a coach-input
                scenario parameter because the confirmed schemas do not contain tackles/interceptions/clearances.
                </div>
                """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # 6) Methodology
    # --------------------------------------------------------
    with tabs[5]:
        st.subheader("Methodology and Validation")

        st.markdown("""
        ### Validated datasets
        - **fbref_pl** supplies player `xG`, `xAG`, `90s`, `PrgC`, `PrgP`, `PrgR`, goals, assists, cards, and squad identity.
        - **team_stats** supplies team `MP`, `W`, `D`, `L`, `GF`, `GA`, `Pts`, `xG`, `xGA`, `xGD`, and `xGD/90`.
        - **squad_stats** supplies a secondary player table with playing time and base production fields.

        ### KPI implementation
        - **ATI** = `0.6 * (xG / 90s) + 0.4 * (xAG / 90s)`.
        - **SQR** = `(team xG / MP) / league average (xG / MP)`.
        - **FTPE** = `(sum PrgC + sum PrgP) / sum 90s` per squad.
        - **FWPL** uses a 10,000-run Poisson simulation from team `xG/MP` and `xGA/MP`, then applies formation multipliers.
        - **SID** uses player attacking indices from actual player data plus manual defensive scenario sliders.

        ### DCS adaptation
        The original DCS specification requires:
        `tackles won + interceptions + clearances` divided by `xG conceded`.

        Those defensive action fields were **not present** in the validated schemas you shared, so this app uses:
        **DCS proxy = points-rate per xGA-per-match**.

        This proxy is transparent, reproducible, and better than inventing non-existent defensive-action columns.

        ### Why some requested filters are absent
        The validated schemas do not show match-level opponent, venue, minute-by-minute event, formation history, or shot-coordinate fields.
        Therefore, those controls are intentionally not fabricated into the app.
        """)

    st.markdown("---")
    st.caption("Built for an MSc-level football analytics submission • Kaggle-only schema-validated implementation")

if __name__ == "__main__":
    main()