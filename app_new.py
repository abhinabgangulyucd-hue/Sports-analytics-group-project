import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Football Coaching Intelligence Dashboard",
    page_icon="Football",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME = {
    "bg": "#0f172a",
    "surface": "#111827",
    "surface_2": "#1f2937",
    "surface_3": "#243244",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "border": "#334155",
    "accent": "#14b8a6",
    "accent_2": "#60a5fa",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "success": "#22c55e",
}

FORMATION_MULTIPLIERS = {
    "4-3-3": (1.08, 0.96),
    "4-2-3-1": (1.03, 0.98),
    "3-5-2": (1.02, 1.02),
    "5-3-2": (0.94, 1.06),
    "4-4-2": (1.00, 1.00),
    "3-4-3": (1.10, 0.95),
    "5-4-1": (0.90, 1.08),
    "4-1-4-1": (1.01, 1.00),
}

TEAM_REQUIRED = [
    "Rk", "Squad", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "Pts/MP",
    "xG", "xGA", "xGD", "xGD/90", "Attendance", "Top Team Scorer", "Goalkeeper", "Notes",
]

FBREF_REQUIRED = [
    "Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR",
    "xG", "npxG", "xAG", "npxG+xAG", "PrgC", "PrgP", "PrgR",
]

SQUAD_REQUIRED = [
    "Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born",
    "Playing Time_MP", "Playing Time_Starts", "Playing Time_Min", "Playing Time_90s",
    "Performance_Gls", "Performance_Ast", "Performance_G+A", "Performance_G-PK",
    "Performance_PK", "Performance_PKatt", "Performance_CrdY", "Performance_CrdR",
    "Per 90 Minutes_Gls", "Per 90 Minutes_Ast", "Per 90 Minutes_G+A",
    "Per 90 Minutes_G-PK", "Per 90 Minutes_G+A-PK", "Matches",
]

NUMERIC_TEAM = ["MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "Pts/MP", "xG", "xGA", "xGD", "xGD/90", "Attendance"]
NUMERIC_FBREF = ["Age", "Born", "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR", "xG", "npxG", "xAG", "npxG+xAG", "PrgC", "PrgP", "PrgR"]
NUMERIC_SQUAD = [
    "Age", "Born", "Playing Time_MP", "Playing Time_Starts", "Playing Time_Min", "Playing Time_90s",
    "Performance_Gls", "Performance_Ast", "Performance_G+A", "Performance_G-PK", "Performance_PK",
    "Performance_PKatt", "Performance_CrdY", "Performance_CrdR", "Per 90 Minutes_Gls",
    "Per 90 Minutes_Ast", "Per 90 Minutes_G+A", "Per 90 Minutes_G-PK", "Per 90 Minutes_G+A-PK",
]

st.markdown(
    f"""
<style>
:root {{
    --bg: {THEME['bg']};
    --surface: {THEME['surface']};
    --surface-2: {THEME['surface_2']};
    --surface-3: {THEME['surface_3']};
    --text: {THEME['text']};
    --muted: {THEME['muted']};
    --border: {THEME['border']};
    --accent: {THEME['accent']};
    --accent2: {THEME['accent_2']};
    --warning: {THEME['warning']};
    --danger: {THEME['danger']};
    --success: {THEME['success']};
}}
html, body, [class*="css"] {{ color: var(--text); font-family: Inter, Segoe UI, sans-serif; }}
[data-testid="stAppViewContainer"] {{ background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%); }}
[data-testid="stHeader"] {{ background: rgba(15, 23, 42, 0.75); }}
[data-testid="stSidebar"] > div:first-child {{ background: var(--surface); border-right: 1px solid var(--border); }}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; display: none; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1180px; }}
    [data-testid="stSidebar"] > div:first-child {{ min-width: 280px; max-width: 280px; padding: 0.8rem 0.9rem 1.0rem; }}
h1, h2, h3 {{ color: var(--text) !important; letter-spacing: -0.02em; }}

.dashboard-card {{
    background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(15,23,42,0.96));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    box-shadow: 0 16px 36px rgba(2, 6, 23, 0.28);
}}
.metric-card {{
    background: linear-gradient(180deg, rgba(17,24,39,1), rgba(16,24,39,0.88));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1rem 0.95rem 1rem;
    min-height: 148px;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: "";
    position: absolute;
    left: 0; top: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}}
.metric-label {{ font-size: 0.76rem; letter-spacing: 0.08em; color: var(--muted); text-transform: uppercase; }}
.metric-value {{ font-size: 2rem; font-weight: 750; line-height: 1.0; color: var(--accent); margin-top: 0.35rem; }}
.metric-sub {{ color: var(--muted); font-size: 0.83rem; margin-top: 0.3rem; }}
.badge {{
    display: inline-block; margin-top: 0.65rem; padding: 0.22rem 0.65rem; border-radius: 999px;
    font-size: 0.71rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
}}
.badge-elite {{ color: #2dd4bf; background: rgba(45,212,191,0.10); border: 1px solid rgba(45,212,191,0.34); }}
.badge-strong {{ color: #4ade80; background: rgba(74,222,128,0.10); border: 1px solid rgba(74,222,128,0.34); }}
.badge-moderate {{ color: #fbbf24; background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.34); }}
.badge-low {{ color: #f87171; background: rgba(248,113,113,0.10); border: 1px solid rgba(248,113,113,0.34); }}
.badge-info {{ color: #93c5fd; background: rgba(147,197,253,0.10); border: 1px solid rgba(147,197,253,0.34); }}
.info-box {{
    background: rgba(20,184,166,0.08); border: 1px solid rgba(20,184,166,0.25); border-radius: 16px;
    padding: 0.95rem 1rem; color: #d1fae5;
}}
.warning-box {{
    background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.25); border-radius: 16px;
    padding: 0.95rem 1rem; color: #fde68a;
}}
.small-muted {{ color: var(--muted); font-size: 0.84rem; }}
.divider-space {{ margin: 0.6rem 0 0.9rem 0; }}
[data-testid="stTabs"] button {{ color: var(--muted) !important; border-radius: 12px 12px 0 0; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: var(--text) !important; }}
[data-testid="stMetricValue"] {{ color: var(--text); }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stSlider {{ border-radius: 12px; }}
</style>
""",
    unsafe_allow_html=True,
)


def locate_csv(possible_names: List[str]) -> str | None:
    search_dirs = [".", "data", "/home/user", "/home/user/data"]
    for base in search_dirs:
        for name in possible_names:
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path
    return None


def validate_columns(df: pd.DataFrame, required: List[str], dataset_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


@st.cache_data(show_spinner=True)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_path = locate_csv(["team_stats.csv", "premier_league_stats.csv", "Premier league stats file.csv"])
    fbref_path = locate_csv(["fbref_pl.csv", "fbref_PL.csv", "fbref player stats.csv"])
    squad_path = locate_csv(["squad_stats.csv", "squad_player_stats.csv", "Squad player stats file.csv"])

    if not team_path or not fbref_path or not squad_path:
        raise FileNotFoundError("Could not find team_stats.csv, fbref_pl.csv and squad_stats.csv in the working directory or /data.")

    team = pd.read_csv(team_path)
    fbref = pd.read_csv(fbref_path)
    squad = pd.read_csv(squad_path)

    team.columns = [c.strip() for c in team.columns]
    fbref.columns = [c.strip() for c in fbref.columns]
    squad.columns = [c.strip() for c in squad.columns]

    validate_columns(team, TEAM_REQUIRED, "team_stats")
    validate_columns(fbref, FBREF_REQUIRED, "fbref_pl")
    validate_columns(squad, SQUAD_REQUIRED, "squad_stats")

    for c in NUMERIC_TEAM:
        team[c] = pd.to_numeric(team[c], errors="coerce")
    for c in NUMERIC_FBREF:
        fbref[c] = pd.to_numeric(fbref[c], errors="coerce")
    for c in NUMERIC_SQUAD:
        squad[c] = pd.to_numeric(squad[c], errors="coerce")

    return team, fbref, squad


def primary_role(pos: str) -> str:
    pos = str(pos)
    if "GK" in pos:
        return "GK"
    if pos.startswith("DF"):
        return "DEF"
    if pos.startswith("MF"):
        return "MID"
    if pos.startswith("FW"):
        return "FWD"
    return "HYBRID"



def position_bucket(pos: str) -> str:
    pos = str(pos)
    markers = [m for m in ["GK", "DF", "MF", "FW"] if m in pos]
    if len(markers) > 1:
        return "Hybrid"
    mapping = {"GK": "Goalkeeper", "DF": "Defender", "MF": "Midfielder", "FW": "Forward"}
    return mapping.get(markers[0], "Hybrid") if markers else "Hybrid"



def add_player_metrics(fbref: pd.DataFrame, squad: pd.DataFrame) -> pd.DataFrame:
    df = fbref.copy()
    df["xG_90"] = np.where(df["90s"] > 0, df["xG"] / df["90s"], 0)
    df["xAG_90"] = np.where(df["90s"] > 0, df["xAG"] / df["90s"], 0)
    df["G+A_90"] = np.where(df["90s"] > 0, df["G+A"] / df["90s"], 0)
    df["Progression_90"] = np.where(df["90s"] > 0, (df["PrgC"] + df["PrgP"]) / df["90s"], 0)
    df["Progressive_Receptions_90"] = np.where(df["90s"] > 0, df["PrgR"] / df["90s"], 0)
    df["ATI"] = 0.6 * df["xG_90"] + 0.4 * df["xAG_90"]
    df["SID_attack_idx"] = 0.7 * df["xG_90"] + 0.3 * df["xAG_90"]
    df["Selection_Load"] = np.where(df["MP"] > 0, df["Starts"] / df["MP"], 0)
    df["Role_Group"] = df["Pos"].apply(primary_role)
    df["Role_Label"] = df["Pos"].apply(position_bucket)
    df["Sample_Flag"] = np.where(df["90s"] < 3, "Small sample", "Stable sample")

    squad_min = squad[["Player", "Squad", "Playing Time_90s"]].rename(columns={"Playing Time_90s": "Squad_90s"})
    df = df.merge(squad_min, on=["Player", "Squad"], how="left")
    return df



def add_team_metrics(team: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    t = team.copy()
    t["xG_per_match"] = np.where(t["MP"] > 0, t["xG"] / t["MP"], np.nan)
    t["xGA_per_match"] = np.where(t["MP"] > 0, t["xGA"] / t["MP"], np.nan)
    t["Points_Rate"] = np.where(t["MP"] > 0, t["Pts"] / t["MP"], np.nan)
    t["Result_Index"] = np.where(t["MP"] > 0, (t["W"] + 0.5 * t["D"]) / t["MP"], np.nan)

    league_xg_pm = t["xG_per_match"].mean()
    t["SQR_Proxy"] = t["xG_per_match"] / league_xg_pm

    t["DCS_Proxy"] = np.where(t["xGA_per_match"] > 0, (t["Result_Index"] * 100) / t["xGA_per_match"], np.nan)

    outfield = player_df[player_df["Role_Group"] != "GK"].copy()
    top_ati = (
        outfield.groupby("Squad")["ATI"]
        .apply(lambda s: s.nlargest(min(3, len(s))).mean() if len(s) else np.nan)
        .reset_index(name="Team_ATI")
    )

    progress = (
        outfield.groupby("Squad")
        .agg(total_prgc=("PrgC", "sum"), total_prgp=("PrgP", "sum"), total_90s=("90s", "sum"))
        .reset_index()
    )
    progress["FTPE"] = np.where(progress["total_90s"] > 0, (progress["total_prgc"] + progress["total_prgp"]) / progress["total_90s"], 0)

    creators = (
        outfield.groupby("Squad")[["xAG_90", "Progressive_Receptions_90"]]
        .mean()
        .reset_index()
        .rename(columns={"xAG_90": "Avg_Creation_90", "Progressive_Receptions_90": "Avg_PrgR_90"})
    )

    t = t.merge(top_ati, on="Squad", how="left")
    t = t.merge(progress[["Squad", "FTPE"]], on="Squad", how="left")
    t = t.merge(creators, on="Squad", how="left")
    return t



def style_plot(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor=THEME["surface"],
        plot_bgcolor=THEME["surface"],
        font=dict(color=THEME["text"]),
        margin=dict(l=30, r=20, t=60, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["border"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=THEME["border"], zeroline=False)
    return fig



def metric_band(value: float, thresholds: Tuple[float, float, float], labels: Tuple[str, str, str, str]) -> Tuple[str, str]:
    a, b, c = thresholds
    if value >= c:
        return labels[3], "badge-elite"
    if value >= b:
        return labels[2], "badge-strong"
    if value >= a:
        return labels[1], "badge-moderate"
    return labels[0], "badge-low"



def metric_card(label: str, value: str, subtitle: str, badge_text: str, badge_class: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
            <div class="badge {badge_class}">{badge_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def simulate_poisson(lam_for: float, lam_against: float, n: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    gf = rng.poisson(max(lam_for, 0), size=n)
    ga = rng.poisson(max(lam_against, 0), size=n)
    return (gf > ga).mean(), (gf == ga).mean(), (gf < ga).mean()



def compute_fwpl(team_row: pd.Series, current_formation: str, new_formation: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    base_for = float(team_row["xG_per_match"])
    base_against = float(team_row["xGA_per_match"])
    att_c, def_c = FORMATION_MULTIPLIERS[current_formation]
    att_n, def_n = FORMATION_MULTIPLIERS[new_formation]
    curr = simulate_poisson(base_for * att_c, base_against * def_c)
    new = simulate_poisson(base_for * att_n, base_against * def_n)
    return curr, new, new[0] - curr[0]



def compute_sid(remaining_minutes: int, baseline_xg_pm: float, baseline_xga_pm: float, atk_out: float, atk_in: float, def_out: float, def_in: float) -> Tuple[float, float, float, float, float]:
    alpha = 0.40
    beta = 0.40
    delta_atk = atk_in - atk_out
    delta_def = def_in - def_out
    xg_created_base = baseline_xg_pm * remaining_minutes
    xg_conceded_base = baseline_xga_pm * remaining_minutes
    xg_created_new = xg_created_base * (1 + alpha * delta_atk)
    xg_conceded_new = xg_conceded_base * max(0.2, (1 - beta * delta_def))
    sid = (xg_created_new - xg_created_base) + (xg_conceded_base - xg_conceded_new)
    return sid, xg_created_base, xg_created_new, xg_conceded_base, xg_conceded_new



def coach_notes(team_row: pd.Series, team_players: pd.DataFrame) -> List[str]:
    notes: List[str] = []
    if team_players.empty:
        return ["No player rows match the active filter set."]

    attacker = team_players.sort_values("ATI", ascending=False).iloc[0]
    creator = team_players.sort_values("xAG_90", ascending=False).iloc[0]
    progressor = team_players.sort_values("Progression_90", ascending=False).iloc[0]

    notes.append(f"Primary attacking reference: {attacker['Player']} leads the filtered squad on ATI at {attacker['ATI']:.2f}.")
    notes.append(f"Best chance creator: {creator['Player']} leads on xAG/90 at {creator['xAG_90']:.2f}.")
    notes.append(f"Best ball progressor: {progressor['Player']} leads on progression/90 at {progressor['Progression_90']:.2f}.")

    low_start_high_impact = team_players[(team_players["Selection_Load"] < 0.50) & (team_players["90s"] >= 3)].sort_values("SID_attack_idx", ascending=False)
    if not low_start_high_impact.empty:
        impact = low_start_high_impact.iloc[0]
        notes.append(f"Impact-sub candidate: {impact['Player']} has a strong attack index ({impact['SID_attack_idx']:.2f}) with a lower starting load.")

    if float(team_row["SQR_Proxy"]) < 0.95:
        notes.append("Chance generation is below league average on the available proxy, so shot volume or chance access likely needs attention.")
    if float(team_row["DCS_Proxy"]) < 30:
        notes.append("Defensive control looks weak on the available proxy, so game-state protection should be treated cautiously.")
    return notes



def normalize_for_radar(team: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    radar = team.copy()
    for c in cols:
        cmin, cmax = radar[c].min(), radar[c].max()
        radar[f"{c}_norm"] = (radar[c] - cmin) / (cmax - cmin + 1e-9)
    return radar



def filtered_player_view(player_df: pd.DataFrame, selected_team: str, role_choices: List[str], min_90s: float, player_search: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = player_df.copy()
    df = df[df["90s"] >= min_90s]
    if role_choices:
        df = df[df["Role_Label"].isin(role_choices)]
    if player_search.strip():
        df = df[df["Player"].str.contains(player_search.strip(), case=False, na=False)]
    team_players = df[df["Squad"] == selected_team].copy()
    return df, team_players



def data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["team_stats", "xG", "Expected goals created across the season."],
            ["team_stats", "xGA", "Expected goals conceded across the season."],
            ["team_stats", "xGD/90", "Expected goal difference per 90 minutes."],
            ["fbref_pl", "xG", "Player expected goals across the season."],
            ["fbref_pl", "xAG", "Player expected assisted goals across the season."],
            ["fbref_pl", "PrgC", "Progressive carries."],
            ["fbref_pl", "PrgP", "Progressive passes."],
            ["fbref_pl", "PrgR", "Progressive receptions."],
            ["derived", "ATI", "0.6 × xG/90 + 0.4 × xAG/90."],
            ["derived", "SQR_Proxy", "xG per match relative to league average because shot-count fields are unavailable."],
            ["derived", "DCS_Proxy", "Results-index divided by xGA per match because defensive action fields are unavailable."],
            ["derived", "FTPE", "(PrgC + PrgP) per 90 at team level."],
            ["derived", "SID_attack_idx", "0.7 × xG/90 + 0.3 × xAG/90 for substitution scenarios."],
        ],
        columns=["Dataset", "Field", "Meaning"],
    )



def qa_checks(team: pd.DataFrame, fbref: pd.DataFrame, squad: pd.DataFrame) -> Dict[str, str]:
    checks = {}
    checks["Team rows"] = f"{len(team)} rows"
    checks["Player rows"] = f"{len(fbref)} rows"
    checks["Squad rows"] = f"{len(squad)} rows"
    checks["Missing team names"] = str(int(team["Squad"].isna().sum()))
    checks["Missing player names"] = str(int(fbref["Player"].isna().sum()))
    checks["Players under 3 x 90s"] = str(int((fbref["90s"] < 3).sum()))
    checks["Unique teams"] = str(team["Squad"].nunique())
    return checks



def main() -> None:
    st.title("Football Coaching Intelligence Dashboard")
    st.caption("Cleaner layout, clearer filters, honest KPI caveats, and more coach-usable outputs from the supplied Premier League datasets.")

    try:
        team_raw, fbref_raw, squad_raw = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    player_df = add_player_metrics(fbref_raw, squad_raw)
    team_df = add_team_metrics(team_raw, player_df)
    teams = sorted(team_df["Squad"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown("### Filters")
        c1, c2 = st.columns(2)
        with c1:
            selected_team = st.selectbox("Focus team", teams, help="Primary team for the dashboard pages.")
            role_choices = st.multiselect(
                "Player role filter",
                ["Forward", "Midfielder", "Defender", "Goalkeeper", "Hybrid"],
                default=["Forward", "Midfielder", "Defender"],
                help="This uses the position labels available in the player dataset.",
            )
        with c2:
            compare_default = teams.index(selected_team)
            compare_team = st.selectbox("Comparison team", teams, index=min(compare_default + 1, len(teams) - 1), help="Used on team comparison visuals.")
            min_90s = st.slider("Minimum 90s played", 0.0, float(player_df["90s"].max()), 3.0, 0.5, help="Raises data reliability by excluding tiny samples.")

        player_search = st.text_input("Player search", placeholder="Type a player name")
        show_small_sample_note = st.toggle("Show small-sample warning", value=True)

        st.markdown("---")
        st.markdown("### Filter logic")
        st.caption("1. Team controls the main narrative. 2. Role filter applies to player views. 3. Minimum 90s removes noisy player rows.")

    filtered_players, team_players = filtered_player_view(player_df, selected_team, role_choices, min_90s, player_search)
    selected_team_row = team_df[team_df["Squad"] == selected_team].iloc[0]
    compare_team_row = team_df[team_df["Squad"] == compare_team].iloc[0]

    if show_small_sample_note and int((filtered_players["90s"] < 3).sum()) > 0:
        st.markdown(
            '<div class="warning-box"><b>Reliability note:</b> low-minute players can distort per-90 metrics, so the minimum 90s filter matters a lot on this dashboard.</div>',
            unsafe_allow_html=True,
        )

    active_filter_text = f"Active view: {selected_team} | Compare: {compare_team} | Roles: {', '.join(role_choices) if role_choices else 'All'} | Minimum 90s: {min_90s:.1f}"
    st.markdown(f'<div class="info-box">{active_filter_text}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Teams", "Players", "Coaching Lab", "Data & Validation"])

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)

        ati_text, ati_cls = metric_band(float(selected_team_row["Team_ATI"]), (0.15, 0.35, 0.60), ("Low", "Moderate", "Strong", "Elite"))
        sqr_text, sqr_cls = metric_band(float(selected_team_row["SQR_Proxy"]), (0.80, 1.00, 1.15), ("Below avg", "Average", "Above avg", "Excellent"))
        dcs_text, dcs_cls = metric_band(float(selected_team_row["DCS_Proxy"]), (20, 40, 60), ("Weak", "Moderate", "Solid", "Excellent"))
        ftpe_text, ftpe_cls = metric_band(float(selected_team_row["FTPE"]), (20, 24, 28), ("Low", "Moderate", "Strong", "Elite"))

        with c1:
            metric_card("Attacking Threat Index", f"{selected_team_row['Team_ATI']:.3f}", "Top-3 outfield ATI average", ati_text, ati_cls)
        with c2:
            metric_card("SQR Proxy", f"{selected_team_row['SQR_Proxy']:.3f}", "xG per match vs league average", sqr_text, sqr_cls)
        with c3:
            metric_card("DCS Proxy", f"{selected_team_row['DCS_Proxy']:.1f}", "Results index relative to xGA/match", dcs_text, dcs_cls)
        with c4:
            metric_card("Final-Third Penetration", f"{selected_team_row['FTPE']:.2f}", "Progressive carries + passes per 90", ftpe_text, ftpe_cls)

        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            ranked = team_df.sort_values("Pts", ascending=True)
            fig = go.Figure(go.Bar(
                x=ranked["Pts"],
                y=ranked["Squad"],
                orientation="h",
                marker_color=[THEME["accent"] if s == selected_team else THEME["accent_2"] for s in ranked["Squad"]],
                text=ranked["Pts"],
                textposition="outside",
            ))
            style_plot(fig, f"League points ranking — highlight: {selected_team}", 520)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            notes = coach_notes(selected_team_row, team_players)
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Coach read-out")
            for note in notes:
                st.markdown(f"- {note}")
            st.markdown("</div>", unsafe_allow_html=True)

        team_profile_df = team_df.copy()
        team_profile_df["Highlight"] = team_profile_df["Squad"].apply(
            lambda squad: "Selected" if squad == selected_team else "Opponent" if squad == compare_team else "Other"
        )
        team_profile_df["Label"] = team_profile_df["Squad"].where(team_profile_df["Highlight"] != "Other", "")

        fig = px.scatter(
            team_profile_df,
            x="xGA",
            y="xG",
            size="Pts",
            color="Highlight",
            text="Label",
            color_discrete_map={
                "Selected": THEME["accent"],
                "Opponent": THEME["accent_2"],
                "Other": "#94a3b8",
            },
            hover_data={"Squad": True, "Pts": True, "SQR_Proxy": True, "xGD/90": True},
            labels={"xGA": "Season xGA", "xG": "Season xG", "SQR_Proxy": "SQR Proxy"},
        )
        diagonal_limit = max(team_df["xG"].max(), team_df["xGA"].max()) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=diagonal_limit, y1=diagonal_limit, line=dict(color=THEME["warning"], dash="dot"))
        fig.update_traces(textposition="top center", marker=dict(opacity=0.85))
        fig.update_layout(legend_title_text="Team")
        style_plot(fig, "Team attack vs defence profile", 430)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        radar_cols = ["Team_ATI", "SQR_Proxy", "DCS_Proxy", "FTPE", "xGD/90"]
        radar_labels = ["ATI", "SQR*", "DCS*", "FTPE", "xGD/90"]
        radar_df = normalize_for_radar(team_df, radar_cols)
        sel = radar_df[radar_df["Squad"] == selected_team].iloc[0]
        cmp = radar_df[radar_df["Squad"] == compare_team].iloc[0]

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=[sel[f"{c}_norm"] for c in radar_cols] + [sel[f"{radar_cols[0]}_norm"]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=selected_team,
            line=dict(color=THEME["accent"], width=2),
            fillcolor="rgba(20,184,166,0.12)",
        ))
        radar.add_trace(go.Scatterpolar(
            r=[cmp[f"{c}_norm"] for c in radar_cols] + [cmp[f"{radar_cols[0]}_norm"]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=compare_team,
            line=dict(color=THEME["accent_2"], width=2),
            fillcolor="rgba(96,165,250,0.10)",
        ))
        radar.update_layout(
            height=500,
            title="Normalized team comparison",
            paper_bgcolor=THEME["surface"],
            plot_bgcolor=THEME["surface"],
            font=dict(color=THEME["text"]),
            polar=dict(bgcolor=THEME["surface"], radialaxis=dict(range=[0, 1], gridcolor=THEME["border"]), angularaxis=dict(gridcolor=THEME["border"])),
        )
        st.plotly_chart(radar, use_container_width=True)

        team_table = team_df[["Rk", "Squad", "Pts", "xG", "xGA", "xGD/90", "Team_ATI", "SQR_Proxy", "DCS_Proxy", "FTPE", "Avg_Creation_90"]].sort_values("Pts", ascending=False).reset_index(drop=True)
        st.dataframe(
            team_table.style.format({
                "xG": "{:.1f}", "xGA": "{:.1f}", "xGD/90": "{:.2f}", "Team_ATI": "{:.3f}",
                "SQR_Proxy": "{:.3f}", "DCS_Proxy": "{:.1f}", "FTPE": "{:.2f}", "Avg_Creation_90": "{:.2f}",
            }),
            use_container_width=True,
            height=430,
            hide_index=True,
        )

        fig = px.scatter(
            team_df,
            x="xGD",
            y="Pts",
            size="FTPE",
            color="SQR_Proxy",
            text="Squad",
            color_continuous_scale="Tealgrn",
            labels={"xGD": "Season xGD", "Pts": "Points", "FTPE": "FTPE"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color=THEME["muted"])
        fig.update_traces(textposition="top center")
        style_plot(fig, "Points vs xGD, sized by final-third penetration", 430)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        if team_players.empty:
            st.warning("No players match the active filter set.")
        else:
            top_row_1, top_row_2 = st.columns(2)
            with top_row_1:
                team_bar = team_players.sort_values("ATI", ascending=True).tail(12)
                fig = go.Figure(go.Bar(
                    x=team_bar["ATI"],
                    y=team_bar["Player"],
                    orientation="h",
                    marker_color=THEME["accent"],
                    text=team_bar["ATI"].round(3),
                    textposition="outside",
                    customdata=np.stack([team_bar["Role_Label"], team_bar["90s"]], axis=-1),
                    hovertemplate="<b>%{y}</b><br>ATI: %{x:.3f}<br>Role: %{customdata[0]}<br>90s: %{customdata[1]:.1f}<extra></extra>",
                ))
                style_plot(fig, f"{selected_team} — top attacking contributors", 500)
                st.plotly_chart(fig, use_container_width=True)

            with top_row_2:
                fig = px.scatter(
                    team_players,
                    x="xG_90",
                    y="xAG_90",
                    size="90s",
                    color="Role_Label",
                    text="Player",
                    hover_data=["Pos", "ATI", "Progression_90"],
                    color_discrete_map={
                        "Forward": THEME["accent"],
                        "Midfielder": THEME["accent_2"],
                        "Defender": THEME["warning"],
                        "Goalkeeper": THEME["danger"],
                        "Hybrid": "#a78bfa",
                    },
                )
                fig.update_traces(textposition="top center")
                style_plot(fig, f"{selected_team} — scoring threat vs chance creation", 500)
                st.plotly_chart(fig, use_container_width=True)

            lower_row_1, lower_row_2 = st.columns(2)
            with lower_row_1:
                fig = px.scatter(
                    team_players,
                    x="Progression_90",
                    y="ATI",
                    size="90s",
                    color="Role_Label",
                    text="Player",
                    hover_data=["xG_90", "xAG_90", "Selection_Load"],
                    color_discrete_map={
                        "Forward": THEME["accent"],
                        "Midfielder": THEME["accent_2"],
                        "Defender": THEME["warning"],
                        "Goalkeeper": THEME["danger"],
                        "Hybrid": "#a78bfa",
                    },
                )
                fig.update_traces(textposition="top center")
                style_plot(fig, f"{selected_team} — progression vs attacking output", 430)
                st.plotly_chart(fig, use_container_width=True)

            with lower_row_2:
                by_role = team_players.groupby("Role_Label")[["xG_90", "xAG_90", "Progression_90"]].mean().reset_index()
                melted = by_role.melt(id_vars="Role_Label", var_name="Metric", value_name="Per90")
                fig = px.bar(
                    melted,
                    x="Role_Label",
                    y="Per90",
                    color="Metric",
                    barmode="group",
                    color_discrete_map={"xG_90": THEME["accent"], "xAG_90": THEME["accent_2"], "Progression_90": THEME["warning"]},
                )
                style_plot(fig, f"{selected_team} — average output by role group", 430)
                st.plotly_chart(fig, use_container_width=True)

            player_table = team_players[["Player", "Pos", "Role_Label", "90s", "Starts", "Gls", "Ast", "xG_90", "xAG_90", "Progression_90", "ATI", "Selection_Load", "Sample_Flag"]].sort_values(["ATI", "Progression_90"], ascending=False)
            st.dataframe(
                player_table.style.format({
                    "90s": "{:.1f}", "xG_90": "{:.2f}", "xAG_90": "{:.2f}", "Progression_90": "{:.2f}", "ATI": "{:.3f}", "Selection_Load": "{:.2f}",
                }),
                use_container_width=True,
                height=460,
            )

    with tabs[3]:
        sub_tabs = st.tabs(["Selection Signals", "Substitution Lab", "Formation Sandbox"])

        with sub_tabs[0]:
            if team_players.empty:
                st.warning("No players match the active filter set.")
            else:
                selection_df = team_players.copy()
                selection_df["Coach_Value"] = 0.50 * selection_df["ATI"] + 0.30 * selection_df["Progression_90"] / max(selection_df["Progression_90"].max(), 1e-9) + 0.20 * selection_df["G+A_90"]
                selection_df["Usage_Tag"] = np.select(
                    [selection_df["Selection_Load"] >= 0.70, selection_df["Selection_Load"] >= 0.40],
                    ["Likely starter", "Rotation option"],
                    default="Impact sub / squad depth",
                )
                shortlist = selection_df[["Player", "Role_Label", "90s", "Starts", "ATI", "xAG_90", "Progression_90", "Coach_Value", "Usage_Tag"]].sort_values("Coach_Value", ascending=False)
                st.dataframe(
                    shortlist.style.format({"90s": "{:.1f}", "ATI": "{:.3f}", "xAG_90": "{:.2f}", "Progression_90": "{:.2f}", "Coach_Value": "{:.3f}"}),
                    use_container_width=True,
                    height=430,
                )
                st.markdown('<div class="info-box"><b>How it helps:</b> this view turns raw player stats into a practical shortlist for starters, rotation options, and likely impact substitutes.</div>', unsafe_allow_html=True)

        with sub_tabs[1]:
            if len(team_players) < 2:
                st.warning("At least two filtered players are required for substitution scenarios.")
            else:
                ordered = team_players.sort_values("ATI", ascending=False)["Player"].tolist()
                c1, c2, c3 = st.columns(3)
                with c1:
                    out_player = st.selectbox("Outgoing player", ordered, index=min(1, len(ordered) - 1))
                with c2:
                    in_player = st.selectbox("Incoming player", ordered, index=0)
                with c3:
                    remaining_minutes = st.slider("Remaining minutes", 5, 60, 30)

                out_row = team_players[team_players["Player"] == out_player].iloc[0]
                in_row = team_players[team_players["Player"] == in_player].iloc[0]

                c1, c2 = st.columns(2)
                with c1:
                    def_out = st.slider("Outgoing defensive scenario score", 0.0, 1.5, 0.50, 0.05)
                with c2:
                    def_in = st.slider("Incoming defensive scenario score", 0.0, 1.5, 0.75, 0.05)

                baseline_xg_pm = float(selected_team_row["xG"]) / (float(selected_team_row["MP"]) * 90)
                baseline_xga_pm = float(selected_team_row["xGA"]) / (float(selected_team_row["MP"]) * 90)
                sid, base_create, new_create, base_concede, new_concede = compute_sid(
                    remaining_minutes,
                    baseline_xg_pm,
                    baseline_xga_pm,
                    float(out_row["SID_attack_idx"]),
                    float(in_row["SID_attack_idx"]),
                    def_out,
                    def_in,
                )
                sid_text, sid_cls = metric_band(sid, (-0.05, 0.05, 0.20), ("Negative", "Neutral", "Mild gain", "Strong gain"))

                m1, m2, m3 = st.columns(3)
                with m1:
                    metric_card("SID", f"{sid:+.3f}", "Net xG balance impact over remaining time", sid_text, sid_cls)
                with m2:
                    metric_card("Baseline xG created", f"{base_create:.2f}", f"Next {remaining_minutes} minutes", "Baseline", "badge-info")
                with m3:
                    metric_card("Scenario xG created", f"{new_create:.2f}", "After the substitution", "Scenario", "badge-strong")

                sid_df = pd.DataFrame({
                    "Metric": ["Attack index out", "Attack index in", "Defensive score out", "Defensive score in", "Baseline xG created", "Scenario xG created", "Baseline xG conceded", "Scenario xG conceded"],
                    "Value": [out_row["SID_attack_idx"], in_row["SID_attack_idx"], def_out, def_in, base_create, new_create, base_concede, new_concede],
                })
                st.dataframe(sid_df.style.format({"Value": "{:.3f}"}), use_container_width=True)
                st.markdown('<div class="warning-box"><b>Important:</b> the attacking side consists real player data, but the defensive side is a coach-entered scenario because the available data to us does not include defensive metrics.</div>', unsafe_allow_html=True)

        with sub_tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                current_formation = st.selectbox("Current formation", list(FORMATION_MULTIPLIERS.keys()), index=0)
            with c2:
                new_formation = st.selectbox("Proposed formation", list(FORMATION_MULTIPLIERS.keys()), index=2)

            current_probs, new_probs, fwpl = compute_fwpl(selected_team_row, current_formation, new_formation)
            scenario = pd.DataFrame({
                "Outcome": ["Win", "Draw", "Loss"],
                current_formation: current_probs,
                new_formation: new_probs,
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(x=scenario["Outcome"], y=scenario[current_formation], name=current_formation, marker_color=THEME["accent_2"]))
            fig.add_trace(go.Bar(x=scenario["Outcome"], y=scenario[new_formation], name=new_formation, marker_color=THEME["accent"]))
            fig.update_layout(barmode="group")
            style_plot(fig, f"Formation scenario probabilities — {selected_team}", 420)
            st.plotly_chart(fig, use_container_width=True)

            fwpl_text, fwpl_cls = metric_band(fwpl, (-0.02, 0.02, 0.05), ("Negative", "Neutral", "Marginal gain", "Meaningful gain"))
            metric_card("FWPL", f"{fwpl:+.3f}", "Change in simulated win probability", fwpl_text, fwpl_cls)
            st.markdown('<div class="warning-box"><b>Important:</b> this is a scenario analysis, not a predictive match model. The uplift depends on fixed formation multipliers because the ingested datasets and  does not include formation-by-match histories.</div>', unsafe_allow_html=True)

    with tabs[4]:
        st.subheader("Data dictionary")
        st.dataframe(data_dictionary(), use_container_width=True, height=420)

        st.subheader("Validation checks")
        qa = pd.DataFrame(list(qa_checks(team_raw, fbref_raw, squad_raw).items()), columns=["Check", "Result"])
        st.dataframe(qa, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
