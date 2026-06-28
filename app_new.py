
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Football Coaching Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
THEME = {
    "bg": "#0f172a", "surface": "#111827", "surface_2": "#1f2937",
    "border": "#334155", "text": "#e5e7eb", "muted": "#94a3b8",
    "accent": "#14b8a6", "accent_2": "#60a5fa",
    "warning": "#f59e0b", "danger": "#ef4444", "success": "#22c55e",
}

st.markdown(f"""
<style>
:root {{
    --bg:{THEME['bg']};--surface:{THEME['surface']};--surface-2:{THEME['surface_2']};
    --border:{THEME['border']};--text:{THEME['text']};--muted:{THEME['muted']};
    --accent:{THEME['accent']};--accent2:{THEME['accent_2']};
    --warn:{THEME['warning']};--danger:{THEME['danger']};
}}
html,body,[class*="css"]{{color:var(--text);font-family:Inter,Segoe UI,sans-serif;}}
[data-testid="stAppViewContainer"]{{background:linear-gradient(180deg,#0b1220,#0f172a);overflow-x:hidden;}}
[data-testid="stHeader"]{{background:rgba(15,23,42,0.80);}}
[data-testid="stSidebar"]>div:first-child{{background:var(--surface);border-right:1px solid var(--border);min-width:250px;max-width:280px;padding:0.75rem 0.85rem 0.85rem;}}
[data-testid="stToolbar"],[data-testid="stDecoration"]{{visibility:hidden;display:none;}}
.block-container{{padding-top:1rem;padding-bottom:1.2rem;max-width:1180px;}}
h1,h2,h3{{color:var(--text)!important;letter-spacing:-0.02em;}}

.metric-card{{
    background:linear-gradient(180deg,rgba(17,24,39,1),rgba(16,24,39,0.88));
    border:1px solid var(--border);border-radius:18px;
    padding:1rem 1rem 0.9rem 1rem;min-height:148px;
    position:relative;overflow:hidden;
}}
.metric-card::before{{
    content:"";position:absolute;left:0;top:0;right:0;height:3px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));
}}
.metric-label{{font-size:0.76rem;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;}}
.metric-value{{font-size:2rem;font-weight:750;line-height:1.0;color:var(--accent);margin-top:.35rem;}}
.metric-sub{{color:var(--muted);font-size:.83rem;margin-top:.3rem;}}
.badge{{
    display:inline-block;margin-top:.65rem;padding:.22rem .65rem;border-radius:999px;
    font-size:.71rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
}}
.badge-elite{{color:#2dd4bf;background:rgba(45,212,191,.10);border:1px solid rgba(45,212,191,.34);}}
.badge-strong{{color:#4ade80;background:rgba(74,222,128,.10);border:1px solid rgba(74,222,128,.34);}}
.badge-moderate{{color:#fbbf24;background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.34);}}
.badge-low{{color:#f87171;background:rgba(248,113,113,.10);border:1px solid rgba(248,113,113,.34);}}
.badge-info{{color:#93c5fd;background:rgba(147,197,253,.10);border:1px solid rgba(147,197,253,.34);}}
.info-box{{background:rgba(20,184,166,.08);border:1px solid rgba(20,184,166,.25);
           border-radius:16px;padding:.95rem 1rem;color:#d1fae5;font-size:.9rem;}}
.warn-box{{background:rgba(245,158,11,.10);border:1px solid rgba(245,158,11,.25);
           border-radius:16px;padding:.95rem 1rem;color:#fde68a;font-size:.9rem;}}
.miss-box{{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);
           border-radius:16px;padding:.95rem 1rem;color:#fca5a5;font-size:.9rem;}}
[data-testid="stTabs"] button{{color:var(--muted)!important;border-radius:12px 12px 0 0;}}
[data-testid="stTabs"] button[aria-selected="true"]{{color:var(--text)!important;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
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

TEAM_REQUIRED = [
    "Rk","Squad","MP","W","D","L","GF","GA","GD","Pts","Pts/MP",
    "xG","xGA","xGD","xGD/90","Attendance","Top Team Scorer","Goalkeeper","Notes",
]
FBREF_REQUIRED = [
    "Rk","Player","Nation","Pos","Squad","Age","Born","MP","Starts","Min","90s",
    "Gls","Ast","G+A","G-PK","PK","PKatt","CrdY","CrdR",
    "xG","npxG","xAG","npxG+xAG","PrgC","PrgP","PrgR",
]
SQUAD_REQUIRED = [
    "Rk","Player","Nation","Pos","Squad","Age","Born",
    "Playing Time_MP","Playing Time_Starts","Playing Time_Min","Playing Time_90s",
    "Performance_Gls","Performance_Ast","Performance_G+A","Performance_G-PK",
    "Performance_PK","Performance_PKatt","Performance_CrdY","Performance_CrdR",
    "Per 90 Minutes_Gls","Per 90 Minutes_Ast","Per 90 Minutes_G+A",
    "Per 90 Minutes_G-PK","Per 90 Minutes_G+A-PK","Matches",
]
# Supplementary defensive file columns needed for true SQR & DCS
DEF_COLS_NEEDED = ["Player", "Squad", "Sh", "TklW", "Int", "Clr", "90s"]

NUMERIC_TEAM  = ["MP","W","D","L","GF","GA","GD","Pts","Pts/MP","xG","xGA","xGD","xGD/90","Attendance"]
NUMERIC_FBREF = ["Age","Born","MP","Starts","Min","90s","Gls","Ast","G+A","G-PK","PK","PKatt",
                 "CrdY","CrdR","xG","npxG","xAG","npxG+xAG","PrgC","PrgP","PrgR"]
NUMERIC_SQUAD = ["Age","Born","Playing Time_MP","Playing Time_Starts","Playing Time_Min",
                 "Playing Time_90s","Performance_Gls","Performance_Ast","Performance_G+A",
                 "Performance_G-PK","Performance_PK","Performance_PKatt",
                 "Performance_CrdY","Performance_CrdR","Per 90 Minutes_Gls",
                 "Per 90 Minutes_Ast","Per 90 Minutes_G+A",
                 "Per 90 Minutes_G-PK","Per 90 Minutes_G+A-PK"]
NUMERIC_DEF   = ["Sh","TklW","Int","Clr","90s"]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def locate_csv(names: List[str]) -> Optional[str]:
    """
    Search a short list of sensible folders for any of the requested filenames.
    Returns the first matching path or None. Matching is forgiving: exact
    filename first, then files that contain the base name and end with `.csv`.
    """
    cwd = os.getcwd()
    script_dir = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        pass

    home = os.path.expanduser('~')

    candidates = [
        cwd,
        os.path.join(cwd, 'data'),
        script_dir,
        os.path.join(script_dir, 'data') if script_dir else None,
        home,
        os.path.join(home, 'Desktop'),
    ]

    # add a few parent folders of cwd/script_dir to catch project layouts
    for start in (cwd, script_dir):
        cur = start
        for _ in range(3):
            if not cur:
                break
            cur = os.path.dirname(cur)
            candidates.append(cur)
            candidates.append(os.path.join(cur, 'data'))

    seen = set()
    for base in candidates:
        if not base or base in seen:
            continue
        seen.add(base)
        try:
            files = os.listdir(base)
        except Exception:
            files = []

        for name in names:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p

        # fuzzy: match files that contain the name base and end with .csv
        for f in files:
            fl = f.lower()
            for name in names:
                base_name = os.path.splitext(name)[0].lower()
                if fl == name.lower() or (base_name in fl and fl.endswith('.csv')):
                    return os.path.join(base, f)

    return None

def validate_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_defensive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to make defensive files forgiving to minor column-name differences.
    Renames common variants like '90'->'90s', 'Tkl'->'TklW', 'Shots'->'Sh'.
    """
    cols = {c: c for c in df.columns}
    lower = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    # map common variations
    mapping = {}
    if pick('90s') is None and pick('90') is not None:
        mapping[pick('90')] = '90s'
    if pick('TklW') is None and pick('Tkl') is not None:
        mapping[pick('Tkl')] = 'TklW'
    if pick('Int') is None and pick('Ints') is not None:
        mapping[pick('Ints')] = 'Int'
    if pick('Clr') is None and pick('Clearances') is not None:
        mapping[pick('Clearances')] = 'Clr'
    if pick('Sh') is None and pick('Shots') is not None:
        mapping[pick('Shots')] = 'Sh'
    if pick('Player') is None and pick('Name') is not None:
        mapping[pick('Name')] = 'Player'
    if pick('Squad') is None and pick('Team') is not None:
        mapping[pick('Team')] = 'Squad'

    if mapping:
        df = df.rename(columns=mapping)
    return df

def load_data():
    team_path  = locate_csv(["team_stats.csv"])
    fbref_path = locate_csv(["fbref_pl.csv", "fbref_PL.csv"])
    squad_path = locate_csv(["squad_stats.csv"])
    def_path   = locate_csv(["defensive_stats.csv",
                              "players_data_light-2024_2025.csv",
                              "players_data-2024_2025.csv"])

    if not team_path or not fbref_path or not squad_path:
        raise FileNotFoundError(
            "Could not find team_stats.csv, fbref_pl.csv and squad_stats.csv. "
            "Place them in ./data/ or the working directory."
        )

    team  = pd.read_csv(team_path)
    fbref = pd.read_csv(fbref_path)
    squad = pd.read_csv(squad_path)

    team.columns  = [c.strip() for c in team.columns]
    fbref.columns = [c.strip() for c in fbref.columns]
    squad.columns = [c.strip() for c in squad.columns]

    validate_columns(team,  TEAM_REQUIRED,  "team_stats")
    validate_columns(fbref, FBREF_REQUIRED, "fbref_pl")
    validate_columns(squad, SQUAD_REQUIRED, "squad_stats")

    for c in NUMERIC_TEAM:  team[c]  = pd.to_numeric(team[c],  errors="coerce")
    for c in NUMERIC_FBREF: fbref[c] = pd.to_numeric(fbref[c], errors="coerce")
    for c in NUMERIC_SQUAD: squad[c] = pd.to_numeric(squad[c], errors="coerce")

    # Optional defensive supplementary file
    def_df = None
    if def_path:
        try:
            def_df = pd.read_csv(def_path)
            def_df.columns = [c.strip() for c in def_df.columns]
            # try to be forgiving about column names
            def_df = _normalize_defensive_columns(def_df)
            # Filter to PL only if Comp column exists
            if "Comp" in def_df.columns:
                def_df = def_df[def_df["Comp"].astype(str).str.contains("Premier", case=False, na=False)]
            missing_def = [c for c in DEF_COLS_NEEDED if c not in def_df.columns]
            if missing_def:
                # If the file lacks strict columns, drop and treat as not available
                def_df = None
            else:
                for c in NUMERIC_DEF:
                    def_df[c] = pd.to_numeric(def_df[c], errors="coerce")
        except Exception:
            def_df = None

    return team, fbref, squad, def_df, def_path

# ─────────────────────────────────────────────
# KPI CALCULATIONS
# ─────────────────────────────────────────────
def add_player_metrics(fbref: pd.DataFrame) -> pd.DataFrame:
    df = fbref.copy()
    df["xG_90"]          = np.where(df["90s"] > 0, df["xG"]  / df["90s"], 0)
    df["xAG_90"]         = np.where(df["90s"] > 0, df["xAG"] / df["90s"], 0)
    df["G+A_90"]         = np.where(df["90s"] > 0, df["G+A"] / df["90s"], 0)
    df["Progression_90"] = np.where(df["90s"] > 0, (df["PrgC"] + df["PrgP"]) / df["90s"], 0)
    df["PrgR_90"]        = np.where(df["90s"] > 0, df["PrgR"] / df["90s"], 0)
    # ── KPI 1: ATI — FULLY DATA-DRIVEN ──
    df["ATI"]            = 0.6 * df["xG_90"] + 0.4 * df["xAG_90"]
    # Attacking index used in SID
    df["SID_atk_idx"]    = 0.7 * df["xG_90"] + 0.3 * df["xAG_90"]
    df["Selection_Load"] = np.where(df["MP"] > 0, df["Starts"] / df["MP"], 0)
    df["Role_Label"]     = df["Pos"].apply(_role_label)
    df["Role_Group"]     = df["Pos"].apply(_role_group)
    df["Sample_Flag"]    = np.where(df["90s"] < 3, "Small sample (<3×90s)", "Stable")
    return df


def add_defensive_player_metrics(player_df: pd.DataFrame,
                                  def_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges true defensive action counts from the supplementary file.
    TklW + Int + Clr per 90 gives the defensive actions per 90 for SID defensive index.
    """
    # If there's no defensive file, create the expected columns with NaN
    if def_df is None:
        for c in ["Sh", "TklW", "Int", "Clr", "Def_Actions_90", "SID_def_idx"]:
            player_df[c] = np.nan
        return player_df

    # Pick relevant columns if they exist
    take = [c for c in ["Player", "Squad", "Sh", "TklW", "Int", "Clr", "90s"] if c in def_df.columns]
    def_agg = def_df[take].copy()
    # Ensure numeric where expected
    for c in ["Sh", "TklW", "Int", "Clr", "90s"]:
        if c in def_agg.columns:
            def_agg[c] = pd.to_numeric(def_agg[c], errors="coerce")

    # Compute per-90 defensive actions if we have the ingredients
    if all(c in def_agg.columns for c in ["TklW", "Int", "Clr", "90s"]):
        def_agg["Def_Actions_90"] = np.where(
            def_agg["90s"] > 0,
            (def_agg["TklW"] + def_agg["Int"] + def_agg["Clr"]) / def_agg["90s"],
            0,
        )
    else:
        def_agg["Def_Actions_90"] = np.nan
    # Normalize player and squad names for a more robust join
    player_df = player_df.copy()
    def_agg = def_agg.copy()
    if "Player" in player_df.columns:
        player_df["Player_key"] = player_df["Player"].astype(str).str.strip().str.lower()
    else:
        player_df["Player_key"] = ""
    if "Squad" in player_df.columns:
        player_df["Squad_key"] = player_df["Squad"].astype(str).str.strip().str.lower()
    else:
        player_df["Squad_key"] = ""

    if "Player" in def_agg.columns:
        def_agg["Player_key"] = def_agg["Player"].astype(str).str.strip().str.lower()
    else:
        def_agg["Player_key"] = ""
    if "Squad" in def_agg.columns:
        def_agg["Squad_key"] = def_agg["Squad"].astype(str).str.strip().str.lower()
    else:
        def_agg["Squad_key"] = ""

    merge_on = ["Player_key", "Squad_key"]
    merged = player_df.merge(def_agg.drop(columns=[c for c in ["90s"] if c in def_agg.columns], errors='ignore'),
                              left_on=merge_on, right_on=merge_on, how="left", suffixes=("","_def"))
    # SID defensive index is simply the Def_Actions_90 for now
    merged["SID_def_idx"] = merged.get("Def_Actions_90")
    # Clean up helper keys
    for k in ["Player_key","Squad_key"]:
        if k in merged.columns:
            merged.drop(columns=[k], inplace=True)
    return merged


def add_team_metrics(team: pd.DataFrame,
                     player_df: pd.DataFrame,
                     def_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    t = team.copy()
    t["xG_per_match"]  = np.where(t["MP"] > 0, t["xG"]  / t["MP"], np.nan)
    t["xGA_per_match"] = np.where(t["MP"] > 0, t["xGA"] / t["MP"], np.nan)
    # League averages stored on every row so opponent-aware FWPL can reference them
    t["league_avg_xG_pm"]  = t["xG_per_match"].mean()
    t["league_avg_xGA_pm"] = t["xGA_per_match"].mean()

    # ── KPI 2: SQR — true formula needs shots per team ──
    if def_df is not None and "Sh" in def_df.columns:
        # Sum shots per team from player-level data
        team_shots = def_df.groupby("Squad")["Sh"].sum().reset_index(name="Team_Sh")
        t = t.merge(team_shots, on="Squad", how="left")
        t["xG_per_shot"]        = np.where(t["Team_Sh"] > 0, t["xG"] / t["Team_Sh"], np.nan)
        league_xg_per_shot      = t["xG_per_shot"].mean()
        t["SQR"]                = t["xG_per_shot"] / league_xg_per_shot
        t["SQR_available"]      = True
    else:
        t["Team_Sh"]            = np.nan
        t["xG_per_shot"]        = np.nan
        t["SQR"]                = np.nan
        t["SQR_available"]      = False

    # ── KPI 3: DCS — true formula needs TklW + Int + Clr per team ──
    if def_df is not None and all(c in def_df.columns for c in ["TklW", "Int", "Clr"]):
        team_def = (
            def_df.groupby("Squad")[["TklW", "Int", "Clr"]]
            .sum()
            .reset_index()
        )
        team_def["Total_Def_Actions"] = team_def["TklW"] + team_def["Int"] + team_def["Clr"]
        t = t.merge(team_def[["Squad", "Total_Def_Actions"]], on="Squad", how="left")
        t["DCS"]           = np.where(t["xGA"] > 0, t["Total_Def_Actions"] / t["xGA"], np.nan)
        t["DCS_available"] = True
    else:
        t["Total_Def_Actions"] = np.nan
        t["DCS"]               = np.nan
        t["DCS_available"]     = False

    # ── Team ATI from top-3 outfield players — FULLY DATA-DRIVEN ──
    outfield = player_df[player_df["Role_Group"] != "GK"].copy()
    top_ati  = (
        outfield.groupby("Squad")["ATI"]
        .apply(lambda s: s.nlargest(min(3, len(s))).mean() if len(s) else np.nan)
        .reset_index(name="Team_ATI")
    )
    t = t.merge(top_ati, on="Squad", how="left")

    # ── FTPE — FULLY DATA-DRIVEN ──
    prog = (
        outfield.groupby("Squad")
        .agg(total_prgc=("PrgC","sum"), total_prgp=("PrgP","sum"), total_90s=("90s","sum"))
        .reset_index()
    )
    prog["FTPE"] = np.where(prog["total_90s"] > 0,
                            (prog["total_prgc"] + prog["total_prgp"]) / prog["total_90s"], 0)
    t = t.merge(prog[["Squad","FTPE"]], on="Squad", how="left")

    return t


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def _role_label(pos: str) -> str:
    pos = str(pos)
    if "GK" in pos: return "Goalkeeper"
    if pos.startswith("DF"): return "Defender"
    if pos.startswith("MF"): return "Midfielder"
    if pos.startswith("FW"): return "Forward"
    return "Hybrid"

def _role_group(pos: str) -> str:
    pos = str(pos)
    if "GK" in pos: return "GK"
    if "DF" in pos: return "DF"
    if "MF" in pos: return "MF"
    if "FW" in pos: return "FW"
    return "Hybrid"

def style_plot(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title, height=height,
        paper_bgcolor=THEME["surface"], plot_bgcolor=THEME["surface"],
        font=dict(color=THEME["text"]),
        margin=dict(l=30, r=20, t=55, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["border"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=THEME["border"], zeroline=False)
    return fig

def metric_card(label: str, value: str, subtitle: str,
                badge_text: str, badge_class: str) -> None:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
        <div class="badge {badge_class}">{badge_text}</div>
    </div>""", unsafe_allow_html=True)

def band(v: float, lo: float, mid: float, hi: float,
         labels=("Weak","Moderate","Strong","Elite")) -> Tuple[str,str]:
    classes = ("badge-low","badge-moderate","badge-strong","badge-elite")
    if v >= hi:  return labels[3], classes[3]
    if v >= mid: return labels[2], classes[2]
    if v >= lo:  return labels[1], classes[1]
    return labels[0], classes[0]

def ati_band(v):  return band(v, 0.15, 0.35, 0.60,
    ("Low involvement","Moderate","Strong contributor","Elite"))
def sqr_band(v):  return band(v, 0.80, 1.00, 1.30,
    ("Poor chance quality","Average","Above average","Excellent"))
def dcs_band(v):  return band(v, 20,   40,   60,
    ("Poor","Moderate","Solid","Excellent"))
def ftpe_band(v): return band(v, 20,   24,   28,
    ("Low","Moderate","Strong","Elite"))
def sid_band(v):  return band(v, -0.05, 0.05, 0.20,
    ("Negative impact","Neutral","Mild improvement","Strong positive"))
def fwpl_band(v): return band(v, -0.02, 0.02, 0.05,
    ("Negative","Negligible","Marginal gain","Meaningful uplift"))

def normalize_radar(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    r = df.copy()
    for c in cols:
        mn, mx = r[c].min(), r[c].max()
        r[f"{c}_n"] = (r[c] - mn) / (mx - mn + 1e-9)
    return r

# ─────────────────────────────────────────────
# SIMULATION LOGIC
# ─────────────────────────────────────────────
def simulate_poisson(lam_for: float, lam_against: float,
                     n: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    gf  = rng.poisson(max(lam_for, 0),     size=n)
    ga  = rng.poisson(max(lam_against, 0), size=n)
    return (gf > ga).mean(), (gf == ga).mean(), (gf < ga).mean()

def compute_fwpl(team_row, opp_row, cur_f: str, new_f: str):
    """
    Opponent-aware FWPL.

    team_lam  = team_xG_pm  * att_factor * opp_def_str
    opp_lam   = team_xGA_pm * opp_att_str / def_factor

    opp_att_str = opp_xG_pm  / league_avg_xG_pm   (how threatening they are)
    opp_def_str = opp_xGA_pm / league_avg_xGA_pm   (how leaky they are — bigger = easier to score against)
    """
    team_xg_pm  = float(team_row["xG_per_match"])
    team_xga_pm = float(team_row["xGA_per_match"])
    opp_xg_pm   = float(opp_row["xG_per_match"])
    opp_xga_pm  = float(opp_row["xGA_per_match"])

    # league context stored on team_row (added in add_team_metrics)
    lg_avg = float(team_row.get("league_avg_xG_pm", team_xg_pm))  # fallback to self if missing

    opp_att_str = opp_xg_pm  / max(lg_avg, 0.01)
    opp_def_str = opp_xga_pm / max(lg_avg, 0.01)

    ac, dc = FORMATION_MULTIPLIERS[cur_f]
    an, dn = FORMATION_MULTIPLIERS[new_f]

    lam_for_cur  = team_xg_pm  * ac * opp_def_str
    lam_opp_cur  = team_xga_pm * opp_att_str / max(dc, 0.01)
    lam_for_new  = team_xg_pm  * an * opp_def_str
    lam_opp_new  = team_xga_pm * opp_att_str / max(dn, 0.01)

    curr = simulate_poisson(lam_for_cur, lam_opp_cur)
    new_ = simulate_poisson(lam_for_new, lam_opp_new)

    return {
        "curr": curr, "new_": new_,
        "fwpl": new_[0] - curr[0],
        "lam_for_cur": lam_for_cur, "lam_opp_cur": lam_opp_cur,
        "lam_for_new": lam_for_new, "lam_opp_new": lam_opp_new,
        "opp_att_str": round(opp_att_str, 3), "opp_def_str": round(opp_def_str, 3),
    }

def compute_sid(rem_min: int, xg_pm: float, xga_pm: float,
                atk_out: float, atk_in: float,
                def_out: float, def_in: float):
    alpha, beta = 0.40, 0.40
    da, dd = atk_in - atk_out, def_in - def_out
    xg_b  = xg_pm  * rem_min
    xga_b = xga_pm * rem_min
    xg_n  = xg_b  * (1 + alpha * da)
    xga_n = xga_b * max(0.2, (1 - beta * dd))
    sid   = (xg_n - xg_b) + (xga_b - xga_n)
    return sid, xg_b, xg_n, xga_b, xga_n

def coach_notes(team_row: pd.Series, tp: pd.DataFrame) -> List[str]:
    if tp.empty:
        return ["No player rows match the active filter set."]
    notes = []
    lead_ati  = tp.nlargest(1, "ATI").iloc[0]
    lead_creat = tp.nlargest(1, "xAG_90").iloc[0]
    lead_prog = tp.nlargest(1, "Progression_90").iloc[0]
    notes.append(f"Top ATI: **{lead_ati['Player']}** — {lead_ati['ATI']:.3f}")
    notes.append(f"Top creator: **{lead_creat['Player']}** — xAG/90 {lead_creat['xAG_90']:.3f}")
    notes.append(f"Top progressor: **{lead_prog['Player']}** — {lead_prog['Progression_90']:.2f} prog/90")
    sub_cands = tp[(tp["Selection_Load"] < 0.50) & (tp["90s"] >= 3)].nlargest(1, "SID_atk_idx")
    if not sub_cands.empty:
        s = sub_cands.iloc[0]
        notes.append(f"Impact-sub candidate: **{s['Player']}** — atk index {s['SID_atk_idx']:.3f}, low start load")
    return notes

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.title("Football Coaching Intelligence Dashboard")
    st.caption("Premier League 2024/25 — All KPIs use only real data fields, no proxies.")

    try:
        team_raw, fbref_raw, squad_raw, def_df, def_path = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    player_df = add_player_metrics(fbref_raw)
    player_df = add_defensive_player_metrics(player_df, def_df)
    team_df   = add_team_metrics(team_raw, player_df, def_df)

    sqr_live = bool(team_df["SQR_available"].iloc[0])
    dcs_live = bool(team_df["DCS_available"].iloc[0])

    teams = sorted(team_df["Squad"].dropna().unique())

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### Dashboard Filters")
        selected_team = st.selectbox("Focus team", teams,
            help="All player views and KPI cards use this team.")
        compare_idx   = min(teams.index(selected_team) + 1, len(teams) - 1)
        compare_team  = st.selectbox("Comparison team", teams, index=compare_idx,
            help="Used in the team radar comparison.")
        role_choices  = st.multiselect("Player role filter",
            ["Forward","Midfielder","Defender","Goalkeeper"],
            default=["Forward","Midfielder","Defender"])
        min_90s       = st.slider("Minimum 90s played", 0.0,
            float(player_df["90s"].max()), 3.0, 0.5,
            help="Removes small-sample outliers from per-90 metrics.")
        player_search = st.text_input("Player search", placeholder="Name substring")

        st.markdown("---")
        st.markdown("### Data status")
        st.markdown(f"**SQR** — {'live (shots available)' if sqr_live else 'N/A (add defensive_stats.csv)'}")
        st.markdown(f"**DCS** — {'live (def actions available)' if dcs_live else 'N/A (add defensive_stats.csv)'}")
        # Defensive file presence is used internally; keep status brief
        # (users should place defensive CSVs in the project's `data/` folder)
        st.markdown("**ATI** — live (xG, xAG, 90s)")
        st.markdown("**FTPE** — live (PrgC, PrgP, 90s)")
        st.markdown("**FWPL** — live (xG, xGA, Poisson sim)")
        st.markdown(f"**SID def** — {'live (TklW, Int, Clr)' if dcs_live else 'manual slider (no def data)'}")

        with st.expander("How each KPI is calculated", expanded=False):
            st.markdown("""
- **xG_90** = xG ÷ 90s
- **xAG_90** = xAG ÷ 90s
- **G+A_90** = G+A ÷ 90s
- **Progression_90** = (PrgC + PrgP) ÷ 90s
- **PrgR_90** = PrgR ÷ 90s
- **ATI** = 0.6 × xG_90 + 0.4 × xAG_90
- **SID attacking index** = 0.7 × xG_90 + 0.3 × xAG_90
- **Selection_Load** = Starts ÷ MP
- **Def_Actions_90** = (TklW + Int + Clr) ÷ 90s
- **SID_def_idx** = Def_Actions_90
- **xG_per_match** = xG ÷ MP
- **xGA_per_match** = xGA ÷ MP
- **xG_per_shot** = xG ÷ Team_Sh
- **SQR** = xG_per_shot ÷ league average xG_per_shot
- **Total_Def_Actions** = TklW + Int + Clr
- **DCS** = Total_Def_Actions ÷ xGA
- **Team_ATI** = average of top 3 outfield player ATI values per team
- **FTPE** = (sum PrgC + sum PrgP) ÷ sum 90s per team
- **FWPL** = Poisson simulation win/draw/loss probability shift from xG/xGA and formation multipliers
""")



    # ── FILTER PLAYERS ──
    fp = player_df[player_df["90s"] >= min_90s].copy()
    if role_choices:
        fp = fp[fp["Role_Label"].isin(role_choices)]
    if player_search.strip():
        fp = fp[fp["Player"].str.contains(player_search.strip(), case=False, na=False)]
    tp = fp[fp["Squad"] == selected_team].copy()

    sel_row = team_df[team_df["Squad"] == selected_team].iloc[0]
    cmp_row = team_df[team_df["Squad"] == compare_team].iloc[0]

    st.markdown(
        f'<div class="info-box">Viewing: <b>{selected_team}</b> | '
        f'Compare: <b>{compare_team}</b> | '
        f'Roles: {", ".join(role_choices) if role_choices else "All"} | '
        f'Min 90s: {min_90s:.1f}</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Overview", "Teams", "Players", "Coaching Lab", "About Data"])

    # ══════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════
    with tabs[0]:
        ati_v  = float(sel_row["Team_ATI"])
        ftpe_v = float(sel_row["FTPE"])

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            t, c = ati_band(ati_v)
            metric_card("Attacking Threat Index", f"{ati_v:.3f}",
                        "Top-3 outfield ATI average", t, c)
        with kpi_cols[1]:
            if sqr_live:
                sqr_v = float(sel_row["SQR"])
                t, c = sqr_band(sqr_v)
                metric_card("Shot Quality Ratio", f"{sqr_v:.3f}",
                            "xG per shot ÷ league avg xG per shot", t, c)
            else:
                metric_card("Shot Quality Ratio", "N/A",
                            "Needs defensive_stats.csv", "Missing data", "badge-info")
        with kpi_cols[2]:
            if dcs_live:
                dcs_v = float(sel_row["DCS"])
                t, c = dcs_band(dcs_v)
                metric_card("Defensive Compactness", f"{dcs_v:.1f}",
                            "Total def actions ÷ xGA season", t, c)
            else:
                metric_card("Defensive Compactness", "N/A",
                            "Needs defensive_stats.csv", "Missing data", "badge-info")
        with kpi_cols[3]:
            t, c = ftpe_band(ftpe_v)
            metric_card("Final-Third Penetration", f"{ftpe_v:.2f}",
                        "PrgC + PrgP per 90 (team)", t, c)

        # Coach notes
        c1, c2 = st.columns([1.3, 1])
        with c1:
            ranked = team_df.sort_values("Pts", ascending=True)
            fig = go.Figure(go.Bar(
                x=ranked["Pts"], y=ranked["Squad"], orientation="h",
                marker_color=[THEME["accent"] if s == selected_team else THEME["accent_2"]
                              for s in ranked["Squad"]],
                text=ranked["Pts"], textposition="outside",
            ))
            style_plot(fig, f"League points — {selected_team} highlighted", 520)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown('<div style="margin-top:0.4rem">', unsafe_allow_html=True)
            st.subheader("Coach read-out")
            for note in coach_notes(sel_row, tp):
                st.markdown(f"- {note}")
            st.markdown("</div>", unsafe_allow_html=True)

        # xG vs xGA scatter
        fig = px.scatter(
            team_df, x="xGA", y="xG", size="Pts", color="Pts",
            text="Squad", color_continuous_scale="Tealgrn",
            labels={"xGA":"Season xGA","xG":"Season xG"},
        )
        selected_row = team_df[team_df["Squad"] == selected_team].iloc[0]
        compare_row = team_df[team_df["Squad"] == compare_team].iloc[0]
        lim = max(team_df["xG"].max(), team_df["xGA"].max()) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                      line=dict(color=THEME["warning"], dash="dot"))
        fig.update_traces(marker=dict(opacity=0.30), selector=dict(mode="markers"))
        fig.add_trace(go.Scatter(
            x=[selected_row["xGA"]], y=[selected_row["xG"]],
            mode="markers+text", text=[selected_row["Squad"]],
            textposition="top center",
            marker=dict(
                size=max(18, selected_row["Pts"] / team_df["Pts"].max() * 35 + 12),
                color=THEME["accent"], line=dict(width=2, color="#ffffff"), opacity=0.95
            ),
            name=selected_team,
        ))
        fig.add_trace(go.Scatter(
            x=[compare_row["xGA"]], y=[compare_row["xG"]],
            mode="markers+text", text=[compare_row["Squad"]],
            textposition="bottom center",
            marker=dict(
                size=max(18, compare_row["Pts"] / team_df["Pts"].max() * 35 + 12),
                color=THEME["accent_2"], line=dict(width=2, color="#ffffff"), opacity=0.95
            ),
            name=compare_team,
        ))
        style_plot(fig, "Attack vs Defence profile — above diagonal = net positive xGD", 430)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 2 — TEAMS
    # ══════════════════════════════════════════
    with tabs[1]:
        radar_base = ["Team_ATI","FTPE","xGD/90"]
        radar_lbl  = ["ATI","FTPE","xGD/90"]
        if sqr_live:
            radar_base.append("SQR"); radar_lbl.append("SQR")
        if dcs_live:
            radar_base.append("DCS"); radar_lbl.append("DCS")

        rdf  = normalize_radar(team_df, radar_base)
        sel_ = rdf[rdf["Squad"] == selected_team].iloc[0]
        cmp_ = rdf[rdf["Squad"] == compare_team].iloc[0]

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=[sel_[f"{c}_n"] for c in radar_base] + [sel_[f"{radar_base[0]}_n"]],
            theta=radar_lbl + [radar_lbl[0]], fill="toself", name=selected_team,
            line=dict(color=THEME["accent"], width=2),
            fillcolor="rgba(20,184,166,0.12)",
        ))
        radar.add_trace(go.Scatterpolar(
            r=[cmp_[f"{c}_n"] for c in radar_base] + [cmp_[f"{radar_base[0]}_n"]],
            theta=radar_lbl + [radar_lbl[0]], fill="toself", name=compare_team,
            line=dict(color=THEME["accent_2"], width=2),
            fillcolor="rgba(96,165,250,0.10)",
        ))
        radar.update_layout(
            height=500, title="Team comparison radar (normalised 0–1)",
            paper_bgcolor=THEME["surface"], plot_bgcolor=THEME["surface"],
            font=dict(color=THEME["text"]),
            polar=dict(bgcolor=THEME["surface"],
                       radialaxis=dict(range=[0,1], gridcolor=THEME["border"]),
                       angularaxis=dict(gridcolor=THEME["border"])),
        )
        st.plotly_chart(radar, use_container_width=True)

        display_cols = ["Rk","Squad","Pts","xG","xGA","xGD/90","Team_ATI","FTPE"]
        fmt = {"xG":"{:.1f}","xGA":"{:.1f}","xGD/90":"{:.2f}",
               "Team_ATI":"{:.3f}","FTPE":"{:.2f}"}
        if sqr_live:
            display_cols.append("SQR"); fmt["SQR"] = "{:.3f}"
        if dcs_live:
            display_cols.append("DCS"); fmt["DCS"] = "{:.1f}"

        st.dataframe(
            team_df[display_cols].sort_values("Pts", ascending=False)
            .reset_index(drop=True).style.format(fmt),
            use_container_width=True, hide_index=True, height=430,
        )

        fig = px.scatter(
            team_df, x="xGD", y="Pts", size="FTPE", color="Team_ATI",
            text="Squad", color_continuous_scale="Tealgrn",
            labels={"xGD":"Season xGD","Pts":"Points","FTPE":"FTPE"},
        )
        selected_row = team_df[team_df["Squad"] == selected_team].iloc[0]
        compare_row = team_df[team_df["Squad"] == compare_team].iloc[0]
        fig.add_vline(x=0, line_dash="dash", line_color=THEME["muted"])
        fig.update_traces(marker=dict(opacity=0.30), selector=dict(mode="markers"))
        fig.add_trace(go.Scatter(
            x=[selected_row["xGD"]], y=[selected_row["Pts"]],
            mode="markers+text", text=[selected_row["Squad"]],
            textposition="top center",
            marker=dict(
                size=max(18, selected_row["FTPE"] / team_df["FTPE"].max() * 35 + 12),
                color=THEME["accent"], line=dict(width=2, color="#ffffff"), opacity=0.95
            ),
            name=selected_team,
        ))
        fig.add_trace(go.Scatter(
            x=[compare_row["xGD"]], y=[compare_row["Pts"]],
            mode="markers+text", text=[compare_row["Squad"]],
            textposition="bottom center",
            marker=dict(
                size=max(18, compare_row["FTPE"] / team_df["FTPE"].max() * 35 + 12),
                color=THEME["accent_2"], line=dict(width=2, color="#ffffff"), opacity=0.95
            ),
            name=compare_team,
        ))
        style_plot(fig, "Points vs xGD (bubble = FTPE, colour = ATI)", 430)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 3 — PLAYERS
    # ══════════════════════════════════════════
    with tabs[2]:
        if tp.empty:
            st.warning("No players match the active filter set.")
        else:
            r1, r2 = st.columns(2)
            with r1:
                bar_df = tp.sort_values("ATI", ascending=True).tail(15)
                fig = go.Figure(go.Bar(
                    x=bar_df["ATI"], y=bar_df["Player"], orientation="h",
                    marker_color=THEME["accent"],
                    text=bar_df["ATI"].round(3), textposition="outside",
                    customdata=np.stack([bar_df["Role_Label"], bar_df["90s"]], axis=-1),
                    hovertemplate="<b>%{y}</b><br>ATI:%{x:.3f}<br>Role:%{customdata[0]}<br>90s:%{customdata[1]:.1f}<extra></extra>",
                ))
                style_plot(fig, f"{selected_team} — ATI ranking", 500)
                st.plotly_chart(fig, use_container_width=True)

            with r2:
                fig = px.scatter(
                    tp, x="xG_90", y="xAG_90", size="90s", color="Role_Label",
                    text="Player", hover_data=["Pos","ATI","Progression_90"],
                    color_discrete_map={
                        "Forward": THEME["accent"], "Midfielder": THEME["accent_2"],
                        "Defender": THEME["warning"], "Goalkeeper": THEME["danger"],
                    },
                )
                fig.update_traces(textposition="top center")
                style_plot(fig, f"{selected_team} — scoring threat vs chance creation", 500)
                st.plotly_chart(fig, use_container_width=True)

            r3, r4 = st.columns(2)
            with r3:
                fig = px.scatter(
                    tp, x="Progression_90", y="ATI", size="90s", color="Role_Label",
                    text="Player", hover_data=["xG_90","xAG_90"],
                    color_discrete_map={
                        "Forward": THEME["accent"], "Midfielder": THEME["accent_2"],
                        "Defender": THEME["warning"], "Goalkeeper": THEME["danger"],
                    },
                )
                fig.update_traces(textposition="top center")
                style_plot(fig, f"{selected_team} — progression vs ATI", 430)
                st.plotly_chart(fig, use_container_width=True)

            with r4:
                by_role = tp.groupby("Role_Label")[["xG_90","xAG_90","Progression_90"]].mean().reset_index()
                melt    = by_role.melt(id_vars="Role_Label", var_name="Metric", value_name="Per90")
                fig = px.bar(melt, x="Role_Label", y="Per90", color="Metric", barmode="group",
                             color_discrete_map={"xG_90": THEME["accent"],
                                                 "xAG_90": THEME["accent_2"],
                                                 "Progression_90": THEME["warning"]})
                style_plot(fig, f"{selected_team} — average per-90 by role", 430)
                st.plotly_chart(fig, use_container_width=True)

            # Player table
            view_cols = ["Player","Pos","Role_Label","90s","Starts","Gls","Ast",
                         "xG_90","xAG_90","Progression_90","ATI","Selection_Load","Sample_Flag"]
            # If defensive data is live, surface the core defensive counts too
            if dcs_live and any(c in tp.columns for c in ["Def_Actions_90","TklW","Int","Clr","Sh"]):
                # Add counts before the sample flag
                def_cols = [c for c in ["Sh","TklW","Int","Clr","Def_Actions_90"] if c in tp.columns]
                for c in reversed(def_cols):
                    view_cols.insert(-1, c)
            st.dataframe(
                tp[view_cols].sort_values("ATI", ascending=False)
                .reset_index(drop=True).style.format({
                    "90s":"{:.1f}","xG_90":"{:.2f}","xAG_90":"{:.2f}",
                    "Progression_90":"{:.2f}","ATI":"{:.3f}","Selection_Load":"{:.2f}",
                    "Def_Actions_90":"{:.2f}","Sh":"{:.0f}","TklW":"{:.1f}","Int":"{:.1f}","Clr":"{:.1f}",
                }),
                use_container_width=True, hide_index=True, height=480,
            )

    # ══════════════════════════════════════════
    # TAB 4 — COACHING LAB
    # ══════════════════════════════════════════
    with tabs[3]:
        sub_tabs = st.tabs(["Selection Signals","Substitution Lab","Formation Sandbox"])

        with sub_tabs[0]:
            if tp.empty:
                st.warning("No players match current filters.")
            else:
                sel_df = tp.copy()
                prog_max = max(sel_df["Progression_90"].max(), 1e-9)
                sel_df["Coach_Value"] = (0.50 * sel_df["ATI"]
                    + 0.30 * sel_df["Progression_90"] / prog_max
                    + 0.20 * sel_df["G+A_90"])
                sel_df["Usage_Tag"] = np.select(
                    [sel_df["Selection_Load"] >= 0.70,
                     sel_df["Selection_Load"] >= 0.40],
                    ["Likely starter","Rotation option"],
                    default="Impact sub / depth",
                )
                shortlist_cols = ["Player","Role_Label","90s","Starts","ATI",
                                  "xAG_90","Progression_90","Coach_Value","Usage_Tag"]
                st.dataframe(
                    sel_df[shortlist_cols].sort_values("Coach_Value", ascending=False)
                    .reset_index(drop=True).style.format({
                        "90s":"{:.1f}","ATI":"{:.3f}","xAG_90":"{:.2f}",
                        "Progression_90":"{:.2f}","Coach_Value":"{:.3f}",
                    }),
                    use_container_width=True, hide_index=True, height=450,
                )
                
        with sub_tabs[1]:
            if len(tp) < 2:
                st.warning("Need at least 2 filtered players.")
            else:
                ordered = tp.sort_values("ATI", ascending=False)["Player"].tolist()
                c1, c2, c3 = st.columns(3)
                with c1: out_p = st.selectbox("Outgoing player", ordered, index=min(1, len(ordered)-1))
                with c2: in_p  = st.selectbox("Incoming player", ordered, index=0)
                with c3: rem   = st.slider("Remaining minutes", 5, 60, 30)

                out_row = tp[tp["Player"] == out_p].iloc[0]
                in_row  = tp[tp["Player"] == in_p].iloc[0]

                atk_out = float(out_row["SID_atk_idx"])
                atk_in  = float(in_row["SID_atk_idx"])

                if dcs_live and not pd.isna(out_row.get("SID_def_idx", np.nan)):
                    def_out = float(out_row["SID_def_idx"])
                    def_in  = float(in_row["SID_def_idx"])
                    st.markdown('<div class="info-box">Defensive indices are from real tackles + interceptions + clearances per 90.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warn-box">Defensive file not loaded — enter estimated defensive scores manually.</div>', unsafe_allow_html=True)
                    cc1, cc2 = st.columns(2)
                    with cc1: def_out = st.slider("Outgoing def score", 0.0, 6.0, 2.0, 0.1)
                    with cc2: def_in  = st.slider("Incoming def score", 0.0, 6.0, 2.5, 0.1)

                xg_pm   = float(sel_row["xG"]) / (float(sel_row["MP"]) * 90)
                xga_pm  = float(sel_row["xGA"]) / (float(sel_row["MP"]) * 90)
                sid, xg_b, xg_n, xga_b, xga_n = compute_sid(
                    rem, xg_pm, xga_pm, atk_out, atk_in, def_out, def_in)

                st_txt, st_cls = sid_band(sid)
                m1, m2, m3 = st.columns(3)
                with m1: metric_card("SID", f"{sid:+.3f}",
                                     "Net xG balance change", st_txt, st_cls)
                with m2: metric_card("Baseline xG created", f"{xg_b:.3f}",
                                     f"Next {rem} mins", "Baseline", "badge-info")
                with m3: metric_card("Scenario xG created", f"{xg_n:.3f}",
                                     "After sub", "Scenario", "badge-strong")

                summary = pd.DataFrame({
                    "Metric":["Atk idx out","Atk idx in","Def idx out","Def idx in",
                              "Baseline xG created","Scenario xG created",
                              "Baseline xGA","Scenario xGA"],
                    "Value":[atk_out, atk_in, def_out, def_in,
                             xg_b, xg_n, xga_b, xga_n],
                })
                st.dataframe(summary.style.format({"Value":"{:.3f}"}),
                             use_container_width=True, hide_index=True)

        with sub_tabs[2]:
            st.markdown(
                '<div class="info-box"><b>Formation Sandbox</b> — 10,000 Poisson simulations per scenario. '                'All base rates come from real season xG/xGA. The only assumed values are the formation '                'attack/defence multipliers. Changing the opponent changes the λ values and therefore '                'the Win/Draw/Loss probabilities.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            # ── Controls ──
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                cur_f = st.selectbox(
                    "Current formation", list(FORMATION_MULTIPLIERS), index=4,
                    help="Your team's current shape (4-4-2 is the neutral baseline).")
            with col_b:
                new_f = st.selectbox(
                    "Proposed formation", list(FORMATION_MULTIPLIERS), index=5,
                    help="The formation you are considering switching to.")
            # Use the main comparison team as the opponent for the sandbox
            with col_c:
                st.markdown("Opponent linked to main 'Comparison team' filter.")
            opponent = compare_team

            opp_row = team_df[team_df["Squad"] == opponent].iloc[0]
            res     = compute_fwpl(sel_row, opp_row, cur_f, new_f)
            fwpl_v  = res["fwpl"]

            # ── Top metric cards ──
            ft, fc = fwpl_band(fwpl_v)
            fm1, fm2, fm3, fm4 = st.columns(4)
            with fm1:
                metric_card("FWPL", f"{fwpl_v:+.3f}",
                            "Win% shift vs selected opponent", ft, fc)
            with fm2:
                metric_card(f"Win% · {cur_f}",
                            f"{res['curr'][0]*100:.1f}%",
                            f"vs {opponent}", "Current", "badge-info")
            with fm3:
                metric_card(f"Win% · {new_f}",
                            f"{res['new_'][0]*100:.1f}%",
                            f"vs {opponent}", "Proposed", "badge-strong")
            with fm4:
                metric_card("λ matchup (proposed)",
                            f"{res['lam_for_new']:.2f} v {res['lam_opp_new']:.2f}",
                            f"{selected_team} xG | {opponent} xG",
                            "Expected goals/match", "badge-moderate")

            st.markdown("")

            # ── LEFT chart: outcome probability bars for selected opponent ──
            c1, c2 = st.columns(2)
            with c1:
                outcomes  = ["Win", "Draw", "Loss"]
                cur_vals  = list(res["curr"])
                new_vals  = list(res["new_"])
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    x=outcomes, y=[v*100 for v in cur_vals], name=cur_f,
                    marker_color=THEME["accent_2"],
                    text=[f"{v*100:.1f}%" for v in cur_vals],
                    textposition="outside",
                ))
                fig1.add_trace(go.Bar(
                    x=outcomes, y=[v*100 for v in new_vals], name=new_f,
                    marker_color=THEME["accent"],
                    text=[f"{v*100:.1f}%" for v in new_vals],
                    textposition="outside",
                ))
                fig1.update_layout(barmode="group", yaxis_range=[0, 105],
                                   yaxis_title="Probability (%)")
                style_plot(fig1, f"{selected_team} vs {opponent} — outcome probabilities", 420)
                st.plotly_chart(fig1, use_container_width=True)

            # ── RIGHT chart: Win% for CURRENT formation across ALL opponents ──
            # This is the proof that opponent changes the result
            with c2:
                # Build list of opponents (all other teams) for the across-opponents chart
                opp_options = [t for t in teams if t != selected_team]
                all_win = []
                for oname in opp_options:
                    orow = team_df[team_df["Squad"] == oname].iloc[0]
                    r    = compute_fwpl(sel_row, orow, cur_f, new_f)
                    all_win.append({
                        "Opponent": oname,
                        f"{cur_f} Win%": round(r["curr"][0]*100, 1),
                        f"{new_f} Win%": round(r["new_"][0]*100, 1),
                        "FWPL": round(r["fwpl"]*100, 2),
                    })
                aw_df = pd.DataFrame(all_win).sort_values(f"{new_f} Win%", ascending=True)

                # Highlight selected opponent
                colours_cur = [
                    THEME["warning"] if row["Opponent"] == opponent else THEME["accent_2"]
                    for _, row in aw_df.iterrows()
                ]
                colours_new = [
                    THEME["warning"] if row["Opponent"] == opponent else THEME["accent"]
                    for _, row in aw_df.iterrows()
                ]

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    y=aw_df["Opponent"],
                    x=aw_df[f"{cur_f} Win%"],
                    name=cur_f,
                    orientation="h",
                    marker_color=colours_cur,
                    opacity=0.7,
                ))
                fig2.add_trace(go.Bar(
                    y=aw_df["Opponent"],
                    x=aw_df[f"{new_f} Win%"],
                    name=new_f,
                    orientation="h",
                    marker_color=colours_new,
                    opacity=0.9,
                ))
                fig2.update_layout(
                    barmode="overlay",
                    xaxis_title="Win % (10,000 simulations)",
                    xaxis_range=[0, 100],
                    legend=dict(orientation="h", y=1.08),
                )
                style_plot(fig2,
                    f"{selected_team} {cur_f}→{new_f} Win% vs every PL team  "
                    f"(highlighted = {opponent})", 500)
                st.plotly_chart(fig2, use_container_width=True)

            # ── Full probability table ──
            prob_table = pd.DataFrame({
                "Formation": [cur_f, new_f],
                "Opponent":  [opponent, opponent],
                "Opp att str": [res["opp_att_str"], res["opp_att_str"]],
                "Opp def str": [res["opp_def_str"], res["opp_def_str"]],
                "λ team":    [res["lam_for_cur"], res["lam_for_new"]],
                "λ opp":     [res["lam_opp_cur"], res["lam_opp_new"]],
                "Win %":     [res["curr"][0]*100, res["new_"][0]*100],
                "Draw %":    [res["curr"][1]*100, res["new_"][1]*100],
                "Loss %":    [res["curr"][2]*100, res["new_"][2]*100],
                "FWPL":      [0.0, fwpl_v*100],
            })
            st.dataframe(
                prob_table.style.format({
                    "Opp att str":"{:.3f}", "Opp def str":"{:.3f}",
                    "λ team":"{:.3f}", "λ opp":"{:.3f}",
                    "Win %":"{:.2f}", "Draw %":"{:.2f}", "Loss %":"{:.2f}",
                    "FWPL":"{:+.2f}",
                }),
                use_container_width=True, hide_index=True,
            )

            st.markdown(
                '<div class="warn-box"><b>Note:</b> Formation multipliers (att/def factors) are the only assumed values. All xG/xGA base rates come from the season data. Changing the opponent updates λ_team and λ_opp and the probabilities shown.</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════
    # TAB 5 — ABOUT DATA
    # ══════════════════════════════════════════
    with tabs[4]:
        st.subheader("KPI data source map")
        rows = [
            ["ATI","fbref_pl: xG, xAG, 90s"],
            ["FTPE","fbref_pl: PrgC, PrgP, 90s"],
            ["FWPL","team_stats: xG, xGA, MP + Poisson sim"],
            ["SID attacking side","fbref_pl: xG, xAG, 90s"],
            ["SID defensive side","defensive_stats: TklW, Int, Clr, 90s"],
            ["SQR","defensive_stats: Sh per player → team total shots"],
            ["DCS","defensive_stats: TklW, Int, Clr → team total; team_stats: xGA"],
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["KPI","Fields used"]),
            use_container_width=True, hide_index=True,
        )


        st.subheader("Column glossary")
        gloss = pd.DataFrame([
            ["xG","fbref_pl / def_stats","Expected goals from player shots"],
            ["xAG","fbref_pl","Expected goal value of chances created by a player"],
            ["90s","fbref_pl","Full 90-minute equivalents played"],
            ["PrgC","fbref_pl","Progressive carries (ball moved ≥10m toward goal)"],
            ["PrgP","fbref_pl","Progressive passes (completed, ball moved ≥10m toward goal)"],
            ["PrgR","fbref_pl","Progressive receptions (off-ball movement into danger)"],
            ["Sh","def_stats","Total shots taken by the player"],
            ["TklW","def_stats","Tackles won"],
            ["Int","def_stats","Interceptions"],
            ["Clr","def_stats","Clearances"],
            ["ATI","Derived","0.6 × xG/90 + 0.4 × xAG/90"],
            ["SQR","Derived","(team xG ÷ team shots) ÷ (league xG ÷ league shots)"],
            ["DCS","Derived","(TklW+Int+Clr season total) ÷ xGA season total"],
            ["FTPE","Derived","(ΣPrgC + ΣPrgP) ÷ Σ90s per squad"],
            ["FWPL","Derived","Win%(new formation) − Win%(current), Poisson 10k sim"],
            ["SID","Derived","Net xG balance change from player swap over remaining minutes"],
        ], columns=["Field","Source","Meaning"])
        st.dataframe(gloss, use_container_width=True, hide_index=True, height=450)

    st.markdown("---")
    st.caption("KPIs are computed from FBref fields; no proxy values are used.")


if __name__ == "__main__":
    main()
