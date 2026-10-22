"""
DAILY DASHBOARD, 09:45 CUTOFF -- step14_dashboard_om_trailcompare_945cutoff_daily.py

Description: Live dashboard for step10_live_paper_om_trailcompare_945cutoff_DAILY.py
             -- shows the trail1pt_v2 variant ONLY (1.0pt trail, tightened
             from 2.0pt -- see TRAIL WIDTH TIGHTENED comment below for the
             validation numbers) (09:45 entry cutoff restored, NOT all-day,
             20s poll), reading
             (openmomentum_trailcompare_945cutoff_daily_paper_portfolio.db).
             Historical trail1pt/trail2pt/trail3pt rows (pre-cooldown-fix)
             remain in the DB untouched, just not rendered here anymore --
             trail1pt_v2 is a fresh label so the new, validated numbers
             never mix with that stale history.

Starting tomorrow this is the dashboard to use -- NOT
step14_dashboard_om_trailcompare_nocutoff_temp.py (that one is today-only,
all-day entries, and its DB stays exactly as today left it).

Port map for this strategy family, for reference:
    8503  step14_dashboard_om_trailcompare_realpremium.py        (earlier before945 run, not daily-use)
    8504  step14_dashboard_om_trailcompare_nocutoff_temp.py       (TODAY ONLY, all-day entries)
    8505  step14_dashboard_om_trailcompare_945cutoff_daily.py     (THIS FILE -- daily use from tomorrow, 09:45 cutoff)

RUN ON ITS OWN PORT:
    streamlit run step14_dashboard_om_trailcompare_945cutoff_daily.py --server.port 8505

then open http://localhost:8505 in a NEW tab.

VISUAL IDENTITY -- a third distinct palette/font so this is unmistakably
different from both other dashboards at a glance: warm amber-and-teal
with 'Manrope' / 'Source Code Pro'.

Pure read-only view -- does not call the Dhan API, does not import
step10, so a fast UI refresh never touches the live engine.
"""

import os
import sys
import datetime
import sqlite3
import pandas as pd
import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from trade_cost_model import compute_trade_cost  # real brokerage+charges formula, used for the "Live P&L" estimate on open positions

st.set_page_config(
    page_title="[DAILY 09:45] Trail Compare 1/2/3pt | Live Monitor",
    page_icon="\u23F0",
    layout="wide",
    initial_sidebar_state="expanded"
)

WAIT_MSG = "Waiting for Live Data"
DB_PATH = r"C:\dhan algo\Research\logs\openmomentum_trailcompare_945cutoff_daily_paper_portfolio.db"
STARTING_CAPITAL = 25_000.0

VARIANTS = [
    # trail1pt/trail2pt/trail3pt (pre-cooldown-fix) REMOVED per explicit
    # request, matching step10_live_paper_om_trailcompare_945cutoff_DAILY.py.
    # TRAIL WIDTH TIGHTENED (2.0pt -> 1.0pt): sweep on top of the validated
    # cooldown fix found tighter trail helps this strategy -- real9 PF
    # 2.40->2.64 (net +Rs2,514->+Rs2,694), plain122 PF 2.43->2.63, noise122
    # PF 2.51->2.63, all three datasets agreeing. OOS-checked, no collapse.
    # Relabeled to trail1pt_v2 (not the old "trail1pt") so this validated
    # data never mixes with the stale pre-cooldown-fix history.
    {"label": "trail1pt_v2", "display": "Trail 1.0pt", "accent": "#2DD4BF"},
]

# --- Sidebar ---
st.sidebar.markdown("## \u23F0 Daily Trail Compare (09:45 cutoff)")
st.sidebar.caption("Real Dhan option-chain LTP -- 1.0pt trail ONLY (tightened from 2pt, validated), entries "
                    "only before 09:45, 20s poll. This is the permanent daily-use engine.")
theme_mode = st.sidebar.radio("Theme Appearance", ["\U0001F506 Warm Amber", "\U0001F30A Cool Teal"], index=0)
is_dark = (theme_mode == "\U0001F506 Warm Amber")

if is_dark:
    bg_color, card_bg, text_color, sub_text, border_color = "#1A1410", "#251D17", "#F5EDE3", "#B3A292", "#3D2F24"
else:
    bg_color, card_bg, text_color, sub_text, border_color = "#F0F9F8", "#FFFFFF", "#122421", "#5E7C78", "#D3E8E5"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700&family=Source+Code+Pro:wght@400;600&display=swap');
    html, body, [class*="css"]  {{ font-family: 'Manrope', sans-serif; }}
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .mono {{ font-family: 'Source Code Pro', monospace; }}
    .variant-header {{ font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 26px;
                        letter-spacing: -0.01em; margin-bottom: 2px; }}
    .temp-banner {{ background-color: #7C2D12; color: #FFF7ED; font-family: 'Source Code Pro', monospace;
                     font-weight: 600; font-size: 13px; letter-spacing: 0.04em; padding: 10px 16px;
                     border-radius: 10px; margin-bottom: 16px; border: 1px solid #EA580C; }}
    .metric-card {{ background-color: {card_bg}; border: 1px solid {border_color}; padding: 16px;
                     border-radius: 14px; margin-bottom: 10px; }}
    .metric-title {{ font-family: 'Source Code Pro', monospace; font-size: 11px; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.08em; color: {sub_text}; margin-bottom: 6px; }}
    .metric-value {{ font-family: 'Manrope', sans-serif; font-size: 22px; font-weight: 700; color: {text_color}; }}
    .metric-delta-pos {{ font-family: 'Source Code Pro', monospace; font-size: 12px; font-weight: 600;
                          color: #2DD4BF; margin-top: 4px; }}
    .metric-delta-neg {{ font-family: 'Source Code Pro', monospace; font-size: 12px; font-weight: 600;
                          color: #FB923C; margin-top: 4px; }}
    .metric-delta-neu {{ font-family: 'Source Code Pro', monospace; font-size: 12px; font-weight: 600;
                          color: #FACC15; margin-top: 4px; }}
    .position-card {{ background-color: {card_bg}; border: 2px solid var(--accent); padding: 18px;
                       border-radius: 16px; margin-bottom: 12px; }}
    .position-badge {{ display:inline-block; font-family: 'Source Code Pro', monospace; font-weight:700;
                        font-size:11px; letter-spacing: 0.06em; text-transform: uppercase;
                        background-color: var(--accent); color:#1A1410; padding:4px 12px; border-radius:999px;
                        margin-bottom:8px; }}
    .reason-box {{ background-color: {card_bg}; border-left: 3px solid var(--accent); padding: 10px 14px;
                    border-radius: 6px; font-size: 13px; color: {text_color}; margin-top: 8px;
                    font-family: 'Source Code Pro', monospace; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {card_bg}; border-radius: 8px; color: {sub_text};
                                     padding: 8px 16px; font-weight: 600; border: 1px solid {border_color}; }}
    .stTabs [aria-selected="true"] {{ background-color: var(--accent) !important; color: #1A1410 !important; }}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Reads from:**\n`logs/openmomentum_trailcompare_945cutoff_daily_paper_portfolio.db`")
st.sidebar.markdown("**Comparing:** stop=20pt fixed, entry logic identical, cutoff=09:45 -- "
                     "only trail width differs.")
st.sidebar.markdown(f"**Capital:** Rs {STARTING_CAPITAL:,.0f} independent pool per variant.")
st.sidebar.markdown("---")
st.sidebar.caption("\U0001F6E1️ **Exhaustion-top fix added**: a 5-bar (10min) cooldown after every exit "
                    "before a new signal can fire -- stops the engine from immediately re-chasing the tail "
                    "end of the same exhausted move right after a stop-out. Validated: real 9-day PF "
                    "0.79->2.40, net -Rs1,017.50->+Rs2,513.75. (An ATR extension cap was tried first and "
                    "did NOT work for this strategy's re-anchoring logic -- cooldown is the one that stuck.)")
st.sidebar.markdown("---")
st.sidebar.caption("Start the engine separately:\n`python step10_live_paper_om_trailcompare_945cutoff_DAILY.py`")
st.sidebar.caption("Run THIS dashboard on its own port:\n"
                    "`streamlit run step14_dashboard_om_trailcompare_945cutoff_daily.py --server.port 8505`")


def query_db(query: str, params=()) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


def fmt_money(val, fallback=WAIT_MSG):
    if val is None or pd.isna(val):
        return fallback
    return f"\u20b9{float(val):,.2f}"


def fmt_num(val, fallback=WAIT_MSG, decimals=2):
    if val is None or pd.isna(val):
        return fallback
    return f"{float(val):,.{decimals}f}"


def card(col, title, value, delta="", d_type="neu"):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value mono">{value}</div>
            <div class="metric-delta-{d_type}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown('<div class="temp-banner">\u23F0 DAILY LIVE PAPER TRADING -- entries only before 09:45 '
            '(cutoff restored), 1.0pt trail ONLY (tightened from 2pt, validated), 20s poll. Separate DB '
            'from both the earlier before945 run and today\'s all-day temp run.</div>', unsafe_allow_html=True)

st.title("\u23F0 Trail Compare -- 1.0pt ONLY (tightened from 2pt, validated) (09:45 Cutoff, Real Premium)")
st.markdown("Unlimited multi-entry, 20pt fixed stop, 1.0pt trail (tightened from 2pt, validated). Entries only before 09:45. Real Dhan "
            "option-chain LTP, 20s poll interval. (1pt and 3pt trail variants have been retired -- their "
            "historical trades remain in the same DB but are no longer shown here.) Gross P&L, "
            "Brokerage+Charges, and Net P&L are all shown separately below, same real cost model as the "
            "VWAP20EMA dashboard.")

if not os.path.exists(DB_PATH):
    st.warning("No database found yet. Start `python step10_live_paper_om_trailcompare_945cutoff_DAILY.py` "
               "to begin generating live data.")


# ================= HEAD-TO-HEAD SUMMARY (5s refresh) =================
@st.fragment(run_every=5)
def render_summary():
    current_time = datetime.datetime.now()
    today_str = str(current_time.date())

    summary_cols = st.columns(len(VARIANTS))
    for i, v in enumerate(VARIANTS):
        df_closed_all = query_db("SELECT * FROM paper_orders WHERE variant=? AND status='CLOSED' ORDER BY id DESC",
                                  (v["label"],))
        df_open = query_db("SELECT * FROM paper_orders WHERE variant=? AND status='OPEN' AND date=? ORDER BY id DESC",
                            (v["label"], today_str))

        capital_now = float(df_closed_all.iloc[0]["capital_after"]) if not df_closed_all.empty else STARTING_CAPITAL
        total_realized = capital_now - STARTING_CAPITAL
        df_today = df_closed_all[df_closed_all["date"] == today_str] if not df_closed_all.empty else pd.DataFrame()
        today_pnl = float(df_today["rupee_pnl"].sum()) if not df_today.empty else 0.0
        total_trades = len(df_closed_all)
        wins = int((df_closed_all["rupee_pnl"] > 0).sum()) if not df_closed_all.empty else 0
        win_pct = (wins / total_trades * 100) if total_trades else None

        with summary_cols[i]:
            st.markdown(f'<div class="variant-header" style="color:{v["accent"]}">{v["display"]}</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            card(c1, "Capital", fmt_money(capital_now),
                 f"{'+' if total_realized >= 0 else ''}{fmt_money(total_realized)} all-time",
                 "pos" if total_realized >= 0 else "neg")
            card(c2, "Today P&L", fmt_money(today_pnl) if not df_today.empty else "\u20b90.00",
                 f"{len(df_today)} closed today", "pos" if today_pnl >= 0 else "neu")
            card(c3, "Win Rate", f"{win_pct:.1f}%" if win_pct is not None else WAIT_MSG,
                 f"{wins}/{total_trades} all-time")

            if not df_open.empty:
                pos = df_open.iloc[0]
                unreal_gross = unreal_charges = unreal_net = None
                if pd.notna(pos.get("last_price")) and pd.notna(pos.get("entry_price")) and pd.notna(pos.get("lot_size")):
                    _entry, _last, _lot = float(pos["entry_price"]), float(pos["last_price"]), float(pos["lot_size"])
                    unreal_gross = (_last - _entry) * _lot
                    unreal_charges = compute_trade_cost(_entry, _last, _lot)
                    unreal_net = unreal_gross - unreal_charges
                st.markdown(f"""
                <div class="position-card" style="--accent: {v['accent']}">
                    <div class="position-badge" style="--accent: {v['accent']}">POSITION OPEN</div>
                    <div style="font-size:16px; font-weight:700;">{pos.get('side','')} {pos.get('direction','')} -- Strike {fmt_num(pos.get('strike'), fallback='N/A', decimals=0)}</div>
                    <div class="mono" style="margin-top:6px; font-size:13px;">
                        Entry {fmt_money(pos.get('entry_price'))} @ {pos.get('entry_time','')}<br>
                        Last {fmt_money(pos.get('last_price'))} | Stop {fmt_money(pos.get('current_stop'))} | Peak {fmt_money(pos.get('peak_price'))}<br>
                        Gross P&L: {fmt_money(unreal_gross) if unreal_gross is not None else WAIT_MSG} &nbsp;|&nbsp;
                        Brokerage+Charges (est.): -{fmt_money(unreal_charges) if unreal_charges is not None else WAIT_MSG}<br>
                        Net P&L (after charges): <b>{fmt_money(unreal_net) if unreal_net is not None else WAIT_MSG}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"No open {v['display']} position -- scanning for entries (before 09:45).")

    st.markdown("---")
    st.caption(f"Last refreshed: {current_time.strftime('%H:%M:%S')}")


render_summary()


# ================= DETAIL TABS PER VARIANT =================
tab_labels = [v["display"] for v in VARIANTS] + ["Combined Event Log"]
tabs = st.tabs(tab_labels)

for i, v in enumerate(VARIANTS):
    with tabs[i]:
        today_str = str(datetime.date.today())
        df_closed_all = query_db("SELECT * FROM paper_orders WHERE variant=? AND status='CLOSED' ORDER BY id DESC",
                                  (v["label"],))
        df_today = df_closed_all[df_closed_all["date"] == today_str] if not df_closed_all.empty else pd.DataFrame()

        st.subheader(f"{v['display']} -- Today's Trade Log")
        if not df_today.empty:
            df_today = df_today.copy()
            df_today["gross_pnl"] = df_today["pnl_pts"] * df_today["lot_size"]
            df_today["charges"] = df_today["gross_pnl"] - df_today["rupee_pnl"]
            log = df_today.rename(columns={
                "direction": "Direction", "side": "Side", "strike": "Strike",
                "entry_time": "Entry Time", "entry_price": "Entry Price",
                "peak_price": "Peak Premium", "current_stop": "Final Stop",
                "exit_time": "Exit Time", "exit_price": "Exit Price",
                "exit_reason": "Exit Reason", "pnl_pts": "P&L (pts)",
                "gross_pnl": "Gross P&L (\u20b9)", "charges": "Brokerage+Charges (\u20b9)",
                "rupee_pnl": "Net P&L (\u20b9)",
            })
            show_cols = ["Direction", "Side", "Strike", "Entry Time", "Entry Price",
                         "Exit Time", "Exit Price", "Exit Reason", "Peak Premium", "Final Stop",
                         "P&L (pts)", "Gross P&L (\u20b9)", "Brokerage+Charges (\u20b9)", "Net P&L (\u20b9)"]
            show_cols = [c for c in show_cols if c in log.columns]
            st.dataframe(log[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades yet today for this variant.")

        st.subheader(f"{v['display']} -- All-Time History (last 30)")
        if not df_closed_all.empty:
            hist_raw = df_closed_all.head(30).copy()
            hist_raw["gross_pnl"] = hist_raw["pnl_pts"] * hist_raw["lot_size"]
            hist_raw["charges"] = hist_raw["gross_pnl"] - hist_raw["rupee_pnl"]
            hist = hist_raw.rename(columns={
                "date": "Date", "direction": "Direction", "side": "Side", "strike": "Strike",
                "entry_time": "Entry Time", "exit_time": "Exit Time",
                "exit_reason": "Exit Reason", "peak_price": "Peak Premium",
                "gross_pnl": "Gross P&L (\u20b9)",
                "charges": "Brokerage+Charges (\u20b9)", "rupee_pnl": "Net P&L (\u20b9)",
            })
            cols = ["Date", "Direction", "Side", "Strike", "Entry Time", "Exit Time", "Exit Reason",
                    "Peak Premium", "Gross P&L (\u20b9)", "Brokerage+Charges (\u20b9)", "Net P&L (\u20b9)"]
            cols = [c for c in cols if c in hist.columns]
            st.dataframe(hist[cols], use_container_width=True, hide_index=True)

            st.subheader(f"{v['display']} -- Cumulative Net P&L")
            chart_df = hist_raw.sort_values("id")[["id", "rupee_pnl"]].copy()
            chart_df["cum_pnl"] = chart_df["rupee_pnl"].cumsum()
            st.line_chart(chart_df.set_index("id")["cum_pnl"])
        else:
            st.info(WAIT_MSG)

with tabs[-1]:
    st.subheader("Combined Live Event Log (all variants, most recent first)")
    df_events = query_db("SELECT * FROM event_logs ORDER BY id DESC LIMIT 40")
    if not df_events.empty:
        for _, row in df_events.iterrows():
            accent = next((v["accent"] for v in VARIANTS if v["label"] == row.get("variant")), "#B3A292")
            st.markdown(f'<span class="mono" style="color:{accent}; font-weight:700;">[{row.get("variant","-")}]</span> '
                        f'**{row.get("timestamp","N/A")}** -- {row.get("event_message","N/A")}',
                        unsafe_allow_html=True)
    else:
        st.warning(WAIT_MSG)
