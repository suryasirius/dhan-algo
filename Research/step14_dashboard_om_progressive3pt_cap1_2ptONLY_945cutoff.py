"""
PROGRESSIVE-3pt CAP=1, 09:45 CUTOFF, 2pt-TRAIL-ONLY -- PERMANENT DASHBOARD
step14_dashboard_om_progressive3pt_cap1_2ptONLY_945cutoff.py

PERMANENT version of step14_dashboard_om_progressive3pt_cap1_2ptONLY_945cutoff_TEMP.py.
The old TEMP dashboard (and its DB, its engine, its control folder) is left
completely UNTOUCHED as a historical record -- this is a separate file,
reading a brand-new, separate, EMPTY DB
(logs/om_progressive3pt_cap1_2ptonly_945cutoff_paper_portfolio.db) written
by the new permanent engine
(step10_live_paper_om_progressive3pt_cap1_2ptONLY_945cutoff.py). Do not run
the TEMP pair and this permanent pair at the same time on the same capital
pool -- they are independent ledgers; running the TEMP dashboard afterwards
still works fine for reviewing that historical session.

Description: Live dashboard for
             step10_live_paper_om_progressive3pt_cap1_2ptONLY_945cutoff.py --
             progressive-3pt entry, cap=1, entries cut off at
             09:45, 2pt-trail-ONLY variant, real Dhan option-chain LTP,
             20s poll interval, real brokerage+charges cost model.

Fully separate from every other dashboard in this project -- own DB, own
port, own control folder for manual exit, own visual identity. Reading
this DB (or any dashboard) makes ZERO Dhan API calls -- pure SQLite reads
off disk on a Streamlit auto-refresh, so running this alongside any
number of live engines or other dashboards cannot cause rate-limit
collisions.

Port map for this strategy family, for reference:
    8503  step14_dashboard_om_trailcompare_realpremium.py               (retired)
    8504  step14_dashboard_om_trailcompare_nocutoff_temp.py              (all-day, daily use)
    8505  step14_dashboard_om_trailcompare_945cutoff_daily.py            (plain 5pt, 09:45 cutoff, daily use)
    8506  step14_dashboard_om_progressive3pt_trailcompare_945cutoff.py   (progressive 3pt, single-position)
    8507  (candle-wipe dashboard, if/when built)
    8508  step14_dashboard_om_progressive3pt_cap2_945cutoff.py           (progressive 3pt, CAP=2, 09:45 cutoff)
    8509  step14_dashboard_om_progressive3pt_cap2_nocutoff_TEMP.py       (CAP=2, NO CUTOFF, 1pt/2pt/3pt TEMP)
    8510  step14_dashboard_om_progressive3pt_cap2_nocutoff_2ptONLY_TEMP.py (CAP=2, NO CUTOFF, 2pt ONLY, TEMP)
    8511  step14_dashboard_om_progressive3pt_cap1_2ptONLY_945cutoff_TEMP.py (CAP=1, 09:45 CUTOFF, 2pt ONLY, TEMP -- historical, left untouched)
    8512  step14_dashboard_om_trailcompare_2ptonly_nocutoff_DAILY.py     (DAILY, NO CUTOFF, 2pt ONLY)
    8523  step14_dashboard_om_progressive3pt_cap1_2ptONLY_945cutoff.py  (THIS FILE -- CAP=1, 09:45 CUTOFF, 2pt ONLY, PERMANENT)

RUN ON ITS OWN PORT:
    streamlit run step14_dashboard_om_progressive3pt_cap1_2ptONLY_945cutoff.py --server.port 8523

then open http://localhost:8523 in a NEW tab.

VISUAL IDENTITY -- indigo-and-rose with 'Outfit' / 'Fira Code', distinct
from both the emerald-and-violet (1pt/2pt/3pt no-cutoff) and amber-and-slate
(2pt-only no-cutoff) dashboards, so all three are unmistakably different at
a glance.

Manual exit: click the button on an open position card, or use the CLI
`python manual_exit_progressive3pt_cap2.py trail2pt` (writes a flag that
closes ALL open positions for the trail2pt variant, matching how the live
engine's manual-exit handling works). Writes to THIS engine's own control
folder (logs/control_progressive3pt_cap1_2ptonly_945cutoff/), never
any other engine's control folder.
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
    page_title="[PROGRESSIVE 3pt / CAP=1 / 09:45-CUTOFF / 2pt-ONLY] Live Monitor",
    page_icon="\U0001F9EA",
    layout="wide",
    initial_sidebar_state="expanded"
)

WAIT_MSG = "Waiting for Live Data"
DB_PATH = r"C:\dhan algo\Research\logs\om_progressive3pt_cap1_2ptonly_945cutoff_paper_portfolio.db"
CONTROL_DIR = r"C:\dhan algo\Research\logs\control_progressive3pt_cap1_2ptonly_945cutoff"
STARTING_CAPITAL = 25_000.0
MAX_POSITIONS = 1

VARIANTS = [
    {"label": "trail2pt", "display": "Trail 2pt", "accent": "#6366F1"},
]

# --- Sidebar ---
st.sidebar.markdown("## \U0001F947 Progressive-3pt, Cap=1, 09:45 CUTOFF, 2pt ONLY")
st.sidebar.caption("Same progressive-3pt entry and cap=1 (one position at a time) as the no-cutoff 2pt-only engine, but "
                    "entries are cut off at 09:45 instead of running all session -- only the trail2pt variant "
                    "runs. Real Dhan option-chain LTP, 20s poll interval. Fully separate engine/DB from every "
                    "other engine in this project, including both engines this was copied from.")
theme_mode = st.sidebar.radio("Theme Appearance", ["\u2600\ufe0f Light", "\U0001F30C Midnight"], index=0)
is_dark = (theme_mode == "\U0001F30C Midnight")

if is_dark:
    bg_color, card_bg, text_color, sub_text, border_color = "#12101C", "#1B1830", "#F1EEFB", "#A79FC9", "#2E2A4D"
else:
    bg_color, card_bg, text_color, sub_text, border_color = "#F7F6FC", "#FFFFFF", "#211C3B", "#7B739E", "#E4E1F5"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700&family=Fira+Code:wght@400;600&display=swap');
    html, body, [class*="css"]  {{ font-family: 'Outfit', sans-serif; }}
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .mono {{ font-family: 'Fira Code', monospace; }}
    .variant-header {{ font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 26px;
                        letter-spacing: -0.01em; margin-bottom: 2px; }}
    .top-banner {{ background-color: #3730A3; color: #E0E7FF; font-family: 'Fira Code', monospace;
                     font-weight: 600; font-size: 13px; letter-spacing: 0.04em; padding: 10px 16px;
                     border-radius: 10px; margin-bottom: 16px; border: 1px solid #6366F1; }}
    .metric-card {{ background-color: {card_bg}; border: 1px solid {border_color}; padding: 16px;
                     border-radius: 14px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    .metric-title {{ font-family: 'Fira Code', monospace; font-size: 11px; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.08em; color: {sub_text}; margin-bottom: 6px; }}
    .metric-value {{ font-family: 'Outfit', sans-serif; font-size: 22px; font-weight: 700; color: {text_color}; }}
    .metric-delta-pos {{ font-family: 'Fira Code', monospace; font-size: 12px; font-weight: 600;
                          color: #34D399; margin-top: 4px; }}
    .metric-delta-neg {{ font-family: 'Fira Code', monospace; font-size: 12px; font-weight: 600;
                          color: #FB7185; margin-top: 4px; }}
    .metric-delta-neu {{ font-family: 'Fira Code', monospace; font-size: 12px; font-weight: 600;
                          color: #FBBF24; margin-top: 4px; }}
    .position-card {{ background-color: {card_bg}; border: 2px solid var(--accent); padding: 18px;
                       border-radius: 16px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .position-badge {{ display:inline-block; font-family: 'Fira Code', monospace; font-weight:700;
                        font-size:11px; letter-spacing: 0.06em; text-transform: uppercase;
                        background-color: var(--accent); color:#0E0B26; padding:4px 12px; border-radius:999px;
                        margin-bottom:8px; }}
    .slot-badge {{ display:inline-block; font-family: 'Fira Code', monospace; font-weight:600;
                    font-size:11px; color: {sub_text}; margin-left: 8px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {card_bg}; border-radius: 8px; color: {sub_text};
                                     padding: 8px 16px; font-weight: 600; border: 1px solid {border_color}; }}
    .stTabs [aria-selected="true"] {{ background-color: var(--accent) !important; color: #0E0B26 !important; }}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Reads from:**\n`logs/om_progressive3pt_cap1_2ptonly_945cutoff_paper_portfolio.db`")
st.sidebar.markdown("**Entry rule:** progressive 3pt (index-based, 20% max retracement), cut off at 09:45.")
st.sidebar.markdown(f"**Concurrency:** up to {MAX_POSITIONS} simultaneous open positions for trail2pt.")
st.sidebar.markdown("**This copy:** ONLY trail2pt runs, entries stop after 09:45, 20s poll interval.")
st.sidebar.markdown(f"**Capital:** Rs {STARTING_CAPITAL:,.0f} independent pool.")
st.sidebar.markdown("---")
st.sidebar.caption("Start the engine separately:\n"
                    "`python step10_live_paper_om_progressive3pt_cap1_2ptONLY_945cutoff.py`")
st.sidebar.caption("Run THIS dashboard on its own port:\n"
                    "`streamlit run step14_dashboard_om_progressive3pt_cap1_2ptONLY_945cutoff.py --server.port 8523`")
st.sidebar.markdown("---")
st.sidebar.markdown("**Manual exit:** click \U0001F6D1 on an open position card below (closes ALL open "
                     "positions for trail2pt), or use the CLI: "
                     "`python manual_exit_progressive3pt_cap2.py trail2pt`. Writes to THIS engine's own "
                     "control folder -- cannot affect any other engine.")


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


def request_manual_exit(label):
    os.makedirs(CONTROL_DIR, exist_ok=True)
    flag_path = os.path.join(CONTROL_DIR, f"MANUAL_EXIT_{label}.flag")
    open(flag_path, "w").close()
    return flag_path


def manual_exit_flag_pending(label):
    all_flag = os.path.join(CONTROL_DIR, "MANUAL_EXIT_ALL.flag")
    return os.path.exists(os.path.join(CONTROL_DIR, f"MANUAL_EXIT_{label}.flag")) or os.path.exists(all_flag)


st.markdown('<div class="top-banner">\U0001F947 2pt-TRAIL-ONLY, 09:45-CUTOFF -- CONCURRENT CAP=1 -- same '
            'progressive-3pt entry as the no-cutoff engine, but entries stop after 09:45. PERMANENT. '
            'Fully separate DB/engine from every other engine.</div>',
            unsafe_allow_html=True)

st.title("\U0001F947 Progressive 3pt Entry, Cap=1, 09:45 CUTOFF (2pt trail ONLY)")
st.markdown(f"Up to {MAX_POSITIONS} simultaneous positions, identical PROGRESSIVE entry logic and "
            "20pt stop as the no-cutoff 2pt-only engine -- this copy stops taking new entries after 09:45. "
            "Real Dhan option-chain LTP, 20s poll interval.")

if not os.path.exists(DB_PATH):
    st.warning("No database found yet. Start "
               "`python step10_live_paper_om_progressive3pt_cap1_2ptONLY_945cutoff.py` to begin generating live data.")


# ================= HEAD-TO-HEAD SUMMARY (5s refresh) =================
@st.fragment(run_every=5)
def render_summary():
    current_time = datetime.datetime.now()
    today_str = str(current_time.date())

    summary_cols = st.columns(len(VARIANTS))
    for i, v in enumerate(VARIANTS):
        df_closed_all = query_db("SELECT * FROM paper_orders WHERE variant=? AND status='CLOSED' ORDER BY id DESC",
                                  (v["label"],))
        df_open = query_db("SELECT * FROM paper_orders WHERE variant=? AND status='OPEN' AND date=? ORDER BY id ASC",
                            (v["label"], today_str))

        capital_now = float(df_closed_all.iloc[0]["capital_after"]) if not df_closed_all.empty else STARTING_CAPITAL
        total_realized = capital_now - STARTING_CAPITAL
        df_today = df_closed_all[df_closed_all["date"] == today_str] if not df_closed_all.empty else pd.DataFrame()
        today_pnl = float(df_today["rupee_pnl"].sum()) if not df_today.empty else 0.0
        total_trades = len(df_closed_all)
        wins = int((df_closed_all["rupee_pnl"] > 0).sum()) if not df_closed_all.empty else 0
        win_pct = (wins / total_trades * 100) if total_trades else None

        with summary_cols[i]:
            st.markdown(f'<div class="variant-header" style="color:{v["accent"]}">{v["display"]}'
                        f'<span class="slot-badge">{len(df_open)}/{MAX_POSITIONS} slots open</span></div>',
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
                for _, pos in df_open.iterrows():
                    unreal_gross = unreal_charges = unreal_net = None
                    if pd.notna(pos.get("last_price")) and pd.notna(pos.get("entry_price")) and pd.notna(pos.get("lot_size")):
                        _entry, _last, _lot = float(pos["entry_price"]), float(pos["last_price"]), float(pos["lot_size"])
                        unreal_gross = (_last - _entry) * _lot
                        unreal_charges = compute_trade_cost(_entry, _last, _lot)
                        unreal_net = unreal_gross - unreal_charges
                    st.markdown(f"""
                    <div class="position-card" style="--accent: {v['accent']}">
                        <div class="position-badge" style="--accent: {v['accent']}">POSITION OPEN (id {pos.get('id','')})</div>
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
                if manual_exit_flag_pending(v["label"]):
                    st.warning("Manual exit flag pending -- will close ALL open positions for this variant on "
                               "the live engine's next poll cycle.")
                else:
                    if st.button(f"\U0001F6D1 Manual Exit ALL {v['display']} Positions", key=f"manual_exit_{v['label']}"):
                        request_manual_exit(v["label"])
                        st.success(f"Manual exit requested for {v['display']} -- closes every open position for "
                                   f"this variant on the live engine's next poll cycle (up to ~20s).")
            else:
                st.info(f"No open {v['display']} position ({len(df_open)}/{MAX_POSITIONS} slots) -- scanning "
                        f"for a clean 3pt move (entries only before 09:45).")

    st.markdown("---")
    all_open = any(
        not query_db("SELECT * FROM paper_orders WHERE variant=? AND status='OPEN' AND date=?",
                      (v["label"], today_str)).empty
        for v in VARIANTS
    )
    if all_open:
        if os.path.exists(os.path.join(CONTROL_DIR, "MANUAL_EXIT_ALL.flag")):
            st.warning("\U0001F6D1 Manual EXIT ALL flag pending -- closing every open position on the next "
                       "poll cycle.")
        elif st.button("\U0001F6D1 Manual EXIT ALL Open Positions", type="primary"):
            os.makedirs(CONTROL_DIR, exist_ok=True)
            open(os.path.join(CONTROL_DIR, "MANUAL_EXIT_ALL.flag"), "w").close()
            st.success("Manual EXIT ALL requested. Live engine will close every open position on its next "
                       "poll cycle (up to ~20s).")
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
                "exit_time": "Exit Time", "exit_price": "Exit Price",
                "exit_reason": "Exit Reason", "pnl_pts": "P&L (pts)",
                "gross_pnl": "Gross P&L (\u20b9)", "charges": "Brokerage+Charges (\u20b9)",
                "rupee_pnl": "Net P&L (\u20b9)",
            })
            show_cols = ["Direction", "Side", "Strike", "Entry Time", "Entry Price",
                         "Exit Time", "Exit Price", "Exit Reason", "P&L (pts)",
                         "Gross P&L (\u20b9)", "Brokerage+Charges (\u20b9)", "Net P&L (\u20b9)"]
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
                "exit_reason": "Exit Reason", "gross_pnl": "Gross P&L (\u20b9)",
                "charges": "Brokerage+Charges (\u20b9)", "rupee_pnl": "Net P&L (\u20b9)",
            })
            cols = ["Date", "Direction", "Side", "Strike", "Entry Time", "Exit Time", "Exit Reason",
                    "Gross P&L (\u20b9)", "Brokerage+Charges (\u20b9)", "Net P&L (\u20b9)"]
            cols = [c for c in cols if c in hist.columns]
            st.dataframe(hist[cols], use_container_width=True, hide_index=True)
        else:
            st.info(WAIT_MSG)

with tabs[-1]:
    st.subheader("Live Event Log (most recent first)")
    df_events = query_db("SELECT * FROM event_logs ORDER BY id DESC LIMIT 40")
    if not df_events.empty:
        for _, row in df_events.iterrows():
            accent = next((v["accent"] for v in VARIANTS if v["label"] == row.get("variant")), "#7B739E")
            st.markdown(f'<span class="mono" style="color:{accent}; font-weight:700;">[{row.get("variant","-")}]</span> '
                        f'**{row.get("timestamp","N/A")}** -- {row.get("event_message","N/A")}',
                        unsafe_allow_html=True)
    else:
        st.warning(WAIT_MSG)
