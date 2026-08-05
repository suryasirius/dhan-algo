"""
Module: step11_dashboard_best2.py
Description: Live monitoring dashboard for step10_live_paper_best2.py ONLY.

             Separate from step8_dashboard.py on purpose - step8 reads
             logs/paper_portfolio.db (step9's full DNA/regime/ML engine),
             this one reads logs/best2_paper_portfolio.db (the 2-strategy,
             no-ML-filter paper engine). Zero hardcoded/demo values - every
             widget reads the latest row(s) written by step10, or shows
             "Waiting for Live Data" if nothing has been written yet.

             Pure read-only view over the DB - does not call the Dhan API
             and does not import step10, so a fast UI refresh never touches
             the live engine or the broker.

Run the live engine as a separate process:
    python step10_live_paper_best2.py
Then run this dashboard:
    streamlit run step11_dashboard_best2.py
"""

import os
import datetime
import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Best-2 Paper Trading | Live Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

WAIT_MSG = "Waiting for Live Data"
DB_PATH = r"C:\dhan algo\Research\logs\best2_paper_portfolio.db"
STARTING_CAPITAL = 25_000.0

# --- Sidebar ---
st.sidebar.markdown("## 🎯 Best-2 Monitor")
theme_mode = st.sidebar.radio("Theme Appearance", ["🌞 Light Mode", "🌙 Dark Mode"], index=1)
is_dark = (theme_mode == "🌙 Dark Mode")

if is_dark:
    bg_color, card_bg, text_color, sub_text, border_color = "#1E293B", "#273449", "#FFFFFF", "#94A3B8", "#334155"
else:
    bg_color, card_bg, text_color, sub_text, border_color = "#FFFFFF", "#F7F9FC", "#111827", "#6B7280", "#E5E7EB"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .metric-card {{ background-color: {card_bg}; border: 1px solid {border_color}; padding: 18px; border-radius: 10px;
                     box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 10px; }}
    .metric-title {{ font-size: 14px; font-weight: 600; color: {sub_text}; margin-bottom: 6px; }}
    .metric-value {{ font-size: 22px; font-weight: 700; color: {text_color}; }}
    .metric-delta-pos {{ font-size: 13px; font-weight: 600; color: #10B981; margin-top: 4px; }}
    .metric-delta-neg {{ font-size: 13px; font-weight: 600; color: #EF4444; margin-top: 4px; }}
    .metric-delta-neu {{ font-size: 13px; font-weight: 600; color: #F59E0B; margin-top: 4px; }}
    .position-card {{ background-color: {card_bg}; border: 2px solid #2563eb; padding: 20px; border-radius: 12px;
                       margin-bottom: 14px; }}
    .reason-box {{ background-color: {card_bg}; border-left: 4px solid #2563eb; padding: 12px 16px;
                    border-radius: 6px; font-size: 14px; color: {text_color}; margin-top: 8px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {card_bg}; border-radius: 6px; color: {sub_text};
                                     padding: 10px 20px; font-weight: 600; border: 1px solid {border_color}; }}
    .stTabs [aria-selected="true"] {{ background-color: #2563eb !important; color: #ffffff !important; }}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Reads from:**\n`logs/best2_paper_portfolio.db`")
st.sidebar.markdown("**Strategies:**\n1. Opening-Drive Momentum (sl20/trail1)\n2. Six Concepts RBI/GBI relaxed (sl8/trail4)")
st.sidebar.markdown("**Capital sharing:** one ₹25,000 pool. open_momentum has priority on conflicts.")
st.sidebar.markdown("---")
st.sidebar.caption("Start the engine separately:\n`python step10_live_paper_best2.py`")


# --- Data Access Helpers ---
def query_db(query: str) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame()


def fmt_money(val, fallback=WAIT_MSG):
    if val is None or pd.isna(val):
        return fallback
    return f"₹{float(val):,.2f}"


def fmt_num(val, fallback=WAIT_MSG, decimals=2):
    if val is None or pd.isna(val):
        return fallback
    return f"{float(val):,.{decimals}f}"


def card(col, title, value, delta="", d_type="neu"):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta-{d_type}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)


st.title("🎯 Best-2 Paper Trading — Live Monitor")
st.markdown("Opening-Drive Momentum + Six Concepts (RBI/GBI relaxed) — no ML filter, both strategies trade unconditionally on their own rules.")

if not os.path.exists(DB_PATH):
    st.warning("No database found yet at `logs/best2_paper_portfolio.db`. Start `python step10_live_paper_best2.py` to begin generating live data.")


# ================= LIVE FRAGMENT (5s refresh, DB read-only) =================
@st.fragment(run_every=5)
def render_best2_terminal():
    current_time = datetime.datetime.now()
    today_str = str(current_time.date())

    df_open = query_db("SELECT * FROM paper_orders WHERE status='OPEN' ORDER BY id DESC")
    df_closed_all = query_db("SELECT * FROM paper_orders WHERE status='CLOSED' ORDER BY id DESC")
    df_events = query_db("SELECT * FROM event_logs ORDER BY id DESC LIMIT 25")

    # --- Heartbeat from the latest event log ---
    last_event_time = None
    heartbeat_status, data_age = "🔴 Disconnected", "Waiting"
    if not df_events.empty:
        try:
            last_event_time = pd.to_datetime(df_events.iloc[0]["timestamp"])
            age_seconds = (current_time - last_event_time).total_seconds()
            data_age = f"{int(age_seconds)}s ago"
            if age_seconds < 60:
                heartbeat_status = "🟢 LIVE"
            elif age_seconds < 300:
                heartbeat_status = "🟡 Idle / Between Cycles"
            else:
                heartbeat_status = "🔴 Stale/Not Running"
        except Exception:
            pass

    # --- Capital / realized P&L (all-time) ---
    if not df_closed_all.empty and "capital_after" in df_closed_all.columns and df_closed_all["capital_after"].notna().any():
        capital_now = float(df_closed_all.iloc[0]["capital_after"])
    else:
        capital_now = STARTING_CAPITAL
    total_realized_pnl = capital_now - STARTING_CAPITAL

    # --- Today's realized P&L ---
    df_closed_today = df_closed_all[df_closed_all["date"] == today_str] if not df_closed_all.empty else pd.DataFrame()
    today_realized_pnl = float(df_closed_today["rupee_pnl"].sum()) if not df_closed_today.empty else 0.0
    today_trade_count = len(df_closed_today)
    today_wins = int((df_closed_today["rupee_pnl"] > 0).sum()) if not df_closed_today.empty else 0

    # --- Unrealized P&L from any open position(s) ---
    total_unrealized = 0.0
    if not df_open.empty:
        for _, row in df_open.iterrows():
            if pd.notna(row.get("last_price")) and pd.notna(row.get("entry_price")) and pd.notna(row.get("lot_size")):
                total_unrealized += (float(row["last_price"]) - float(row["entry_price"])) * float(row["lot_size"])

    # --- All-time stats ---
    total_trades = len(df_closed_all)
    total_wins = int((df_closed_all["rupee_pnl"] > 0).sum()) if not df_closed_all.empty else 0
    win_pct = (total_wins / total_trades * 100) if total_trades > 0 else None

    sb1, sb2, sb3, sb4 = st.columns(4)
    with sb1: st.markdown(f"**🕒 Time:** `{current_time.strftime('%H:%M:%S')}`")
    with sb2: st.markdown(f"**📡 Last Event:** `{last_event_time.strftime('%H:%M:%S') if last_event_time is not None else WAIT_MSG}`")
    with sb3: st.markdown(f"**⚡ Engine Heartbeat:** {heartbeat_status}")
    with sb4: st.markdown(f"**⏳ Data Age:** `{data_age}`")

    st.markdown("---")

    m_cols = st.columns(6)
    card(m_cols[0], "💰 Capital", fmt_money(capital_now),
         f"{'+' if total_realized_pnl >= 0 else ''}{fmt_money(total_realized_pnl)} all-time",
         "pos" if total_realized_pnl >= 0 else "neg")
    card(m_cols[1], "📅 Today's Realized P&L", fmt_money(today_realized_pnl) if today_trade_count > 0 else "₹0.00 (no closed trades yet)",
         f"{today_wins}/{today_trade_count} wins today", "pos" if today_realized_pnl >= 0 else ("neu" if today_trade_count == 0 else "neg"))
    card(m_cols[2], "🔓 Unrealized P&L (open)", fmt_money(total_unrealized) if not df_open.empty else "No open position",
         f"{len(df_open)} position(s) open", "pos" if total_unrealized >= 0 else ("neu" if df_open.empty else "neg"))
    card(m_cols[3], "🏆 All-Time Win Rate", f"{win_pct:.1f}%" if win_pct is not None else WAIT_MSG, f"{total_wins}/{total_trades} trades")
    card(m_cols[4], "📊 Total Trades (all-time)", str(total_trades) if total_trades else WAIT_MSG)
    card(m_cols[5], "🎯 Net Return on ₹25k", f"{(total_realized_pnl / STARTING_CAPITAL * 100):+.1f}%" if total_trades else WAIT_MSG,
         "", "pos" if total_realized_pnl >= 0 else "neu")

    st.markdown("---")

    # ================= OPEN POSITION(S) =================
    st.subheader("🔴 Live Open Position(s)")
    if df_open.empty:
        st.info("No open position right now. Both strategies are scanning for entries.")
    else:
        for _, pos in df_open.iterrows():
            strat_label = "🚀 Opening-Drive Momentum" if pos["strategy"] == "open_momentum" else "🔄 Six Concepts (RBI/GBI relaxed)"
            unreal = None
            if pd.notna(pos.get("last_price")) and pd.notna(pos.get("entry_price")) and pd.notna(pos.get("lot_size")):
                unreal = (float(pos["last_price"]) - float(pos["entry_price"])) * float(pos["lot_size"])
            st.markdown(f"""
            <div class="position-card">
                <div style="font-size:18px; font-weight:700;">{strat_label} — {pos.get('side', '')} ({pos.get('direction', '')})</div>
            </div>
            """, unsafe_allow_html=True)

            p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
            card(p_c1, "Entry Price", fmt_money(pos.get("entry_price")), f"at {pos.get('entry_time', '')}")
            card(p_c2, "Last Observed Price", fmt_money(pos.get("last_price")))
            card(p_c3, "🛡️ Current Stop-Loss", fmt_money(pos.get("current_stop")),
                 f"Initial: {fmt_money(pos.get('initial_stop'))}")
            card(p_c4, "📈 Peak Price", fmt_money(pos.get("peak_price")))
            card(p_c5, "Unrealized P&L", fmt_money(unreal) if unreal is not None else WAIT_MSG,
                 "", "pos" if (unreal or 0) >= 0 else "neg")

            st.markdown(f"""<div class="reason-box">
                <b>🎯 Target:</b> {pos.get('target_note', WAIT_MSG)}<br>
                <b>💡 Reason for entry:</b> {pos.get('reason', WAIT_MSG)}
            </div>""", unsafe_allow_html=True)
            st.markdown("")

    st.markdown("---")

    # ================= TABS: TODAY'S TRADES / 6-DAY LOG / EVENT LOG =================
    tab1, tab2, tab3 = st.tabs(["📅 Today's Trades", "🗓️ Last 6 Trading Days", "📜 Live Event Log"])

    with tab1:
        st.subheader("📅 Today's Closed Trades")
        if not df_closed_today.empty:
            show_cols = ["strategy", "direction", "side", "entry_time", "entry_price", "exit_time",
                         "exit_price", "exit_reason", "pnl_pts", "rupee_pnl", "reason"]
            show_cols = [c for c in show_cols if c in df_closed_today.columns]
            st.dataframe(df_closed_today[show_cols], use_container_width=True)
        else:
            st.info("No closed trades yet today.")
        if not df_open.empty:
            st.caption(f"Plus {len(df_open)} still-open position(s) shown above.")

    with tab2:
        st.subheader("🗓️ Last 6 Trading Days — Daily Summary")
        if not df_closed_all.empty:
            daily = (df_closed_all.groupby("date")
                     .agg(trades=("id", "count"),
                          wins=("rupee_pnl", lambda s: int((s > 0).sum())),
                          net_pnl=("rupee_pnl", "sum"))
                     .reset_index()
                     .sort_values("date", ascending=False)
                     .head(6))
            daily["win_rate"] = (daily["wins"] / daily["trades"] * 100).round(1).astype(str) + "%"
            daily["net_pnl"] = daily["net_pnl"].round(2)
            daily = daily.rename(columns={"date": "Date", "trades": "Trades", "wins": "Wins",
                                           "win_rate": "Win Rate", "net_pnl": "Net P&L (₹)"})
            st.dataframe(daily[["Date", "Trades", "Wins", "Win Rate", "Net P&L (₹)"]],
                         use_container_width=True, hide_index=True)

            with st.expander("See individual trades for these days"):
                recent_dates = daily["Date"].tolist()
                detail = df_closed_all[df_closed_all["date"].isin(recent_dates)][
                    ["date", "strategy", "direction", "side", "entry_time", "entry_price",
                     "exit_time", "exit_price", "exit_reason", "rupee_pnl"]
                ].sort_values(["date", "entry_time"], ascending=[False, False])
                st.dataframe(detail, use_container_width=True, hide_index=True)
        else:
            st.info(WAIT_MSG)

    with tab3:
        st.subheader("📜 Live Event Log Stream")
        if not df_events.empty:
            for _, row in df_events.iterrows():
                st.markdown(f"* **{row.get('timestamp', 'N/A')}** — {row.get('event_message', 'N/A')}")
        else:
            st.warning(WAIT_MSG)


render_best2_terminal()
