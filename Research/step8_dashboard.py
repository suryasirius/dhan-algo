"""
Module: step8_dashboard.py
Description: Institutional-Grade Real-Time Live Trading Terminal v7.0.
             Zero hardcoded/demo values. Every widget reads the latest row written by
             step9_live_engine.py (logs/paper_portfolio.db) or displays
             "Waiting for Live Data" if that row doesn't exist yet.

             This dashboard does NOT call the Dhan API itself - it is a pure read-only
             view over the DB so a 1-second UI refresh never hits API rate limits.
             Run the live engine as a separate process:
                 python step9_live_engine.py
             Then run this dashboard:
                 streamlit run step8_dashboard.py
"""

import os
import datetime
import sqlite3
import pandas as pd
import streamlit as st
from token_manager import validate_token, update_access_token, get_access_token, get_client_id

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Quant Trading Control Room | Real-Time Live Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

WAIT_MSG = "Waiting for Live Data"
DB_PATH = r"C:\dhan algo\Research\logs\paper_portfolio.db"

# --- Sidebar Theme Selector ---
st.sidebar.markdown("## 🎛️ Terminal Control")
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
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {card_bg}; border-radius: 6px; color: {sub_text};
                                     padding: 10px 20px; font-weight: 600; border: 1px solid {border_color}; }}
    .stTabs [aria-selected="true"] {{ background-color: #2563eb !important; color: #ffffff !important; }}
</style>
""", unsafe_allow_html=True)


# --- Data Access Helpers ---
def query_db(query: str) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame()


def latest_val(df: pd.DataFrame, column: str, fallback=WAIT_MSG, is_numeric=False, fmt="{:,.2f}"):
    if df is None or df.empty or column not in df.columns:
        return fallback
    val = df.iloc[0][column]
    if pd.isna(val):
        return fallback
    if is_numeric:
        try:
            return fmt.format(float(val))
        except Exception:
            return str(val)
    return str(val)


def card(col, title, value, delta="", d_type="neu"):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta-{d_type}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)


# --- Sidebar Controls ---
instrument = st.sidebar.selectbox("Active Instrument", ["Nifty 50 (13)", "BankNifty", "FinNifty"])
token_active = validate_token()
st.sidebar.markdown(f"{'🟢 Gateway Healthy' if token_active else '🔴 Token Expired'}")

with st.sidebar.expander("⚙️ Update Dhan Access Token"):
    new_client_id = st.text_input("Client ID", value=get_client_id())
    new_token_input = st.text_area("New Access Token", value="")
    if st.button("Update Token"):
        if new_token_input.strip():
            update_access_token(new_token_input)
            st.success("Token saved. The live engine will pick it up on its next cycle.")
            st.rerun()
        else:
            st.warning("Please paste a valid access token.")

st.title("AI Quant Trading Control Room v7.0 (Live Core)")
st.markdown("Zero static values. Every widget reads live_engine output or shows *Waiting for Live Data*.")

if not os.path.exists(DB_PATH):
    st.warning("No database found yet at `logs/paper_portfolio.db`. Start `python step9_live_engine.py` to begin generating live data.")


# ================= LIVE FRAGMENT (1s refresh, DB read-only) =================
@st.fragment(run_every=1)
def render_live_terminal():
    current_time = datetime.datetime.now()

    df_market = query_db("SELECT * FROM market_data ORDER BY id DESC LIMIT 1")
    df_dna = query_db("SELECT * FROM market_dna ORDER BY id DESC LIMIT 1")
    df_oi = query_db("SELECT * FROM oi_data ORDER BY id DESC LIMIT 1")
    df_strategy = query_db("SELECT * FROM strategy_engine ORDER BY id DESC LIMIT 1")
    df_events = query_db("SELECT * FROM event_logs ORDER BY id DESC LIMIT 15")
    df_oc = query_db("SELECT * FROM option_chain ORDER BY strike ASC")
    df_positions = query_db("SELECT * FROM paper_orders WHERE status='OPEN' ORDER BY id DESC LIMIT 1")
    df_closed = query_db("SELECT * FROM paper_orders WHERE status='CLOSED' ORDER BY id DESC LIMIT 20")

    # --- Heartbeat / data age from the last real market_data row ---
    last_tick_raw = latest_val(df_market, "timestamp", fallback=None)
    data_age, heartbeat_status = "Waiting", "🔴 Disconnected"
    if last_tick_raw:
        try:
            age_seconds = (current_time - pd.to_datetime(last_tick_raw)).total_seconds()
            data_age = f"{int(age_seconds)}s ago"
            if age_seconds < 20:
                heartbeat_status = "🟢 LIVE"
            elif age_seconds < 90:
                heartbeat_status = "🟡 Waiting for Feed"
            else:
                heartbeat_status = "🔴 Stale/Disconnected"
        except Exception:
            data_age = "Error parsing time"

    sys_cpu = f"{psutil.cpu_percent()}%" if HAS_PSUTIL else "psutil not installed"

    sb1, sb2, sb3, sb4, sb5, sb6 = st.columns(6)
    with sb1: st.markdown(f"**🕒 Time:** `{current_time.strftime('%H:%M:%S')}`")
    with sb2: st.markdown(f"**📡 Last Tick:** `{last_tick_raw if last_tick_raw else WAIT_MSG}`")
    with sb3: st.markdown(f"**⚡ Heartbeat:** {heartbeat_status}")
    with sb4: st.markdown(f"**⏳ Data Age:** `{data_age}`")
    with sb5: st.markdown(f"**🔑 Token:** `{'Active' if token_active else 'Expired'}`")
    with sb6: st.markdown(f"**💻 System CPU:** `{sys_cpu}`")

    st.markdown("---")

    # --- Live Market Summary ---
    spot = latest_val(df_market, "close", is_numeric=True)
    opn = latest_val(df_market, "open", is_numeric=True)
    high = latest_val(df_market, "high", is_numeric=True)
    low = latest_val(df_market, "low", is_numeric=True)
    vwap = latest_val(df_market, "vwap", is_numeric=True)
    atr = latest_val(df_market, "atr", is_numeric=True)
    pct_change = latest_val(df_market, "pct_change", is_numeric=True, fmt="{:,.2f}%")

    m_cols = st.columns(6)
    card(m_cols[0], "📈 Spot Price", spot, f"Change: {pct_change}")
    card(m_cols[1], "📊 Today's Open", opn, "Live Candle")
    card(m_cols[2], "📉 High / Low", f"{high} / {low}" if high != WAIT_MSG else WAIT_MSG, "Live Range")
    card(m_cols[3], "⚡ VWAP", vwap, "Live VWAP")
    card(m_cols[4], "📐 ATR", atr, "Live Volatility")
    card(m_cols[5], "⏳ Clock", current_time.strftime("%H:%M:%S"), f"{60 - current_time.second}s to next minute")

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧬 Market DNA & Live OI", "📊 Option Chain", "🤖 Strategy & Decision Engine",
        "💼 Live Position & PnL", "📜 Live Event Log", "🖥️ System Health"
    ])

    # ================= TAB 1: MARKET DNA & LIVE OI =================
    with tab1:
        st.subheader("🧬 Real-Time Market DNA Scorecard")
        trend = latest_val(df_dna, "trend")
        momentum = latest_val(df_dna, "momentum")
        vol_profile = latest_val(df_dna, "volume_profile")
        regime = latest_val(df_dna, "market_regime")
        overall_score = latest_val(df_dna, "overall_score", is_numeric=True, fmt="{:,.1f}%")
        smart_money = latest_val(df_dna, "smart_money")

        dna_c1, dna_c2, dna_c3, dna_c4 = st.columns(4)
        card(dna_c1, "Trend", trend)
        card(dna_c2, "Momentum", momentum)
        card(dna_c3, "Volume Profile", vol_profile)
        card(dna_c4, "Market Regime", regime)

        dna_c5, dna_c6 = st.columns(2)
        card(dna_c5, "🌟 Overall DNA Score", overall_score)
        card(dna_c6, "🧠 Smart Money Bias", smart_money)

        st.markdown("---")
        st.subheader("📊 Live OI Intelligence Feed")
        call_oi = latest_val(df_oi, "total_call_oi", is_numeric=True, fmt="{:,.0f}")
        put_oi = latest_val(df_oi, "total_put_oi", is_numeric=True, fmt="{:,.0f}")
        pcr = latest_val(df_oi, "pcr", is_numeric=True, fmt="{:,.2f}")
        oi_bias = latest_val(df_oi, "institutional_bias")
        atm_strike = latest_val(df_oi, "atm_strike", is_numeric=True, fmt="{:,.0f}")
        max_call_strike = latest_val(df_oi, "max_call_oi_strike", is_numeric=True, fmt="{:,.0f}")
        max_put_strike = latest_val(df_oi, "max_put_oi_strike", is_numeric=True, fmt="{:,.0f}")

        oi_c1, oi_c2, oi_c3, oi_c4 = st.columns(4)
        card(oi_c1, "📊 Total Call OI", call_oi)
        card(oi_c2, "📊 Total Put OI", put_oi)
        card(oi_c3, "⚖️ PCR", pcr)
        card(oi_c4, "🧠 Institutional Bias", oi_bias)

        oi_c5, oi_c6, oi_c7 = st.columns(3)
        card(oi_c5, "🎯 ATM Strike", atm_strike)
        card(oi_c6, "⚠ Max Call OI (Resistance)", max_call_strike)
        card(oi_c7, "⚠ Max Put OI (Support)", max_put_strike)

    # ================= TAB 2: LIVE OPTION CHAIN =================
    with tab2:
        st.subheader("📊 Real-Time Option Chain")
        if not df_oc.empty:
            st.dataframe(df_oc.drop(columns=["id", "timestamp"], errors="ignore"), use_container_width=True)
        else:
            st.warning(WAIT_MSG)

    # ================= TAB 3: STRATEGY & DECISION ENGINE =================
    with tab3:
        st.subheader("🎯 Real-Time Strategy Evaluator")
        curr_strat = latest_val(df_strategy, "current_strategy")
        prev_strat = latest_val(df_strategy, "previous_strategy")
        prob = latest_val(df_strategy, "probability", is_numeric=True, fmt="{:,.1f}%")
        conf = latest_val(df_strategy, "confidence")
        risk = latest_val(df_strategy, "risk_status")
        signal = latest_val(df_strategy, "signal")
        reason = latest_val(df_strategy, "reason")

        s_c1, s_c2, s_c3 = st.columns(3)
        card(s_c1, "🎯 Active Strategy", curr_strat, f"Prev: {prev_strat}")
        card(s_c2, "🧠 Probability / Confidence", prob, conf)
        card(s_c3, "⚡ Execution Signal", signal, f"Risk: {risk}",
             "pos" if signal == "BUY" else "neg" if signal == "NO_TRADE" else "neu")

        st.info(f"**Decision Reason:** {reason}")

    # ================= TAB 4: LIVE POSITION & PnL =================
    with tab4:
        st.subheader("💰 Live Paper Position & Risk Matrix")
        if not df_positions.empty:
            entry_price = latest_val(df_positions, "price", is_numeric=True)
            sl = latest_val(df_positions, "stop_loss", is_numeric=True)
            tgt = latest_val(df_positions, "target", is_numeric=True)
            qty = latest_val(df_positions, "quantity")
            entry_time_raw = latest_val(df_positions, "timestamp", fallback=None)

            unrealized_pnl = WAIT_MSG
            hold_time = WAIT_MSG
            if entry_time_raw:
                try:
                    entry_time = pd.to_datetime(entry_time_raw)
                    hold_time = str(current_time - entry_time).split(".")[0]
                except Exception:
                    pass
            if entry_price != WAIT_MSG and spot != WAIT_MSG:
                try:
                    unrealized_pnl = f"₹{(float(spot.replace(',', '')) - float(entry_price.replace(',', ''))) * float(qty):,.2f}"
                except Exception:
                    unrealized_pnl = WAIT_MSG

            p_c1, p_c2, p_c3, p_c4 = st.columns(4)
            card(p_c1, "💰 Unrealized PnL", unrealized_pnl)
            card(p_c2, "⏱ Holding Time", hold_time)
            card(p_c3, "🎯 Target", tgt)
            card(p_c4, "🛡 Stop Loss", sl)

            st.dataframe(df_positions.drop(columns=["id"], errors="ignore"), use_container_width=True)
        else:
            st.info("No active open positions. Waiting for the strategy engine to approve a trade.")

        st.markdown("---")
        st.subheader("📒 Recent Closed Trades")
        if not df_closed.empty:
            st.dataframe(df_closed.drop(columns=["id"], errors="ignore"), use_container_width=True)
        else:
            st.info(WAIT_MSG)

    # ================= TAB 5: EVENT LOG =================
    with tab5:
        st.subheader("📜 Live Event Log Stream")
        if not df_events.empty:
            for _, row in df_events.iterrows():
                st.markdown(f"* **{row.get('timestamp', 'N/A')}** — {row.get('event_message', 'N/A')}")
        else:
            st.warning(WAIT_MSG)

    # ================= TAB 6: SYSTEM HEALTH =================
    with tab6:
        st.subheader("🖥️ Infrastructure & Real-Time System Diagnostics")

        cpu_val = f"{psutil.cpu_percent()}%" if HAS_PSUTIL else "psutil not installed"
        ram_val = f"{psutil.virtual_memory().percent}%" if HAS_PSUTIL else "psutil not installed"
        db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0
        engine_status = "🟢 Running" if heartbeat_status == "🟢 LIVE" else ("🟡 Idle/Market Closed" if heartbeat_status != "🔴 Disconnected" else "🔴 Not Running")

        h_c1, h_c2, h_c3, h_c4 = st.columns(4)
        card(h_c1, "🖥️ CPU Usage", cpu_val)
        card(h_c2, "🖥️ RAM Usage", ram_val)
        card(h_c3, "🗄️ Database Size", f"{db_size_mb:.2f} MB")
        card(h_c4, "🔌 Live Engine (step9)", engine_status)

        h_c5, h_c6 = st.columns(2)
        card(h_c5, "🔑 Dhan Token", "Active" if token_active else "Expired", d_type="pos" if token_active else "neg")
        card(h_c6, "📡 Data Feed", heartbeat_status)


render_live_terminal()
