"""
PREMIUM DASHBOARD -- VWAP 20 EMA (trail2act10), PAPER
step14_dashboard_vwap20ema_trail2act10_v2.py

Same design as the progressive-3pt cap-1 dashboard, for the VWAP paper engine
step10_live_paper_vwap20ema_trail2act10.py
(DB: logs/vwap20ema_trail2act10_paper_portfolio.db). The older
step14_dashboard_vwap20ema_trail2act10.py (port 8520) is left untouched.

Colour rule: every P&L number is BLUE whether positive or negative (the sign
carries the direction); everything else neutral. No green/red.

The only things this dashboard writes: the engine's paper manual-exit flag
(Exit position now), the engine's KILL_SWITCH.flag (Kill switch: closes the
open position and blocks new entries until turned off), and the Dhan token
file via token_manager. PAPER TRADING ONLY.

Run (from the Research folder):
    python -m streamlit run step14_dashboard_vwap20ema_trail2act10_v2.py --server.port 8531
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta, timezone, time as dtime

import pandas as pd
import altair as alt
import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from trade_cost_model import compute_trade_cost  # same cost model the engine uses
import token_manager  # Research/token_manager.py -- same token file the paper engines read every cycle

DB_PATH = os.path.join(_THIS_DIR, "logs", "vwap20ema_trail2act10_paper_portfolio.db")
CONTROL_DIR = os.path.join(_THIS_DIR, "logs", "control_vwap20ema_trail2act10")
VARIANT = "vwap20ema_trail2act10"
STARTING_CAPITAL = 25_000.0
ENTRY_CUTOFF = dtime(15, 20)
MARKET_OPEN, MARKET_CLOSE = dtime(9, 15), dtime(15, 30)
IST = timezone(timedelta(hours=5, minutes=30))

BLUE, BLACK = "#1D4ED8", "#111111"
GREEN, RED, MUTED = "#15803D", "#B91C1C", "#6B7280"

st.set_page_config(page_title="VWAP 20 EMA · paper", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, p, span, div {font-family:'Inter', system-ui, sans-serif;}
.stApp {background:#F4F5F7;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1240px;}
#MainMenu, footer, header, [data-testid="stToolbar"] {visibility: hidden; height:0;}

/* header band -- deep navy, the only dark surface on the page */
.hdr {display:flex; justify-content:space-between; align-items:center; background:#0B1220;
      border-radius:20px; padding:22px 26px; margin-bottom:18px;
      box-shadow:0 10px 30px rgba(11,18,32,0.18);}
.eyebrow {font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:#7C8BA5; margin:0 0 6px;}
.ttl {font-size:26px; font-weight:600; color:#F8FAFC; margin:0; letter-spacing:-0.4px;}
.sub {font-size:13px; color:#94A3B8; margin:4px 0 0;}
.hdr .pill {display:inline-flex; align-items:center; gap:8px; font-size:13px; font-weight:500; padding:7px 14px;
      border-radius:999px; background:rgba(255,255,255,0.08) !important; color:#E2E8F0 !important;
      border:1px solid rgba(255,255,255,0.12);}
.dot {width:8px; height:8px; border-radius:50%; display:inline-block; box-shadow:0 0 0 4px rgba(59,130,246,0.18);}
.upd {font-size:12px; color:#7C8BA5; margin-top:8px; text-align:right;}

/* shared card surface: white, hairline border, soft layered shadow */
.hero, .k, .card {background:#FFFFFF; border:1px solid #ECEEF2;
      box-shadow:0 1px 2px rgba(16,24,40,0.04), 0 6px 18px rgba(16,24,40,0.05);}
.hero {border-radius:18px; padding:20px 22px; position:relative; overflow:hidden;}
.hero:before {content:""; position:absolute; left:0; top:0; height:3px; width:100%; background:#1D4ED8; opacity:.85;}
.hl {font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:#64748B; margin:0; font-weight:600;}
.hv {font-family:'JetBrains Mono', ui-monospace, monospace; font-size:38px; font-weight:600;
     margin:10px 0 6px; letter-spacing:-1px; font-variant-numeric: tabular-nums;}
.hs {font-size:12px; color:#64748B; margin:0;}
.k {border-radius:16px; padding:15px 18px;}
.kl {font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:#94A3B8; margin:0; font-weight:600;}
.kv {font-family:'JetBrains Mono', ui-monospace, monospace; font-size:22px; font-weight:600; color:#0F172A;
     margin:6px 0 0; font-variant-numeric: tabular-nums;}
.card {border-radius:18px; padding:20px 22px;}
.ch {font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:#64748B; margin:0 0 14px; font-weight:600;}
.badge {font-size:11px; font-weight:500; padding:3px 10px; border-radius:999px; letter-spacing:.02em;}
.row {display:grid; grid-template-columns:60px 40px 1fr 74px 96px; gap:8px; padding:10px 0;
      border-bottom:1px solid #F1F3F6; font-size:13px; color:#1E293B; align-items:center;
      font-family:'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums;}
.stat {display:flex; justify-content:space-between; font-size:13px; padding:8px 0; color:#0F172A;
       border-bottom:1px solid #F1F3F6;}
.stat span:first-child {color:#64748B;}
.stat span:last-child {font-family:'JetBrains Mono', ui-monospace, monospace; font-weight:500;}
div[data-testid="stVegaLiteChart"] {background:#FFFFFF; border:1px solid #ECEEF2; border-radius:18px;
      padding:14px 10px 6px; box-shadow:0 1px 2px rgba(16,24,40,0.04), 0 6px 18px rgba(16,24,40,0.05);}
.stButton > button {border-radius:999px; border:1px solid #CBD5E1; color:#0F172A; font-weight:500; background:#FFFFFF;}
.stButton > button:hover {border-color:#0B1220; color:#0B1220;}

/* ---- polish layer ---- */
.stApp {background:radial-gradient(1200px 500px at 50% -120px, #E9EEF8 0%, #F4F5F7 60%);}
.hero, .k, .card {transition: transform .18s ease, box-shadow .18s ease;}
.hero:hover, .k:hover {transform: translateY(-2px);
      box-shadow:0 2px 4px rgba(16,24,40,0.05), 0 14px 30px rgba(16,24,40,0.08);}
.hdr {background:linear-gradient(135deg, #0B1220 0%, #111C33 60%, #13213D 100%);}
@keyframes pulse {0% {box-shadow:0 0 0 0 rgba(59,130,246,.55);} 70% {box-shadow:0 0 0 9px rgba(59,130,246,0);}
                  100% {box-shadow:0 0 0 0 rgba(59,130,246,0);}}
.dot.live {animation: pulse 1.8s infinite;}
/* bordered Streamlit containers become the same premium card */
div[data-testid="stVerticalBlockBorderWrapper"] {background:#FFFFFF; border:1px solid #ECEEF2 !important;
      border-radius:18px !important; box-shadow:0 1px 2px rgba(16,24,40,0.04), 0 6px 18px rgba(16,24,40,0.05);}
div[data-testid="stVegaLiteChart"] {background:transparent !important; border:none !important;
      box-shadow:none !important; padding:0 !important;}
.empty {height:220px; border:1px dashed #DCE2EA; border-radius:14px; display:flex; flex-direction:column;
        align-items:center; justify-content:center; color:#94A3B8; font-size:13px; gap:4px;
        background:repeating-linear-gradient(0deg, transparent, transparent 43px, #F6F8FB 43px, #F6F8FB 44px);}
.empty b {color:#64748B; font-weight:500; font-size:14px;}
.welcome {display:flex; align-items:center; justify-content:space-between; gap:16px; background:#FFFFFF;
          border:1px solid #ECEEF2; border-radius:16px; padding:14px 20px; margin:-4px 0 16px; font-size:13px; color:#475569;}
.welcome code {background:#F1F5F9; color:#0F172A; padding:4px 10px; border-radius:8px; font-size:12px;}
.ctl-state {display:flex; gap:10px; flex-wrap:wrap; margin-top:4px;}
.chip {display:inline-flex; align-items:center; gap:7px; font-size:12px; color:#334155; background:#F6F8FB;
       border:1px solid #EDF1F6; padding:6px 12px; border-radius:999px;}
.chip i {width:7px; height:7px; border-radius:50%; display:inline-block;}
.st-key-btn_kill button, .st-key-btn_kill_yes button {background:#0B1220 !important; color:#F8FAFC !important;
      border-color:#0B1220 !important;}
.st-key-btn_kill button:hover, .st-key-btn_kill_yes button:hover {background:#1E293B !important;}
div[data-testid="stExpander"] details {background:#FFFFFF; border:1px solid #ECEEF2; border-radius:18px;
      box-shadow:0 1px 2px rgba(16,24,40,0.04), 0 6px 18px rgba(16,24,40,0.05);}
div[data-testid="stExpander"] summary p {font-size:13px !important; color:#334155; font-weight:500;}
.foot {text-align:center; font-size:11px; letter-spacing:.08em; color:#A3AEC0; margin-top:22px;}

/* Streamlit styles every <p> inside markdown; these selectors + !important make ours win */
.stMarkdown p.ttl {font-size:26px !important; line-height:1.2 !important;}
.stMarkdown p.sub {font-size:13px !important;}
.stMarkdown p.eyebrow, .stMarkdown p.hl, .stMarkdown p.kl, .stMarkdown p.ch {font-size:11px !important;}
.stMarkdown p.hv {font-size:38px !important; line-height:1.15 !important;}
.stMarkdown p.kv {font-size:22px !important; line-height:1.25 !important;}
.stMarkdown p.hs {font-size:12px !important;}
.stMarkdown p {margin-bottom:0;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data ----
def _clean_ts(series):
    return pd.to_datetime(series.astype(str).str.slice(0, 19), errors="coerce")


def load():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(), None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            orders = pd.read_sql("SELECT * FROM paper_orders WHERE variant=? ORDER BY id", conn, params=(VARIANT,))
            last_evt = conn.execute("SELECT MAX(timestamp) FROM event_logs").fetchone()[0]
    except Exception:
        return pd.DataFrame(), None
    if not orders.empty:
        orders["entry_dt"] = _clean_ts(orders["entry_time"])
        orders["exit_dt"] = _clean_ts(orders["exit_time"])
    return orders, last_evt


def money(v, sign=True):
    if v is None or pd.isna(v):
        return "–"
    s = f"₹{abs(v):,.0f}"
    if not sign:
        return s
    return ("+" if v >= 0 else "−") + s


def hero_color(v):
    return BLUE  # always blue -- sign shows direction, no green/red bias


def max_drawdown(pnls):
    if len(pnls) == 0:
        return 0.0
    eq = STARTING_CAPITAL + pd.Series(pnls).cumsum()
    peak = eq.cummax().clip(lower=STARTING_CAPITAL)
    return float((eq - peak).min())


def request_exit():
    os.makedirs(CONTROL_DIR, exist_ok=True)
    open(os.path.join(CONTROL_DIR, f"MANUAL_EXIT_{VARIANT}.flag"), "w").close()


def exit_pending():
    return (os.path.exists(os.path.join(CONTROL_DIR, f"MANUAL_EXIT_{VARIANT}.flag"))
            or os.path.exists(os.path.join(CONTROL_DIR, "MANUAL_EXIT_ALL.flag")))


def kill_path():
    return os.path.join(CONTROL_DIR, "KILL_SWITCH.flag")


def kill_on():
    return os.path.exists(kill_path())


def set_kill(on):
    os.makedirs(CONTROL_DIR, exist_ok=True)
    if on:
        open(kill_path(), "w").close()
    elif os.path.exists(kill_path()):
        os.remove(kill_path())


def style_chart(ch):
    """Quiet, premium axis styling shared by every chart."""
    return (ch.configure_view(strokeWidth=0)
              .configure_axis(labelColor="#94A3B8", labelFont="Inter", labelFontSize=11,
                              gridColor="#F1F3F6", domain=False, tickColor="#F1F3F6"))


# ------------------------------------------------------------ dashboard ---
@st.fragment(run_every=10)
def dashboard():
    now = datetime.now(IST)
    today = now.date()
    orders, last_evt = load()

    closed = orders[orders["status"] == "CLOSED"].copy() if not orders.empty else pd.DataFrame()
    open_pos = orders[orders["status"] == "OPEN"].copy() if not orders.empty else pd.DataFrame()
    if not closed.empty:
        closed["gross"] = closed["pnl_pts"] * closed["lot_size"]
        closed["charges"] = closed["gross"] - closed["rupee_pnl"]
    today_closed = closed[closed["date"] == str(today)] if not closed.empty else pd.DataFrame()

    # running P&L on the open position (from the last premium the engine saved)
    running, pos = None, None
    if not open_pos.empty:
        pos = open_pos.iloc[-1]
        ltp = pos["last_price"] if pd.notna(pos["last_price"]) else pos["entry_price"]
        lot = int(pos["lot_size"])
        running = (ltp - pos["entry_price"]) * lot - compute_trade_cost(pos["entry_price"], ltp, lot)

    today_net = float(today_closed["rupee_pnl"].sum()) if not today_closed.empty else 0.0
    all_net = float(closed["rupee_pnl"].sum()) if not closed.empty else 0.0
    capital = STARTING_CAPITAL + all_net

    # -------- header + status
    evt_age = None
    if last_evt:
        try:
            evt_age = (now.replace(tzinfo=None) - datetime.fromisoformat(last_evt[:19])).total_seconds() / 60
        except Exception:
            pass
    market_on = MARKET_OPEN <= now.time() <= MARKET_CLOSE and now.weekday() < 5
    if kill_on():
        status, bg, fg, dot = "Kill switch on", "#F1F5F9", "#0F172A", "#0F172A"
    elif pos is not None:
        status, bg, fg, dot = "In trade", "#F1F5F9", "#0F172A", BLUE
    elif not market_on:
        status, bg, fg, dot = "Market closed", "#F1F5F9", MUTED, "#94A3B8"
    elif now.time() >= ENTRY_CUTOFF:
        status, bg, fg, dot = "Entries closed · after 15:20", "#F1F5F9", MUTED, "#94A3B8"
    else:
        status, bg, fg, dot = "Scanning", "#F1F5F9", "#0F172A", BLUE
    if market_on and (evt_age is None or evt_age > 30) and pos is None and now.time() < ENTRY_CUTOFF:
        status, bg, fg, dot = "Engine not running?", "#FEF3C7", "#92400E", "#D97706"
    upd = f"Last engine update {last_evt[11:19]}" if last_evt else "No engine data yet"
    live = "live" if status in ("In trade", "Scanning") else ""

    st.markdown(f"""
    <div class="hdr">
      <div><p class="eyebrow">Paper trading · NIFTY options</p><p class="ttl">VWAP 20 EMA</p>
        <p class="sub">1 position · max 5 trades/day · 10pt stop · staged trail from +10 · 40pt VWAP filter · exit 15:20</p></div>
      <div><span class="pill" style="background:{bg};color:{fg}"><span class="dot {live}" style="background:{dot}"></span>{status}</span>
        <div class="upd">{upd} · refreshed {now:%H:%M:%S}</div></div>
    </div>""", unsafe_allow_html=True)

    if orders.empty:
        st.markdown('<div class="welcome"><span>No trades recorded yet. Numbers fill in as the engine trades.</span>'
                    '<code>python step10_live_paper_vwap20ema_trail2act10.py</code></div>',
                    unsafe_allow_html=True)

    # -------- hero row (blue >= 0, black < 0)
    pos_sub = "No open position"
    if pos is not None:
        pts = (pos["last_price"] if pd.notna(pos["last_price"]) else pos["entry_price"]) - pos["entry_price"]
        pos_sub = f"{pos['side']} {int(pos['strike'])} · {pts:+.2f} pts · after charges"
    wins_today = int((today_closed["rupee_pnl"] > 0).sum()) if not today_closed.empty else 0
    h1, h2, h3 = st.columns(3)
    for col, label, val, sub in [
        (h1, "Running P&L", running, pos_sub),
        (h2, "Today net", today_net, f"{len(today_closed)} closed trades · {wins_today} wins"),
        (h3, "All-time net", all_net, f"{len(closed)} trades · since start ₹25,000"),
    ]:
        shown = "₹0" if val is None else money(val)
        col.markdown(f"""<div class="hero"><p class="hl">{label}</p>
            <p class="hv" style="color:{hero_color(val)}">{shown}</p><p class="hs">{sub}</p></div>""",
                     unsafe_allow_html=True)

    # -------- small KPIs
    st.write("")
    wins = closed[closed["rupee_pnl"] > 0]["rupee_pnl"] if not closed.empty else pd.Series(dtype=float)
    losses = closed[closed["rupee_pnl"] <= 0]["rupee_pnl"] if not closed.empty else pd.Series(dtype=float)
    win_rate = f"{100 * len(wins) / len(closed):.0f}%" if len(closed) else "–"
    pf = f"{wins.sum() / -losses.sum():.2f}" if len(losses) and losses.sum() < 0 else ("–" if not len(wins) else "∞")
    mdd = max_drawdown(closed["rupee_pnl"].tolist()) if not closed.empty else 0.0
    for col, label, val in zip(st.columns(4), ["Capital", "Win rate", "Profit factor", "Max drawdown"],
                               [money(capital, sign=False), win_rate, pf, money(mdd) if mdd < 0 else "₹0"]):
        col.markdown(f'<div class="k"><p class="kl">{label}</p><p class="kv">{val}</p></div>', unsafe_allow_html=True)

    # -------- controls: manual exit + kill switch
    st.write("")
    ks = kill_on()
    ks_txt = ("<b style='color:#0F172A'>ON</b> · no new entries" if ks else "off")
    ex_txt = ("requested, closing next cycle" if (pos is not None and exit_pending())
              else ("ready" if pos is not None else "no open position"))
    ctl = st.container(border=True)
    sp, ca, cb = ctl.columns([3, 1.1, 1.1], vertical_alignment="center")
    sp.markdown(f"""<p class="ch" style="margin:0 0 8px">Controls</p><div class="ctl-state">
        <span class="chip"><i style="background:{'#1D4ED8' if pos is not None else '#CBD5E1'}"></i>Manual exit · {ex_txt}</span>
        <span class="chip"><i style="background:{'#0B1220' if ks else '#CBD5E1'}"></i>Kill switch · {ks_txt}</span>
        </div>""", unsafe_allow_html=True)
    with ca:
        if st.button("Exit position now", use_container_width=True, key="btn_exit"):
            if pos is None:
                st.toast("No open position to exit.")
            else:
                request_exit()  # only written when a position is open, so it can't close a future trade
                st.toast("Exit requested (paper). Engine closes it on its next cycle.")
    with cb:
        if ks:
            if st.button("Turn kill switch off", use_container_width=True, key="btn_kill_off"):
                set_kill(False)
                st.toast("Kill switch off. Entries allowed again until 15:20.")
                st.rerun(scope="fragment")
        elif st.session_state.get("confirm_kill"):
            k1, k2 = st.columns(2)
            if k1.button("Confirm", type="primary", use_container_width=True, key="btn_kill_yes"):
                set_kill(True)
                st.session_state["confirm_kill"] = False
                st.toast("Kill switch ON. Open position will close; no new entries.")
                st.rerun(scope="fragment")
            if k2.button("Cancel", use_container_width=True, key="btn_kill_no"):
                st.session_state["confirm_kill"] = False
                st.rerun(scope="fragment")
        elif st.button("Kill switch", use_container_width=True, key="btn_kill"):
            st.session_state["confirm_kill"] = True
            st.rerun(scope="fragment")

    # -------- open position
    st.write("")
    if pos is not None:
        ltp = pos["last_price"] if pd.notna(pos["last_price"]) else pos["entry_price"]
        lot = int(pos["lot_size"])
        gross = (ltp - pos["entry_price"]) * lot
        chg = compute_trade_cost(pos["entry_price"], ltp, lot)
        stop, peak = pos["current_stop"], pos["peak_price"]
        lo = min(pos["initial_stop"], ltp)
        hi = max(peak, ltp) if peak > lo else lo + 1
        fill = max(0, min(100, (ltp - lo) / (hi - lo) * 100))
        stop_at = max(0, min(100, (stop - lo) / (hi - lo) * 100))
        mins = (now.replace(tzinfo=None) - pos["entry_dt"]).total_seconds() / 60 if pd.notna(pos["entry_dt"]) else 0
        trail = "Trail active · staged 2–4pt" if pos["trailing_active"] else "Initial stop · trail from +10"
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
            <div><span class="ch" style="margin:0 10px 0 0">Open position</span><span class="badge" style="background:#F1F5F9;color:#0F172A;border:1px solid #E2E8F0">{pos['side']} {int(pos['strike'])} · 1 lot</span>
              <span style="font-size:13px;color:#94A3B8;margin-left:8px">opened {pos['entry_dt']:%H:%M} · {mins:.0f} min</span></div>
            <span class="badge" style="background:#F1F5F9;color:#334155">{trail}</span></div>
          <div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr 1fr;gap:12px;align-items:end">
            <div><p class="kl">Live premium</p><p style="font-family:'JetBrains Mono',monospace;font-size:32px !important;font-weight:600;margin:6px 0 0;letter-spacing:-1px;color:{BLUE}">{ltp:.2f}</p></div>
            <div><p class="kl">Entry</p><p class="kv" style="font-size:17px !important">{pos['entry_price']:.2f}</p></div>
            <div><p class="kl">Peak</p><p class="kv" style="font-size:17px !important">{peak:.2f}</p></div>
            <div><p class="kl">Stop</p><p class="kv" style="font-size:17px !important">{stop:.2f}</p></div>
            <div><p class="kl">Gross / charges</p><p class="kv" style="font-size:17px !important;color:{BLUE}">{money(gross)} / −₹{chg:,.0f}</p></div>
          </div>
          <div style="position:relative;height:7px;background:#F1F5F9;border-radius:999px;margin:16px 0 4px">
            <div style="position:absolute;left:0;width:{fill:.0f}%;height:7px;background:{BLUE};border-radius:999px"></div>
            <div style="position:absolute;left:{stop_at:.0f}%;top:-4px;width:2px;height:15px;background:#0F172A"></div></div>
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#94A3B8">
            <span>Initial stop {pos['initial_stop']:.2f}</span><span>dark line = current stop</span></div>
        </div>""", unsafe_allow_html=True)

    # -------- charts
    st.write("")
    c1, c2 = st.columns([1.4, 1])
    with c1.container(border=True):
        st.markdown('<p class="ch">Equity curve · all days, net of charges</p>', unsafe_allow_html=True)
        if not closed.empty:
            eq = closed[["exit_dt", "rupee_pnl"]].dropna().sort_values("exit_dt")
            eq["capital"] = STARTING_CAPITAL + eq["rupee_pnl"].cumsum()
            eq = pd.concat([pd.DataFrame({"exit_dt": [eq["exit_dt"].iloc[0] - timedelta(minutes=1)],
                                          "capital": [STARTING_CAPITAL]}), eq[["exit_dt", "capital"]]])
            eq["n"] = range(len(eq))
            chart = alt.Chart(eq).mark_area(
                line={"color": BLUE, "strokeWidth": 2.5},
                color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                                   stops=[alt.GradientStop(color="rgba(29,78,216,0.0)", offset=0),
                                          alt.GradientStop(color="rgba(29,78,216,0.22)", offset=1)]),
                interpolate="monotone").encode(
                x=alt.X("n:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
                y=alt.Y("capital:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(format=",.0f", grid=True)),
                tooltip=[alt.Tooltip("exit_dt:T", title="Time", format="%d %b %H:%M"),
                         alt.Tooltip("capital:Q", title="Capital", format=",.0f")])
            base = alt.Chart(pd.DataFrame({"y": [STARTING_CAPITAL]})).mark_rule(strokeDash=[4, 4], color="#CBD5E1").encode(y="y:Q")
            st.altair_chart(style_chart((base + chart).properties(height=240)), use_container_width=True)
        else:
            st.markdown('<div class="empty"><b>Your equity curve starts here</b>Appears after the first closed trade</div>',
                        unsafe_allow_html=True)
    with c2.container(border=True):
        st.markdown('<p class="ch">Daily net P&L · last 15 days</p>', unsafe_allow_html=True)
        if not closed.empty:
            daily = closed.groupby("date", as_index=False)["rupee_pnl"].sum().tail(15)
            daily["day"] = pd.to_datetime(daily["date"]).dt.strftime("%d %b")
            bars = alt.Chart(daily).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                             cornerRadiusBottomLeft=4, cornerRadiusBottomRight=4, size=18).encode(
                x=alt.X("day:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("rupee_pnl:Q", title=None, axis=alt.Axis(format=",.0f")),
                color=alt.value(BLUE),
                tooltip=[alt.Tooltip("day:N", title="Day"), alt.Tooltip("rupee_pnl:Q", title="Net", format=",.0f")])
            st.altair_chart(style_chart(bars.properties(height=240)), use_container_width=True)
        else:
            st.markdown('<div class="empty"><b>No trading days yet</b>One bar per day, net of charges</div>',
                        unsafe_allow_html=True)

    # -------- today's trades + stats
    st.write("")
    t1, t2 = st.columns([1.6, 1])
    with t1:
        rows = ""
        for _, r in (today_closed.sort_values("entry_dt").iterrows() if not today_closed.empty else []):
            reason = str(r["exit_reason"])
            rb = ("#F1F5F9", "#334155")  # neutral badge for every exit reason
            col = BLUE
            rows += (f'<div class="row"><span>{r["entry_dt"]:%H:%M}</span><span>{r["side"]}</span>'
                     f'<span>{r["entry_price"]:.2f} → {r["exit_price"]:.2f}</span>'
                     f'<span class="badge" style="background:{rb[0]};color:{rb[1]};text-align:center">{reason.replace("_", " ").title()}</span>'
                     f'<span style="text-align:right;color:{col};font-weight:600">{money(r["rupee_pnl"])}</span></div>')
        if not rows:
            rows = '<p style="font-size:13px;color:#94A3B8;margin:6px 0">No closed trades today yet.</p>'
        st.markdown(f"""<div class="card"><p class="ch">Today's trades</p>
          <div class="row" style="color:#94A3B8;font-size:11px"><span>Time</span><span>Side</span><span>Entry → exit</span><span>Exit</span><span style="text-align:right">Net</span></div>
          {rows}</div>""", unsafe_allow_html=True)
    with t2:
        if not closed.empty:
            reasons = closed["exit_reason"].fillna("OTHER").value_counts(normalize=True) * 100
            trail_p = reasons.get("TRAIL", 0)
            stop_p = reasons.get("STOP", 0) + reasons.get("HARD_STOP", 0)
            other_p = max(0, 100 - trail_p - stop_p)
            stats = [("Trades", f"{len(closed)}"),
                     ("Avg win", money(wins.mean()) if len(wins) else "–"),
                     ("Avg loss", money(losses.mean()) if len(losses) else "–"),
                     ("Best trade", money(closed["rupee_pnl"].max())),
                     ("Worst trade", money(closed["rupee_pnl"].min())),
                     ("Total charges", money(closed["charges"].sum(), sign=False))]
        else:
            trail_p = stop_p = other_p = 0
            stats = [("Trades", "0")]
        def _num_color(v):
            return BLUE if v.startswith(("+", "−", "₹")) else "#0F172A"  # money values blue, counts neutral
        stat_html = "".join(f'<div class="stat"><span>{k}</span><span style="color:{_num_color(v)}">{v}</span></div>'
                            for k, v in stats)
        st.markdown(f"""<div class="card"><p class="ch">Stats · all days</p>{stat_html}
          <p class="ch" style="margin:14px 0 6px">Exit reasons</p>
          <div style="display:flex;height:8px;border-radius:999px;overflow:hidden;background:#F1F5F9">
            <div style="width:{trail_p:.0f}%;background:{BLUE}"></div><div style="width:{stop_p:.0f}%;background:#93C5FD"></div>
            <div style="width:{other_p:.0f}%;background:#CBD5E1"></div></div>
          <div style="display:flex;gap:12px;font-size:11px;color:#94A3B8;margin-top:6px">
            <span>Trail {trail_p:.0f}%</span><span>Stop {stop_p:.0f}%</span><span>Time / EOD / other {other_p:.0f}%</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="foot">PAPER TRADING ONLY  ·  AUTO-REFRESH 10S  ·  '
                'LOGS/VWAP20EMA_TRAIL2ACT10_PAPER_PORTFOLIO.DB</p>', unsafe_allow_html=True)


dashboard()


# ------------------------------------------------------- Dhan token box ---
# Lives OUTSIDE the auto-refreshing fragment so typing isn't interrupted.
# Saves to Research/logs/dhan_token.txt (+ dhan_client_id.txt) via
# token_manager -- the paper engines pick it up on their next cycle
# (DhanConnector.refresh_credentials), no restart needed. The token is
# never displayed back; only its last 4 characters and when it was saved.
def _token_status():
    path = token_manager.TOKEN_FILE
    tok = token_manager.get_access_token() or ""
    if not tok:
        return "No token saved yet", None
    age_h = None
    if os.path.exists(path):
        age_h = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).total_seconds() / 3600
    saved = f"saved {age_h:.0f}h ago" if age_h is not None else "from environment variable"
    return f"Token ending ••••{tok[-4:]} · {saved}", age_h


st.write("")
_status, _age = _token_status()
with st.expander(f"Dhan access token  ·  {_status}", expanded=False):
    if _age is not None and _age >= 24:
        st.warning("This token is over 24 hours old. Dhan tokens usually expire daily, so paste a fresh one.")
    with st.form("token_form", clear_on_submit=True):
        new_tok = st.text_input("New access token", type="password", placeholder="Paste the token from Dhan web")
        new_cid = st.text_input("Client ID (optional, only if it changed)",
                                placeholder=token_manager.get_client_id() or "")
        if st.form_submit_button("Save token"):
            tok = (new_tok or "").strip().replace("\n", "").replace("\r", "")
            if len(tok) <= 20:
                st.error("That doesn't look like a full token. Copy the whole token and try again.")
            else:
                token_manager.update_access_token(tok)
                cid = (new_cid or "").strip()
                if cid:
                    os.makedirs(os.path.dirname(token_manager.CLIENT_ID_FILE), exist_ok=True)
                    with open(token_manager.CLIENT_ID_FILE, "w") as fh:
                        fh.write(cid)
                st.success(f"Saved token ending ••••{tok[-4:]}. The paper engine uses it from its next cycle.")
    st.caption("Stored in Research/logs/dhan_token.txt on this computer only. Applies to the paper engines "
               "in the Research folder.")
