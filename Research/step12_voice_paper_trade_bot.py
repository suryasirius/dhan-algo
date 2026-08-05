"""
Module: step12_voice_paper_trade_bot.py
Description: Voice-controlled MANUAL paper trading bot for Nifty options.

             You speak a trade like:
                 "Nifty 24200 CE take trade at market price, exit when
                  Nifty reach 24350"
             The bot transcribes it, shows/speaks back exactly what it
             understood, and waits for you to say "confirm" before placing
             anything -- this matters because misheard strike prices are a
             real risk with voice input, so nothing gets logged as a trade
             without you explicitly approving what it heard.

             Once confirmed, it buys the option at the live market premium
             (paper only, no real order sent anywhere) and then does ONE of:
               - auto-exits when Nifty's SPOT price reaches a level you gave
                 ("exit when nifty reach 24350")
               - auto-exits when the OPTION PREMIUM reaches a target you gave
                 ("exit at 150") or gains N points ("exit after 20 points")
               - waits for you to just say "exit" whenever you want out, if
                 you didn't give any exit condition at entry
             Whichever happens first also closes it, even if you set a
             condition -- you can always say "exit" to override it manually.

             Only one voice-placed position open at a time, by design --
             keeps the voice interface simple and unambiguous.

SEPARATE FROM EVERYTHING ELSE: does not touch step9_live_engine.py,
step10_live_paper_best2.py, or their databases. Logs to its own
logs/voice_paper_trades.db so it can never interfere with the automated
strategies already running.

SETUP (run on the machine with a working microphone):
    pip install SpeechRecognition pyaudio --break-system-packages
    (Windows pyaudio install issues: if `pip install pyaudio` fails, try
    `pip install pipwin` then `pipwin install pyaudio`.)
    Optional, for spoken replies instead of just printed text:
    pip install pyttsx3 --break-system-packages

Run:
    cd "C:\\dhan algo\\Research"
    python step12_voice_paper_trade_bot.py

Uses Google's free web speech API for transcription (needs internet; no API
key required for light/personal use, but it is a third-party service your
audio is sent to for transcription -- mentioned here so that's not a
surprise).
"""

import os
import re
import sys
import time
import sqlite3
import logging
import threading
from datetime import datetime, timezone, timedelta

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from step1_config_and_connector import PlatformConfig, DhanConnector

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

# Two possible microphone backends:
#   - pyaudio (speech_recognition's default) -- often fails to install on
#     Windows with a brand-new Python version, since there's no precompiled
#     wheel yet and building from source needs the PortAudio C headers.
#   - sounddevice -- ships its own bundled audio engine, no compiling
#     required, so this is the recommended/default path on Windows.
try:
    import pyaudio  # noqa: F401
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    import sounddevice as sd
    import numpy as np
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

HAS_MIC_BACKEND = HAS_SR and (HAS_PYAUDIO or HAS_SOUNDDEVICE)

SD_SAMPLE_RATE = 16000
SD_SILENCE_RMS_THRESHOLD = 500.0   # int16 RMS below this counts as silence
SD_SILENCE_HANGOVER_SEC = 1.0      # stop recording after this much continuous silence
SD_CHUNK_SEC = 0.1

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Step12_VoicePaperTrade")

IST = timezone(timedelta(hours=5, minutes=30))
STARTING_CAPITAL = 25_000.0
BROKERAGE_PER_TRADE = 40.0
DEFAULT_LOT_SIZE = 75   # overridden by the real NSE lot-size lookup when available (see _lot_size)
EXIT_POLL_SECONDS = 5


# ======================================================================
# Command parsing -- pure functions, fully testable without a mic or a
# live connection (this is where a misheard-number bug would live, so
# it's kept separate and self-checked hard).
# ======================================================================

NUMBER_RE = re.compile(r"(\d{4,6})")
CE_PE_RE = re.compile(r"\b(ce|call|pe|put)\b", re.I)
EXIT_LEVEL_RE = re.compile(r"exit\s+when\s+nifty\s+(?:reach(?:es)?|hits?|touch(?:es)?)\s+(\d{4,6})", re.I)
EXIT_PREMIUM_RE = re.compile(r"exit\s+at\s+(\d+(?:\.\d+)?)", re.I)
EXIT_POINTS_RE = re.compile(r"exit\s+after\s+(\d+(?:\.\d+)?)\s*points?", re.I)
MARKET_PRICE_RE = re.compile(r"market\s+price", re.I)
EXIT_NOW_RE = re.compile(r"^\s*exit\s*$", re.I)
CONFIRM_RE = re.compile(r"\b(confirm|yes|correct|go\s*ahead)\b", re.I)
CANCEL_RE = re.compile(r"\b(cancel|no|wrong|stop)\b", re.I)


def parse_entry_command(text):
    """Turns a spoken/transcribed sentence into a structured entry order, or
    returns None if it doesn't look like a valid entry command (missing
    strike, missing CE/PE, or it's actually a bare 'exit' command)."""
    if not text:
        return None
    t = text.strip()
    if EXIT_NOW_RE.match(t):
        return None

    m_strike = NUMBER_RE.search(t)
    m_side = CE_PE_RE.search(t)
    if not m_strike or not m_side:
        return None

    strike = int(m_strike.group(1))
    side = "CE" if m_side.group(1).lower() in ("ce", "call") else "PE"

    exit_condition = None
    m_level = EXIT_LEVEL_RE.search(t)
    m_prem = EXIT_PREMIUM_RE.search(t)
    m_pts = EXIT_POINTS_RE.search(t)
    if m_level:
        exit_condition = {"type": "SPOT_LEVEL", "value": float(m_level.group(1))}
    elif m_prem:
        exit_condition = {"type": "PREMIUM_TARGET", "value": float(m_prem.group(1))}
    elif m_pts:
        exit_condition = {"type": "PREMIUM_POINTS", "value": float(m_pts.group(1))}

    return {
        "strike": strike,
        "side": side,
        "entry_type": "MARKET",
        "exit_condition": exit_condition,
        "raw_text": t,
    }


def is_manual_exit_command(text):
    return bool(text and EXIT_NOW_RE.match(text.strip()))


def is_confirmation(text):
    return bool(text and CONFIRM_RE.search(text))


def is_cancellation(text):
    return bool(text and CANCEL_RE.search(text))


def describe_order(order):
    ex = order["exit_condition"]
    if ex is None:
        ex_desc = "no auto-exit set -- you'll say 'exit' when you want out"
    elif ex["type"] == "SPOT_LEVEL":
        ex_desc = f"auto-exit when Nifty spot reaches {ex['value']:.0f}"
    elif ex["type"] == "PREMIUM_TARGET":
        ex_desc = f"auto-exit when the option premium reaches {ex['value']:.2f}"
    else:
        ex_desc = f"auto-exit after {ex['value']:.0f} points of profit"
    return f"BUY NIFTY {order['strike']} {order['side']} at market price. {ex_desc}."


def compute_rms(int16_chunk):
    """Root-mean-square loudness of a chunk of int16 audio samples -- the
    energy measure the sounddevice backend uses to decide when speech starts
    and stops. Plain-Python implementation (no numpy dependency) so this is
    testable in any environment, including one without audio libraries
    installed at all."""
    if int16_chunk is None or len(int16_chunk) == 0:
        return 0.0
    values = [float(x) for x in int16_chunk]
    return (sum(x * x for x in values) / len(values)) ** 0.5


def check_exit_trigger(exit_condition, entry_spot, entry_price, current_spot, current_premium):
    """Pure decision function: given the exit condition set at entry and the
    latest live readings, should we exit now? Kept separate from any network
    call so it can be unit-tested with fixed numbers, no mocking required."""
    if exit_condition is None:
        return False, None
    etype, value = exit_condition["type"], exit_condition["value"]
    if etype == "SPOT_LEVEL":
        if current_spot is None:
            return False, None
        # "reach" a level from either side -- works whether the level is
        # above or below the spot price at the moment the trade was placed.
        if (entry_spot <= value <= current_spot) or (entry_spot >= value >= current_spot):
            return True, f"Nifty spot reached {value:.0f}"
        return False, None
    if etype == "PREMIUM_TARGET":
        if current_premium is None:
            return False, None
        if current_premium >= value:
            return True, f"Premium reached {value:.2f}"
        return False, None
    if etype == "PREMIUM_POINTS":
        if current_premium is None:
            return False, None
        if (current_premium - entry_price) >= value:
            return True, f"+{value:.0f} points reached"
        return False, None
    return False, None


# ======================================================================
# The bot itself (needs a live Dhan connection + microphone to actually run)
# ======================================================================

class VoicePaperTradeBot:
    def __init__(self, config: PlatformConfig = None):
        self.config = config or PlatformConfig()
        self.connector = DhanConnector(self.config)
        self.db_path = os.path.join(self.config.log_dir, "voice_paper_trades.db")
        self._init_db()

        self.capital = STARTING_CAPITAL
        self._load_capital_from_db()

        self.open_position = None
        self._expiry_cache, self._expiry_cache_date = None, None
        self._stop = threading.Event()

        self.mic_backend = "pyaudio" if HAS_PYAUDIO else ("sounddevice" if HAS_SOUNDDEVICE else None)
        self.recognizer = sr.Recognizer() if HAS_SR else None
        self.mic = sr.Microphone() if (HAS_SR and self.mic_backend == "pyaudio") else None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS voice_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, strike REAL, side TEXT,
            entry_time TEXT, exit_time TEXT,
            entry_price REAL, exit_price REAL, exit_reason TEXT,
            entry_spot REAL, exit_condition_type TEXT, exit_condition_value REAL,
            pnl_pts REAL, rupee_pnl REAL, lot_size INTEGER,
            capital_after REAL, status TEXT, raw_command TEXT
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, event_message TEXT
        )''')
        conn.commit()
        conn.close()

    def _load_capital_from_db(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            row = conn.execute(
                "SELECT capital_after FROM voice_orders WHERE status='CLOSED' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row and row[0] is not None:
                self.capital = float(row[0])
        except Exception:
            pass

    def log_event(self, message: str):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("INSERT INTO event_logs (timestamp, event_message) VALUES (?, ?)",
                         (datetime.now().isoformat(), message))
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to write event log")
        logger.info(message)

    def _record_open(self, order, entry_time, entry_price, entry_spot, lot):
        cond = order["exit_condition"]
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''INSERT INTO voice_orders
            (date, strike, side, entry_time, entry_price, entry_spot,
             exit_condition_type, exit_condition_value, lot_size, status, capital_after, raw_command)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)''',
            (str(entry_time.date()), order["strike"], order["side"], entry_time.isoformat(),
             entry_price, entry_spot, cond["type"] if cond else None, cond["value"] if cond else None,
             lot, self.capital, order["raw_text"]))
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id

    def _record_close(self, row_id, exit_time, exit_price, exit_reason, pnl_pts, rupee_pnl, lot):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute('''UPDATE voice_orders SET exit_time=?, exit_price=?, exit_reason=?,
            pnl_pts=?, rupee_pnl=?, lot_size=?, capital_after=?, status='CLOSED' WHERE id=?''',
            (exit_time.isoformat(), exit_price, exit_reason, pnl_pts, rupee_pnl, lot, self.capital, row_id))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Voice I/O
    # ------------------------------------------------------------------
    def speak(self, text: str):
        print(f"[BOT] {text}")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass  # pyttsx3 not installed / no audio output device -- printed text is enough to function

    def listen_once(self, timeout=6, phrase_time_limit=8):
        if not HAS_SR:
            raise RuntimeError("SpeechRecognition is not installed. Run: "
                                "pip install SpeechRecognition --break-system-packages")
        if self.mic_backend == "pyaudio":
            return self._listen_once_pyaudio(timeout, phrase_time_limit)
        if self.mic_backend == "sounddevice":
            return self._listen_once_sounddevice(timeout, phrase_time_limit)
        raise RuntimeError("No working microphone backend installed. Run: "
                            "pip install sounddevice numpy --break-system-packages")

    def _listen_once_pyaudio(self, timeout, phrase_time_limit):
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except sr.WaitTimeoutError:
                return None
        return self._transcribe(audio)

    def _listen_once_sounddevice(self, timeout, phrase_time_limit):
        """Records via sounddevice (no compiled dependency, unlike pyaudio) using
        a simple energy-based start/stop: waits up to `timeout` seconds for speech
        to begin, then keeps recording until SD_SILENCE_HANGOVER_SEC of continuous
        silence follows, capped at `phrase_time_limit` seconds of actual speech."""
        chunk_frames = int(SD_SAMPLE_RATE * SD_CHUNK_SEC)
        frames = []
        started, silence_time, waited = False, 0.0, 0.0

        with sd.InputStream(samplerate=SD_SAMPLE_RATE, channels=1, dtype="int16") as stream:
            while True:
                chunk, _ = stream.read(chunk_frames)
                chunk = chunk.reshape(-1)
                rms = compute_rms(chunk)

                if not started:
                    if rms > SD_SILENCE_RMS_THRESHOLD:
                        started = True
                        frames.append(chunk)
                    else:
                        waited += SD_CHUNK_SEC
                        if waited >= timeout:
                            return None
                    continue

                frames.append(chunk)
                silence_time = 0.0 if rms > SD_SILENCE_RMS_THRESHOLD else (silence_time + SD_CHUNK_SEC)
                speech_elapsed = sum(len(f) for f in frames) / SD_SAMPLE_RATE
                if silence_time >= SD_SILENCE_HANGOVER_SEC or speech_elapsed >= phrase_time_limit:
                    break

        if not frames:
            return None
        audio_bytes = np.concatenate(frames).tobytes()
        audio_data = sr.AudioData(audio_bytes, SD_SAMPLE_RATE, 2)
        return self._transcribe(audio_data)

    def _transcribe(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data)
            print(f"[HEARD] {text}")
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            self.log_event(f"Speech recognition service error: {e}")
            return None

    # ------------------------------------------------------------------
    # Live data
    # ------------------------------------------------------------------
    def _get_expiry(self):
        today = datetime.now(IST).date()
        if self._expiry_cache and self._expiry_cache_date == today:
            return self._expiry_cache
        expiries = self.connector.fetch_expiry_list()
        if not expiries:
            return None
        self._expiry_cache, self._expiry_cache_date = sorted(expiries)[0], today
        return self._expiry_cache

    def fetch_live_spot(self):
        df = self.connector.fetch_historical_intraday(self.config.nifty_security_id, self.config.nifty_exchange_segment)
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])

    def fetch_live_premium(self, strike, side):
        expiry = self._get_expiry()
        if not expiry:
            return None
        chain = self.connector.fetch_option_chain(expiry)
        if not chain or "oc" not in chain:
            return None
        oc = chain["oc"]
        for strike_str, legs in oc.items():
            try:
                if abs(float(strike_str) - float(strike)) < 0.01:
                    leg = (legs.get("ce") if side == "CE" else legs.get("pe")) or {}
                    lp = leg.get("last_price")
                    return float(lp) if lp is not None else None
            except (TypeError, ValueError):
                continue
        return None

    def _lot_size(self, on_date):
        try:
            import nifty_open_momentum_stoptrail_sweep as om
            return om.lot_size_for_date(on_date)
        except Exception:
            return DEFAULT_LOT_SIZE

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------
    def execute_entry(self, order):
        premium = self.fetch_live_premium(order["strike"], order["side"])
        if premium is None:
            self.speak("Could not fetch a live premium for that strike right now. Trade not placed.")
            return False
        spot = self.fetch_live_spot()
        entry_time = datetime.now(IST)
        lot = self._lot_size(entry_time.date())

        row_id = self._record_open(order, entry_time, premium, spot, lot)
        self.open_position = {
            "row_id": row_id, "strike": order["strike"], "side": order["side"],
            "entry_time": entry_time, "entry_price": premium, "entry_spot": spot,
            "exit_condition": order["exit_condition"], "lot": lot,
        }
        self.speak(f"Trade placed. Bought {order['side']} {order['strike']} at {premium:.2f}.")
        self.log_event(f"OPENED {order['side']} {order['strike']} @ {premium:.2f} (spot {spot}) "
                        f"from voice command: \"{order['raw_text']}\"")
        return True

    def execute_exit(self, reason, premium=None):
        pos = self.open_position
        if pos is None:
            return
        if premium is None:
            premium = self.fetch_live_premium(pos["strike"], pos["side"])
        if premium is None:
            self.speak("Could not fetch a live premium to exit right now -- will retry next cycle.")
            return
        exit_time = datetime.now(IST)
        pnl_pts = premium - pos["entry_price"]
        rupee_pnl = pnl_pts * pos["lot"] - BROKERAGE_PER_TRADE
        self.capital += rupee_pnl
        self._record_close(pos["row_id"], exit_time, premium, reason, pnl_pts, rupee_pnl, pos["lot"])
        self.speak(f"Exited {pos['side']} {pos['strike']} at {premium:.2f}. "
                    f"P and L {rupee_pnl:+.2f} rupees. {reason}.")
        self.log_event(f"CLOSED {pos['side']} {pos['strike']} @ {premium:.2f} ({reason}) | "
                        f"PnL Rs {rupee_pnl:,.2f} | capital now Rs {self.capital:,.2f}")
        self.open_position = None

    def check_exit_conditions(self):
        pos = self.open_position
        if pos is None:
            return
        premium = self.fetch_live_premium(pos["strike"], pos["side"])
        spot = self.fetch_live_spot()
        triggered, reason = check_exit_trigger(
            pos["exit_condition"], pos["entry_spot"], pos["entry_price"], spot, premium
        )
        if triggered:
            self.execute_exit(reason, premium)

    # ------------------------------------------------------------------
    # Main loops (voice listener runs on its own thread; exit-condition
    # polling runs on the main thread so both work concurrently)
    # ------------------------------------------------------------------
    def _listen_loop(self):
        while not self._stop.is_set():
            text = self.listen_once()
            if text is None:
                continue

            if self.open_position is not None:
                if is_manual_exit_command(text):
                    self.execute_exit("Manual voice exit command")
                # ignore anything else while a position is open -- one at a time, by design
                continue

            order = parse_entry_command(text)
            if order is None:
                continue
            self.speak(f"I heard: {describe_order(order)} Say confirm to place this trade, or cancel.")
            confirm_text = self.listen_once(timeout=8, phrase_time_limit=5)
            if is_confirmation(confirm_text):
                self.execute_entry(order)
            elif is_cancellation(confirm_text):
                self.speak("Cancelled. Nothing was placed.")
            else:
                self.speak("Didn't hear a clear confirm or cancel, so this trade was discarded for safety.")

    def run_forever(self):
        if not HAS_MIC_BACKEND:
            raise RuntimeError("No working voice setup. Run: "
                                "pip install SpeechRecognition sounddevice numpy --break-system-packages")
        self.log_event(f"Voice paper trading bot started (mic backend: {self.mic_backend}). "
                        f"Capital: Rs {self.capital:,.2f}")
        self.speak("Voice paper trading bot ready. Say a trade, for example: "
                    "Nifty 24200 CE take trade at market price.")
        listener = threading.Thread(target=self._listen_loop, daemon=True)
        listener.start()
        try:
            while not self._stop.is_set():
                if self.open_position is not None:
                    self.check_exit_conditions()
                time.sleep(EXIT_POLL_SECONDS)
        except KeyboardInterrupt:
            self._stop.set()
            self.speak("Bot stopped.")


# ======================================================================
# Self-checks (synthetic / mocked, no mic or network needed)
# ======================================================================

def run_self_checks():
    print("Running step12_voice_paper_trade_bot self-checks (synthetic data)...")

    # --- parse_entry_command: the exact example from the user's own phrasing ---
    o = parse_entry_command("nifty 24200 ce take trade at market price exit when nifty reach 24350")
    assert o is not None and o["strike"] == 24200 and o["side"] == "CE"
    assert o["exit_condition"] == {"type": "SPOT_LEVEL", "value": 24350.0}
    print("  parse_entry_command: spot-level exit phrasing ok")

    o2 = parse_entry_command("nifty 24500 put take trade at market price exit at 150")
    assert o2["side"] == "PE" and o2["exit_condition"] == {"type": "PREMIUM_TARGET", "value": 150.0}
    print("  parse_entry_command: premium-target exit phrasing ok")

    o3 = parse_entry_command("nifty 24000 call take trade exit after 20 points")
    assert o3["side"] == "CE" and o3["exit_condition"] == {"type": "PREMIUM_POINTS", "value": 20.0}
    print("  parse_entry_command: points-target exit phrasing ok")

    o4 = parse_entry_command("nifty 23900 ce take trade at market price")
    assert o4["exit_condition"] is None
    print("  parse_entry_command: no exit condition -> manual-exit-only order ok")

    assert parse_entry_command("exit") is None
    assert parse_entry_command("just some noise") is None
    assert parse_entry_command("nifty ce take trade") is None, "missing strike number must not parse"
    assert parse_entry_command("nifty 24200 take trade") is None, "missing CE/PE must not parse"
    print("  parse_entry_command: rejects incomplete/non-entry text ok")

    # --- exit/confirm/cancel word detection ---
    assert is_manual_exit_command("exit") and is_manual_exit_command("  Exit  ")
    assert not is_manual_exit_command("exit at 150")  # this is an entry-time condition, not a live exit command
    assert is_confirmation("yes confirm that") and not is_confirmation("cancel")
    assert is_cancellation("no cancel it") and not is_cancellation("yes")
    print("  manual-exit / confirm / cancel word detection ok")

    # --- compute_rms: the sounddevice backend's silence/speech detector ---
    assert compute_rms([]) == 0.0
    assert compute_rms([0, 0, 0]) == 0.0
    assert abs(compute_rms([1000, -1000, 1000, -1000]) - 1000.0) < 1e-6
    assert compute_rms([2000, 2000]) > compute_rms([200, 200]), "louder audio must score higher RMS"
    print("  compute_rms ok")

    # --- describe_order: human-readable readback (the safety confirmation step) ---
    desc = describe_order(o)
    assert "24200" in desc and "CE" in desc and "24350" in desc
    print("  describe_order readback ok")

    # --- check_exit_trigger: the actual money-moving decision, tested directly ---
    # spot-level exit, target ABOVE entry spot
    trig, _ = check_exit_trigger({"type": "SPOT_LEVEL", "value": 24350}, entry_spot=24280,
                                  entry_price=100, current_spot=24300, current_premium=110)
    assert trig is False, "should not trigger before spot reaches the level"
    trig, reason = check_exit_trigger({"type": "SPOT_LEVEL", "value": 24350}, entry_spot=24280,
                                       entry_price=100, current_spot=24360, current_premium=130)
    assert trig is True and "24350" in reason
    # spot-level exit, target BELOW entry spot (downside level, e.g. a put)
    trig, _ = check_exit_trigger({"type": "SPOT_LEVEL", "value": 24200}, entry_spot=24280,
                                  entry_price=100, current_spot=24350, current_premium=90)
    assert trig is False, "moving away from a downside target must not trigger"
    trig, _ = check_exit_trigger({"type": "SPOT_LEVEL", "value": 24200}, entry_spot=24280,
                                  entry_price=100, current_spot=24190, current_premium=140)
    assert trig is True
    # premium target
    trig, _ = check_exit_trigger({"type": "PREMIUM_TARGET", "value": 150}, entry_spot=24280,
                                  entry_price=100, current_spot=24300, current_premium=149)
    assert trig is False
    trig, _ = check_exit_trigger({"type": "PREMIUM_TARGET", "value": 150}, entry_spot=24280,
                                  entry_price=100, current_spot=24300, current_premium=151)
    assert trig is True
    # points target
    trig, _ = check_exit_trigger({"type": "PREMIUM_POINTS", "value": 20}, entry_spot=24280,
                                  entry_price=100, current_spot=24300, current_premium=119)
    assert trig is False
    trig, _ = check_exit_trigger({"type": "PREMIUM_POINTS", "value": 20}, entry_spot=24280,
                                  entry_price=100, current_spot=24300, current_premium=120)
    assert trig is True
    # no condition at all -- must never auto-trigger, only manual "exit" closes it
    trig, _ = check_exit_trigger(None, entry_spot=24280, entry_price=100, current_spot=99999, current_premium=99999)
    assert trig is False
    # missing live data this cycle -- must not crash or false-trigger
    trig, _ = check_exit_trigger({"type": "PREMIUM_TARGET", "value": 150}, entry_spot=24280,
                                  entry_price=100, current_spot=None, current_premium=None)
    assert trig is False
    print("  check_exit_trigger: all exit-condition types + missing-data safety ok")

    print("All step12_voice_paper_trade_bot self-checks passed.\n")


if __name__ == "__main__":
    run_self_checks()
    if not HAS_MIC_BACKEND:
        print("No working voice setup yet -- self-checks above prove the parsing and exit-trigger "
              "logic are correct, but the bot itself can't run without a mic backend.")
        print("Recommended (no compiling required, works on any recent Windows Python):")
        print("    pip install SpeechRecognition sounddevice numpy --break-system-packages")
        print("Alternative, if you'd rather use pyaudio and it builds cleanly on your machine:")
        print("    pip install SpeechRecognition pyaudio --break-system-packages")
        sys.exit(0)
    print(f"Using microphone backend: {'pyaudio' if HAS_PYAUDIO else 'sounddevice'}")
    bot = VoicePaperTradeBot()
    bot.run_forever()
