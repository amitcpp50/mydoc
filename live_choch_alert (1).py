"""
live_choch_alert.py  —  SELF-CONTAINED
───────────────────────────────────────────────────────────────────────────────
Live CHoCH (Change of Character) Detector
Instruments : NIFTY 50 + SENSEX  |  Timeframe : 5-min candles
Data source : Dhan REST API v2  (polled every 5 min during market hours)
Alerts      : Telegram Bot

No external project files needed — swing detection and CHoCH logic
are embedded directly in this file.

SETUP
─────
  pip install requests pandas pytz numpy

  export DHAN_CLIENT_ID="..."
  export DHAN_ACCESS_TOKEN="..."
  export TELEGRAM_BOT_TOKEN="..."       # from @BotFather on Telegram
  export TELEGRAM_CHAT_ID="..."         # your chat/group ID

  python live_choch_alert.py

HOW IT WORKS
────────────
  1. At startup  : fetch today's already-closed 5-min candles → seed buffer
                   run CHoCH on seed → alert on anything already formed
  2. Every 5 min : fetch fresh candles from Dhan REST
                   append only NEW candles to today's buffer
                   re-run CHoCH on full buffer
                   alert on any new event not seen before (dedup guard)
  3. At 15:30    : final poll → EOD summary → exit

CHOCH DEFINITION
────────────────
  BULLISH CHoCH : bearish structure (LH+LL) broken when a candle closes
                  ABOVE the most recent Lower High (LH).
  BEARISH CHoCH : bullish structure (HH+HL) broken when a candle closes
                  BELOW the most recent Higher Low (HL).
"""

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests


# ═════════════════════════════════════════════════════════════════════════════
# SWING DETECTION  (ZigZag method)
# ═════════════════════════════════════════════════════════════════════════════
def detect_swings_pivot(
    df: pd.DataFrame,
    left: int = 5,
    right: int = 5,
) -> pd.DataFrame:
    """
    Classic pivot swing detection.

    A candle is a swing HIGH if its high is the highest over
    [i - left … i + right].  Symmetric window by default (left == right),
    but asymmetric windows are supported for faster confirmation with fewer
    look-right bars.

    Parameters
    ----------
    df    : DataFrame with columns [high, low]
    left  : bars to the LEFT  that must be lower/higher
    right : bars to the RIGHT that must be lower/higher
              (introduces `right`-bar lag before a point is confirmed)

    Returns
    -------
    DataFrame with added columns:
        swing_high  bool    True at confirmed swing high
        swing_low   bool    True at confirmed swing low
        sh_price    float   high value at swing highs,  NaN elsewhere
        sl_price    float   low  value at swing lows,   NaN elsewhere
    """
    df = df.copy()
    n = len(df)

    highs = df["high"].values
    lows  = df["low"].values

    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows [i - left : i + right + 1]
        # The pivot candle is at position `left` inside the window
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            sh[i] = True
        if lows[i]  == window_l.min() and (window_l == lows[i] ).sum() == 1:
            sl[i] = True

    df["swing_high"] = sh
    df["swing_low"]  = sl
    df["sh_price"]   = np.where(sh, highs, np.nan)
    df["sl_price"]   = np.where(sl, lows,  np.nan)
    return df


def detect_swings_zigzag(
    df: pd.DataFrame,
    min_pct: float = 0.5,
    pivot_bars: int = 3,
) -> pd.DataFrame:
    """
    ZigZag swing detector.

    Step 1 : Identify candidate pivot points using detect_swings_pivot with
             `pivot_bars` on each side.
    Step 2 : Walk through candidates and keep a pivot only if the price moved
             at least `min_pct` % from the previous confirmed pivot.
             Consecutive highs → keep only the higher one.
             Consecutive lows  → keep only the lower one.

    This eliminates minor noise swings and keeps only structurally significant
    turning points.

    Parameters
    ----------
    df         : DataFrame with columns [high, low]
    min_pct    : minimum % move required to confirm a new swing (default 0.5 %)
    pivot_bars : look-left/right bars for the initial pivot scan (default 3)

    Returns
    -------
    Same columns as detect_swings_pivot.
    """
    # ── Step 1: raw pivots ────────────────────────────────────────────────
    raw = detect_swings_pivot(df, left=pivot_bars, right=pivot_bars)

    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)

    # Collect candidate indices and type
    candidates = []
    for i in range(n):
        if raw.at[raw.index[i], "swing_high"] and raw.at[raw.index[i], "swing_low"]:
            # Rare but possible on very flat candles: pick high
            candidates.append((i, "H", highs[i]))
        elif raw.at[raw.index[i], "swing_high"]:
            candidates.append((i, "H", highs[i]))
        elif raw.at[raw.index[i], "swing_low"]:
            candidates.append((i, "L", lows[i]))

    if not candidates:
        df = df.copy()
        df["swing_high"] = False
        df["swing_low"]  = False
        df["sh_price"]   = np.nan
        df["sl_price"]   = np.nan
        return df

    # ── Step 2: ZigZag filter ─────────────────────────────────────────────
    confirmed = [candidates[0]]  # seed with first candidate

    for idx, typ, price in candidates[1:]:
        last_idx, last_typ, last_price = confirmed[-1]

        if typ == last_typ:
            # Same direction: keep the more extreme one
            if typ == "H" and price > last_price:
                confirmed[-1] = (idx, typ, price)
            elif typ == "L" and price < last_price:
                confirmed[-1] = (idx, typ, price)
        else:
            # Direction change: only keep if move is large enough
            pct_move = abs(price - last_price) / last_price * 100
            if pct_move >= min_pct:
                confirmed.append((idx, typ, price))
            else:
                # Still same effective direction, update to more extreme
                if typ == "H" and price > last_price:
                    confirmed[-1] = (idx, typ, price)
                elif typ == "L" and price < last_price:
                    confirmed[-1] = (idx, typ, price)

    for idx, typ, price in confirmed:
        if typ == "H":
            sh[idx] = True
        else:
            sl[idx] = True

    df = df.copy()
    df["swing_high"] = sh
    df["swing_low"]  = sl
    df["sh_price"]   = np.where(sh, highs, np.nan)
    df["sl_price"]   = np.where(sl, lows,  np.nan)
    return df


def get_swing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean DataFrame of only the swing points, with columns:
        datetime | type | price | candle_index
    Sorted by datetime ascending.
    """
    sh_df = df[df["swing_high"]].copy()
    sh_df["type"]  = "SWING_HIGH"
    sh_df["price"] = sh_df["sh_price"]

    sl_df = df[df["swing_low"]].copy()
    sl_df["type"]  = "SWING_LOW"
    sl_df["price"] = sl_df["sl_price"]

    combined = pd.concat([sh_df[["datetime", "type", "price"]],
                          sl_df[["datetime", "type", "price"]]])
    combined.reset_index(names="candle_index", inplace=True)
    combined.sort_values("datetime", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
# ═════════════════════════════════════════════════════════════════════════════
# CHoCH DETECTION  (Change of Character)
# ═════════════════════════════════════════════════════════════════════════════
def label_swing_structure(swing_seq: pd.DataFrame) -> pd.DataFrame:
    """
    Given an alternating sequence of swing highs and lows (single day),
    label each point as HH / HL / LH / LL relative to the previous swing
    of the same type.

    Parameters
    ----------
    swing_seq : DataFrame with columns [datetime, type, price]
                sorted chronologically, alternating SH / SL

    Returns
    -------
    Same DataFrame with an added 'label' column:
        HH  Higher High   (SH > previous SH)
        LH  Lower  High   (SH < previous SH)
        HL  Higher Low    (SL > previous SL)
        LL  Lower  Low    (SL < previous SL)
        EH  Equal  High   (SH == previous SH)  — treated as LH in logic
        EL  Equal  Low    (SL == previous SL)  — treated as HL in logic
        --  First point of that type (no prior to compare)
    """
    seq = swing_seq.copy().reset_index(drop=True)
    seq["label"] = "--"

    last_sh_price = None
    last_sl_price = None

    for i, row in seq.iterrows():
        if row["type"] == "SWING_HIGH":
            if last_sh_price is None:
                seq.at[i, "label"] = "--"
            elif row["price"] > last_sh_price:
                seq.at[i, "label"] = "HH"
            elif row["price"] < last_sh_price:
                seq.at[i, "label"] = "LH"
            else:
                seq.at[i, "label"] = "EH"
            last_sh_price = row["price"]

        else:  # SWING_LOW
            if last_sl_price is None:
                seq.at[i, "label"] = "--"
            elif row["price"] > last_sl_price:
                seq.at[i, "label"] = "HL"
            elif row["price"] < last_sl_price:
                seq.at[i, "label"] = "LL"
            else:
                seq.at[i, "label"] = "EL"
            last_sl_price = row["price"]

    return seq


def assign_bias(labeled_seq: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through the labeled swing sequence and assign a rolling market bias:
      BULLISH  — last pair of labels was (HH + HL)
      BEARISH  — last pair of labels was (LH + LL)
      NEUTRAL  — mixed / insufficient data

    The bias is set *after* each new swing is confirmed, so the
    CHoCH watcher knows what structure is currently active.
    """
    seq = labeled_seq.copy()
    seq["bias"] = "NEUTRAL"

    # We need at least one SH label and one SL label to determine bias
    last_sh_label = None
    last_sl_label = None
    current_bias  = "NEUTRAL"

    for i, row in seq.iterrows():
        if row["type"] == "SWING_HIGH" and row["label"] not in ("--",):
            last_sh_label = row["label"]
        elif row["type"] == "SWING_LOW" and row["label"] not in ("--",):
            last_sl_label = row["label"]

        # Update bias when we have both a recent SH and SL label
        if last_sh_label is not None and last_sl_label is not None:
            if last_sh_label == "HH" and last_sl_label == "HL":
                current_bias = "BULLISH"
            elif last_sh_label in ("LH", "EH") and last_sl_label in ("LL", "EL"):
                current_bias = "BEARISH"
            # Partial confirmation: one leg agrees
            elif last_sh_label == "HH" or last_sl_label == "HL":
                current_bias = "BULLISH"
            elif last_sh_label in ("LH", "EH") or last_sl_label in ("LL", "EL"):
                current_bias = "BEARISH"

        seq.at[i, "bias"] = current_bias

    return seq


def detect_choch_day(
    day_candles: pd.DataFrame,
    day_swings:  pd.DataFrame,
    use_wick:    bool = False,
) -> pd.DataFrame:
    """
    Detect all CHoCH events for a single trading day.

    Parameters
    ----------
    day_candles : 5-min OHLCV candles for the day (datetime sorted)
    day_swings  : labeled + biased swing sequence for the day
                  (output of assign_bias ∘ label_swing_structure)
    use_wick    : if True, use high/low for breach check instead of close.
                  Close-based (default) is more conservative and avoids
                  false triggers from wicks that snap back.

    Returns
    -------
    pd.DataFrame — one row per CHoCH event, with columns:

        choch_time       datetime of the candle that triggered the CHoCH
        choch_type       BULLISH_CHOCH  or  BEARISH_CHOCH
        break_price      the swing level that was broken
        break_level_type HL (broken from above) or LH (broken from below)
        break_level_time datetime when that swing level was formed
        trigger_price    close (or wick) that confirmed the break
        candle_idx       index in the full DataFrame
        prior_bias       the market structure that was active BEFORE this CHoCH
        swing_count      how many swings had formed before this CHoCH fired
    """
    candles = day_candles.copy().reset_index(drop=True)
    swings  = day_swings.copy().reset_index(drop=True)

    choch_events = []

    # ── State machine variables ──────────────────────────────────────────
    # Active level to watch: for BULLISH bias we watch the most recent HL;
    # for BEARISH bias we watch the most recent LH.
    active_hl_price = None   # current HL level (bearish CHoCH trigger line)
    active_hl_time  = None
    active_lh_price = None   # current LH level (bullish CHoCH trigger line)
    active_lh_time  = None
    current_bias    = "NEUTRAL"
    swing_count     = 0
    last_choch_time = None   # prevent re-triggering on the same break

    # ── Map swing times to candle indices for efficient lookup ────────────
    candle_dt_index = pd.Series(candles.index, index=candles["datetime"])

    # ── Walk candle by candle ─────────────────────────────────────────────
    swing_ptr = 0  # pointer into the swings DataFrame
    n_swings  = len(swings)

    for ci, candle in candles.iterrows():
        cdt = candle["datetime"]

        # ── Absorb all swings that have been confirmed up to this candle ──
        while swing_ptr < n_swings and swings.loc[swing_ptr, "datetime"] <= cdt:
            sw = swings.loc[swing_ptr]
            current_bias = sw["bias"]
            swing_count += 1

            # Update the watched levels
            if sw["type"] == "SWING_LOW" and sw["label"] in ("HL", "--"):
                # Any swing low can serve as the HL watch level if it's
                # higher than the prior low (or first low of the day)
                if active_hl_price is None or sw["price"] > active_hl_price:
                    active_hl_price = sw["price"]
                    active_hl_time  = sw["datetime"]
                # Also reset if label is HL (confirmed higher low)
                if sw["label"] == "HL":
                    active_hl_price = sw["price"]
                    active_hl_time  = sw["datetime"]

            if sw["type"] == "SWING_LOW" and sw["label"] == "LL":
                # In bearish structure, update LL as new reference low
                # but keep the LH watch level unchanged (that's what we break)
                pass  # LH watch level only updates on new LH swings

            if sw["type"] == "SWING_HIGH" and sw["label"] in ("LH", "EH"):
                active_lh_price = sw["price"]
                active_lh_time  = sw["datetime"]

            if sw["type"] == "SWING_HIGH" and sw["label"] == "HH":
                # In bullish structure, new HH raises the bar but
                # HL watch level remains (updated by HL swings)
                pass

            # Always track the most recent swing low as potential HL watch
            if sw["type"] == "SWING_LOW":
                active_hl_price = sw["price"]
                active_hl_time  = sw["datetime"]

            # Always track the most recent swing high as potential LH watch
            if sw["type"] == "SWING_HIGH":
                active_lh_price = sw["price"]
                active_lh_time  = sw["datetime"]

            swing_ptr += 1

        # ── Skip if no established structure yet ──────────────────────────
        if current_bias == "NEUTRAL" or swing_count < 2:
            continue

        # ── Skip if this candle is itself a swing candle (avoid self-trigger)
        # CHoCH must fire on a candle AFTER the swing was formed
        swing_times = set(swings["datetime"].values)
        if cdt in swing_times:
            continue

        # Use close or wick for breach check
        breach_low  = candle["low"]   if use_wick else candle["close"]
        breach_high = candle["high"]  if use_wick else candle["close"]

        # ── BEARISH CHoCH: bullish structure broken ────────────────────────
        # Price closes below the most recent HL in a BULLISH bias
        if (current_bias == "BULLISH"
                and active_hl_price is not None
                and breach_low < active_hl_price
                and (last_choch_time is None or cdt > last_choch_time)):

            choch_events.append({
                "choch_time":        cdt,
                "choch_type":        "BEARISH_CHOCH",
                "break_price":       active_hl_price,
                "break_level_type":  "HL",
                "break_level_time":  active_hl_time,
                "trigger_price":     candle["close"] if not use_wick else candle["low"],
                "candle_idx":        ci,
                "prior_bias":        "BULLISH",
                "swing_count":       swing_count,
            })
            last_choch_time = cdt
            current_bias    = "BEARISH"   # structure has changed
            # Reset the HL watch; now watch for LH to break upward
            active_hl_price = None
            active_hl_time  = None

        # ── BULLISH CHoCH: bearish structure broken ────────────────────────
        # Price closes above the most recent LH in a BEARISH bias
        elif (current_bias == "BEARISH"
                and active_lh_price is not None
                and breach_high > active_lh_price
                and (last_choch_time is None or cdt > last_choch_time)):

            choch_events.append({
                "choch_time":        cdt,
                "choch_type":        "BULLISH_CHOCH",
                "break_price":       active_lh_price,
                "break_level_type":  "LH",
                "break_level_time":  active_lh_time,
                "trigger_price":     candle["close"] if not use_wick else candle["high"],
                "candle_idx":        ci,
                "prior_bias":        "BEARISH",
                "swing_count":       swing_count,
            })
            last_choch_time = cdt
            current_bias    = "BULLISH"   # structure has changed
            # Reset the LH watch; now watch for HL to break downward
            active_lh_price = None
            active_lh_time  = None

    return pd.DataFrame(choch_events)
# ═════════════════════════════════════════════════════════════════════════════
# LIVE SESSION  —  Dhan REST + Telegram
# ═════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DHAN_CLIENT_ID     = os.getenv("DHAN_CLIENT_ID",     "YOUR_CLIENT_ID")
DHAN_ACCESS_TOKEN  = os.getenv("DHAN_ACCESS_TOKEN",  "YOUR_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID")

# ZigZag sensitivity — 0.2 is correct for intraday micro-structure
CHOCH_MIN_PCT    = 0.2
CHOCH_PIVOT_BARS = 3

CANDLE_MINUTES = 5          # poll interval matches candle size
POLL_GRACE_SEC = 30         # seconds after candle boundary before polling
                            # (gives Dhan server time to finalise the candle)

IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN_H,  MARKET_OPEN_M  = 9,  15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30

INSTRUMENTS = [
    {
        "name":             "NIFTY 50",
        "security_id":      "13",
        "exchange_segment": "IDX_I",
        "instrument":       "INDEX",
    },
    {
        "name":             "SENSEX",
        "security_id":      "51",
        "exchange_segment": "IDX_I",
        "instrument":       "INDEX",
    },
]

DHAN_INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING  (console + file)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("choch_live.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("choch")

# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS
# ─────────────────────────────────────────────────────────────────────────────

def now_ist() -> datetime:
    return datetime.now(IST)

def is_market_open() -> bool:
    n = now_ist()
    if n.weekday() >= 5:        # Saturday / Sunday
        return False
    open_  = n.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
    close_ = n.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    return open_ <= n <= close_

def next_poll_time() -> datetime:
    """
    Return the next datetime at which we should poll.
    That is: the next 5-min candle boundary + POLL_GRACE_SEC.
    e.g. if now is 10:18:05, next candle closes at 10:20, we poll at 10:20:30.
    """
    n = now_ist()
    current_slot_min = (n.minute // CANDLE_MINUTES) * CANDLE_MINUTES
    next_slot = n.replace(minute=current_slot_min, second=0, microsecond=0) \
                + timedelta(minutes=CANDLE_MINUTES)
    return next_slot + timedelta(seconds=POLL_GRACE_SEC)

def seconds_until_open() -> float:
    """Seconds until next market open (could be today or a future weekday)."""
    n = now_ist()
    for delta in range(8):          # look at most 1 week ahead
        candidate = (n + timedelta(days=delta)).replace(
            hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
        if candidate > n and candidate.weekday() < 5:
            return (candidate - n).total_seconds()
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────

def telegram_send(text: str) -> bool:
    """POST a message to Telegram. Returns True on success."""
    if "YOUR_" in TELEGRAM_BOT_TOKEN or "YOUR_" in TELEGRAM_CHAT_ID:
        print(f"\n{'━'*60}\n🔔 [TELEGRAM ALERT]\n{text}\n{'━'*60}\n")
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": "HTML",
        }, timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        log.error(f"Telegram send failed: {exc}")
        return False

def fmt_choch_alert(instr_name: str, row: pd.Series) -> str:
    """Build the Telegram message for a CHoCH event."""
    is_bull  = row["choch_type"] == "BULLISH_CHOCH"
    emoji    = "🟢" if is_bull else "🔴"
    label    = "BULLISH CHoCH" if is_bull else "BEARISH CHoCH"
    candle_t = pd.Timestamp(row["choch_time"]).strftime("%H:%M")
    level_t  = pd.Timestamp(row["break_level_time"]).strftime("%H:%M")

    lines = [
        f"{emoji} <b>{label} — {instr_name}</b>",
        "",
        f"⏰ Candle close : <b>{candle_t} IST</b>",
        f"📍 Broke {row['break_level_type']} level : <b>{row['break_price']:.2f}</b>  (formed {level_t})",
        f"🎯 Close price  : <b>{row['trigger_price']:.2f}</b>",
        f"↔️  Prior bias   : {row['prior_bias']}",
        f"📊 Timeframe    : 5-min  |  {date.today().strftime('%d-%b-%Y')}",
        "",
        f"<i>Structure changed — watch for entry confirmation</i>",
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# DHAN DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

def dhan_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id":    DHAN_CLIENT_ID,
    }

def fetch_candles(instr: dict, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Call Dhan v2 intraday endpoint and return a clean OHLCV DataFrame.
    Returns empty DataFrame on any error.
    """
    payload = {
        "securityId":      instr["security_id"],
        "exchangeSegment": instr["exchange_segment"],
        "instrument":      instr["instrument"],
        "interval":        str(CANDLE_MINUTES),
        "oi":              False,
        "fromDate":        from_date,
        "toDate":          to_date,
    }
    try:
        resp = requests.post(
            DHAN_INTRADAY_URL,
            json=payload,
            headers=dhan_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        log.error(f"[{instr['name']}] Dhan fetch error: {exc}")
        return pd.DataFrame()

    required = {"open", "high", "low", "close", "volume", "timestamp"}
    if not required.issubset(data.keys()):
        log.warning(f"[{instr['name']}] Unexpected response keys: {list(data.keys())}")
        return pd.DataFrame()

    if not data["timestamp"]:
        return pd.DataFrame()

    df = pd.DataFrame({
        "datetime": [datetime.fromtimestamp(ts, tz=IST) for ts in data["timestamp"]],
        "open":     [float(x) for x in data["open"]],
        "high":     [float(x) for x in data["high"]],
        "low":      [float(x) for x in data["low"]],
        "close":    [float(x) for x in data["close"]],
        "volume":   [float(x) for x in data["volume"]],
    })
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def fetch_today_closed_candles(instr: dict) -> pd.DataFrame:
    """
    Fetch today's 5-min candles, returning only *closed* ones.
    A candle is closed if its start time is earlier than the current
    5-min boundary (i.e. its bar has fully elapsed).
    """
    today     = date.today().strftime("%Y-%m-%d")
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    df = fetch_candles(instr, yesterday, today)
    if df.empty:
        return df

    # Keep only today's candles
    df = df[df["datetime"].dt.date == date.today()].copy()

    # Drop the still-open candle: the one whose slot == current slot
    n = now_ist()
    current_slot = n.replace(
        minute=(n.minute // CANDLE_MINUTES) * CANDLE_MINUTES,
        second=0, microsecond=0,
    ).replace(tzinfo=IST)

    df = df[df["datetime"] < current_slot].copy()
    df.reset_index(drop=True, inplace=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CHOCH DETECTION ON A CANDLE BUFFER
# ─────────────────────────────────────────────────────────────────────────────

def run_choch_on_buffer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full CHoCH pipeline on a day's candle buffer.
    Returns a DataFrame of CHoCH events (may be empty).
    df must have columns: datetime, open, high, low, close, volume
    """
    if len(df) < 6:
        return pd.DataFrame()

    df = df.copy()
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 1: swing detection
    try:
        df_sw  = detect_swings_zigzag(df, min_pct=CHOCH_MIN_PCT, pivot_bars=CHOCH_PIVOT_BARS)
        sw_tbl = get_swing_table(df_sw)
    except Exception as exc:
        log.debug(f"Swing detection error: {exc}")
        return pd.DataFrame()

    if len(sw_tbl) < 2:
        return pd.DataFrame()

    # Step 2: label + bias
    labeled = label_swing_structure(sw_tbl)
    biased  = assign_bias(labeled)

    # Step 3: CHoCH detection
    try:
        events = detect_choch_day(df, biased, use_wick=False)
    except Exception as exc:
        log.debug(f"CHoCH detection error: {exc}")
        return pd.DataFrame()

    return events

# ─────────────────────────────────────────────────────────────────────────────
# PER-INSTRUMENT STATE
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentSession:
    """
    Holds all mutable state for one instrument during the trading session.
      candle_buffer   : all closed 5-min candles for today, growing each poll
      alerted_keys    : set of CHoCH event keys already alerted (dedup guard)
      last_candle_dt  : datetime of the last candle in the buffer
    """

    def __init__(self, instr: dict):
        self.instr          = instr
        self.candle_buffer  = pd.DataFrame()
        self.alerted_keys: set[str] = set()
        self.last_candle_dt = None

    # ── Key that uniquely identifies a CHoCH event ────────────────────────
    @staticmethod
    def event_key(row: pd.Series) -> str:
        return f"{row['choch_type']}|{row['choch_time']}|{row['break_price']:.2f}"

    # ── Append only genuinely new candles to the buffer ───────────────────
    def update_buffer(self, fresh: pd.DataFrame) -> int:
        """
        Merge fresh candles into the buffer.
        Returns the count of new candles added.
        """
        if fresh.empty:
            return 0

        if self.candle_buffer.empty:
            self.candle_buffer = fresh.copy()
            self.last_candle_dt = fresh["datetime"].iloc[-1]
            return len(fresh)

        existing_times = set(self.candle_buffer["datetime"].astype(str))
        new_rows = fresh[~fresh["datetime"].astype(str).isin(existing_times)]

        if new_rows.empty:
            return 0

        self.candle_buffer = pd.concat(
            [self.candle_buffer, new_rows], ignore_index=True
        )
        self.candle_buffer.sort_values("datetime", inplace=True)
        self.candle_buffer.reset_index(drop=True, inplace=True)
        self.last_candle_dt = self.candle_buffer["datetime"].iloc[-1]
        return len(new_rows)

    # ── Run detection and alert on new events ─────────────────────────────
    def detect_and_alert(self) -> int:
        """
        Run CHoCH on the current buffer.
        Send Telegram for any new event not yet alerted.
        Returns count of new alerts fired.
        """
        events = run_choch_on_buffer(self.candle_buffer)
        if events.empty:
            return 0

        fired = 0
        for _, row in events.iterrows():
            key = self.event_key(row)
            if key in self.alerted_keys:
                continue
            self.alerted_keys.add(key)

            is_bull  = row["choch_type"] == "BULLISH_CHOCH"
            label    = "BULLISH CHoCH" if is_bull else "BEARISH CHoCH"
            candle_t = pd.Timestamp(row["choch_time"]).strftime("%H:%M")
            log.info(
                f"[{self.instr['name']}] *** {label} at {candle_t}  "
                f"break={row['break_price']:.2f}  close={row['trigger_price']:.2f}"
            )
            msg = fmt_choch_alert(self.instr["name"], row)
            telegram_send(msg)
            fired += 1

        return fired

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE POLL — fetch + detect + alert for all instruments
# ─────────────────────────────────────────────────────────────────────────────

def poll_all(sessions: list[InstrumentSession]) -> None:
    """Execute one poll cycle for every instrument."""
    n = now_ist()
    log.info(f"{'─'*55}")
    log.info(f"Poll at {n.strftime('%H:%M:%S IST')}")

    for sess in sessions:
        fresh = fetch_today_closed_candles(sess.instr)
        new_count = sess.update_buffer(fresh)

        buf_len = len(sess.candle_buffer)
        last_close = (
            f"{sess.candle_buffer['close'].iloc[-1]:.2f}"
            if not sess.candle_buffer.empty else "—"
        )
        log.info(
            f"  [{sess.instr['name']:>10}]  "
            f"buffer={buf_len:>3} candles  "
            f"+{new_count} new  "
            f"last_close={last_close}"
        )

        if new_count > 0:
            fired = sess.detect_and_alert()
            if fired == 0:
                log.info(f"  [{sess.instr['name']:>10}]  no new CHoCH")

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP SEED  — run once at the beginning of the session
# ─────────────────────────────────────────────────────────────────────────────

def seed_session(sessions: list[InstrumentSession]) -> None:
    """
    Pull today's candles that have already closed before this script started.
    Alert on any CHoCH that already formed — trader may have missed it.
    """
    log.info("Seeding candle buffers with today's historical data …")

    for sess in sessions:
        fresh = fetch_today_closed_candles(sess.instr)
        if fresh.empty:
            log.info(f"  [{sess.instr['name']}]  no candles yet (pre-open?)")
            continue

        sess.update_buffer(fresh)
        log.info(
            f"  [{sess.instr['name']}]  seeded {len(sess.candle_buffer)} candles  "
            f"({fresh['datetime'].iloc[0].strftime('%H:%M')} → "
            f"{fresh['datetime'].iloc[-1].strftime('%H:%M')})"
        )
        fired = sess.detect_and_alert()
        if fired:
            log.info(f"  [{sess.instr['name']}]  {fired} historical CHoCH(s) alerted on startup")

# ─────────────────────────────────────────────────────────────────────────────
# END-OF-DAY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def send_eod_summary(sessions: list[InstrumentSession]) -> None:
    lines = [
        "📋 <b>CHoCH Session Summary</b>",
        f"📅 {date.today().strftime('%d-%b-%Y')}  |  5-min",
        "",
    ]
    for sess in sessions:
        n_choch = len(sess.alerted_keys)
        n_candles = len(sess.candle_buffer)
        lines.append(f"<b>{sess.instr['name']}</b>")
        lines.append(f"  Candles processed : {n_candles}")
        lines.append(f"  CHoCH events fired: {n_choch}")
        if n_choch:
            for key in sorted(sess.alerted_keys):
                # key format: TYPE|datetime|price
                parts = key.split("|")
                etype = parts[0].replace("_CHOCH","")
                etime = pd.Timestamp(parts[1]).strftime("%H:%M") if len(parts)>1 else "?"
                eprice = parts[2] if len(parts)>2 else "?"
                icon = "🟢" if "BULL" in etype else "🔴"
                lines.append(f"    {icon} {etype} @ {etime}  break={eprice}")
        lines.append("")

    telegram_send("\n".join(lines))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Live CHoCH Detector  |  NIFTY 50 + SENSEX  |  5-min")
    log.info(f"  Started : {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")
    log.info("=" * 60)

    # Credential check
    if "YOUR_" in DHAN_CLIENT_ID or "YOUR_" in DHAN_ACCESS_TOKEN:
        log.error("Dhan credentials not set. "
                  "Export DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN.")
        sys.exit(1)

    # Build per-instrument session state
    sessions = [InstrumentSession(instr) for instr in INSTRUMENTS]

    # ── Wait for market open if running before 09:15 ─────────────────────
    if not is_market_open():
        wait_s = seconds_until_open()
        log.info(f"Market not open. Waiting {wait_s/60:.1f} min …")
        telegram_send(
            f"⏳ <b>CHoCH Detector waiting</b>\n"
            f"Market opens in {wait_s/60:.0f} min  "
            f"({date.today().strftime('%d-%b-%Y')})"
        )
        time.sleep(wait_s)

    # ── Startup notification ──────────────────────────────────────────────
    telegram_send(
        f"🚀 <b>CHoCH Detector LIVE</b>\n\n"
        f"📈 NIFTY 50 + SENSEX\n"
        f"⏱ 5-min candles, REST poll every 5 min\n"
        f"🕐 {now_ist().strftime('%d-%b-%Y %H:%M IST')}\n\n"
        f"Alerts fire on every new CHoCH. "
        f"Use close-based breach (conservative)."
    )

    # ── Seed with candles already closed today ────────────────────────────
    seed_session(sessions)

    # ── Main poll loop ────────────────────────────────────────────────────
    try:
        while is_market_open():
            # Calculate sleep until the next 5-min boundary + grace
            target = next_poll_time()
            sleep_s = (target - now_ist()).total_seconds()
            if sleep_s > 0:
                log.info(
                    f"Next poll at {target.strftime('%H:%M:%S')}  "
                    f"(sleeping {sleep_s:.0f}s)"
                )
                time.sleep(sleep_s)

            # One last check before polling — market might have just closed
            if not is_market_open():
                break

            poll_all(sessions)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")

    # ── End-of-day ────────────────────────────────────────────────────────
    log.info("Market closed. Running final CHoCH pass on full day …")

    # Final poll to catch anything in the last few candles
    poll_all(sessions)

    log.info("Sending end-of-day summary …")
    send_eod_summary(sessions)

    log.info("Session complete. Exiting.")


if __name__ == "__main__":
    main()
