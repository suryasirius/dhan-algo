"""
Reads the strike_momentum_trades.csv already saved by test_strike_momentum.py
(no re-fetching from Dhan needed) and breaks results down by weekday, with
each day further split into 30-minute sessions from 9:15 to 15:30.

Run this AFTER test_strike_momentum.py has already produced
strike_momentum_trades.csv in the same folder.
"""

import datetime as dt
import pandas as pd

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# 12 full 30-minute slots + one final 15-minute slot to close out at 15:30
SESSION_BUCKETS = [
    (dt.time(9, 15), dt.time(9, 45)),
    (dt.time(9, 45), dt.time(10, 15)),
    (dt.time(10, 15), dt.time(10, 45)),
    (dt.time(10, 45), dt.time(11, 15)),
    (dt.time(11, 15), dt.time(11, 45)),
    (dt.time(11, 45), dt.time(12, 15)),
    (dt.time(12, 15), dt.time(12, 45)),
    (dt.time(12, 45), dt.time(13, 15)),
    (dt.time(13, 15), dt.time(13, 45)),
    (dt.time(13, 45), dt.time(14, 15)),
    (dt.time(14, 15), dt.time(14, 45)),
    (dt.time(14, 45), dt.time(15, 15)),
    (dt.time(15, 15), dt.time(15, 30)),
]


def classify_30min(ts):
    t = ts.time()
    for i, (start, end) in enumerate(SESSION_BUCKETS):
        is_last = (i == len(SESSION_BUCKETS) - 1)
        if (start <= t < end) or (is_last and start <= t <= end):
            return f"{i + 1:02d} {start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    return "99 Outside session"


def report(group, label):
    if len(group) == 0:
        return
    wr = 100 * (group["points"] > 0).mean()
    print(f"  {label:<20} trades={len(group):<4} win rate={wr:5.1f}%  "
          f"avg={group['points'].mean():6.2f}  net={group['points'].sum():8.2f}")


def main():
    df = pd.read_csv("strike_momentum_trades.csv", parse_dates=["entry_time", "exit_time"])
    df["weekday"] = df["entry_time"].dt.day_name()
    df["session_30min"] = df["entry_time"].apply(classify_30min)

    print(f"Total trades in file: {len(df)}\n")

    print("================ BY WEEKDAY, EACH SPLIT INTO 30-MIN SESSIONS ================")
    for wd in WEEKDAY_ORDER:
        day_group = df[df["weekday"] == wd]
        if day_group.empty:
            continue
        wr = 100 * (day_group["points"] > 0).mean()
        print(f"\n{wd} -- total trades={len(day_group)}  win rate={wr:.1f}%  "
              f"net={day_group['points'].sum():.2f}")
        for session_label, session_group in day_group.groupby("session_30min"):
            report(session_group, session_label)
    print("\n===============================================================================")

    print("\n================ ALL WEEKDAYS COMBINED, BY 30-MIN SESSION ================")
    for session_label, session_group in df.groupby("session_30min"):
        report(session_group, session_label)
    print("============================================================================")

    df.to_csv("strike_momentum_trades_with_sessions.csv", index=False)
    print("\nSaved strike_momentum_trades_with_sessions.csv (same data + weekday/session columns)")


if __name__ == "__main__":
    main()
