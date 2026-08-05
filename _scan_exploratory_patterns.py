import pandas as pd
import numpy as np

df = pd.read_csv('dhan_historical_data_180days_v5.csv', parse_dates=['datetime'])
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time
days = sorted(df['date'].unique())
print(f"Total days: {len(days)}")

daily = {}
for d in days:
    dd = df[df['date']==d].sort_values('datetime').reset_index(drop=True)
    daily[d] = dd

results = {}

# ---------- 1. Gap fill ----------
prev_close = None
gap_events = []
for i, d in enumerate(days):
    dd = daily[d]
    day_open = dd.iloc[0]['open']
    if prev_close is not None:
        gap = day_open - prev_close
        gap_pct = gap/prev_close*100
        if abs(gap_pct) >= 0.05:  # meaningful gap threshold
            if gap > 0:
                filled = (dd['low'] <= prev_close).any()
            else:
                filled = (dd['high'] >= prev_close).any()
            gap_events.append({'date':d,'gap_pct':gap_pct,'direction':'up' if gap>0 else 'down','filled':filled})
    prev_close = dd.iloc[-1]['close']

gdf = pd.DataFrame(gap_events)
print("\n=== GAP FILL ===")
print(f"Days with gap >=0.05%: {len(gdf)} / {len(days)-1}")
if len(gdf):
    print(f"Overall fill rate: {gdf['filled'].mean()*100:.1f}%")
    for direc in ['up','down']:
        sub = gdf[gdf['direction']==direc]
        print(f"  Gap {direc}: n={len(sub)}, fill rate={sub['filled'].mean()*100:.1f}%")
    biggdf = gdf[gdf['gap_pct'].abs()>=0.15]
    if len(biggdf):
        print(f"Large gaps (>=0.15%): n={len(biggdf)}, fill rate={biggdf['filled'].mean()*100:.1f}%")
    else:
        print("no large gaps")

# ---------- 2. Opening range breakout (first 15 min = 9:15-9:30) ----------
orb_events = []
for d in days:
    dd = daily[d]
    orb = dd[(dd['datetime'].dt.time >= pd.Timestamp('09:15').time()) & (dd['datetime'].dt.time < pd.Timestamp('09:30').time())]
    if len(orb)==0: continue
    orb_high = orb['high'].max()
    orb_low = orb['low'].min()
    rest = dd[dd['datetime'].dt.time >= pd.Timestamp('09:30').time()]
    if len(rest)==0: continue
    broke_up = (rest['high'] > orb_high).any()
    broke_down = (rest['low'] < orb_low).any()
    day_close = dd.iloc[-1]['close']
    up_first_idx = rest[rest['high']>orb_high].index.min() if broke_up else np.inf
    down_first_idx = rest[rest['low']<orb_low].index.min() if broke_down else np.inf
    if up_first_idx < down_first_idx:
        direction = 'up'
        continuation = day_close > orb_high
    elif down_first_idx < up_first_idx:
        direction = 'down'
        continuation = day_close < orb_low
    else:
        direction = 'none'
        continuation = None
    orb_events.append({'date':d,'direction':direction,'continuation':continuation, 'orb_range':orb_high-orb_low})

odf = pd.DataFrame(orb_events)
print("\n=== OPENING RANGE BREAKOUT (9:15-9:30) ===")
valid = odf[odf['direction']!='none']
print(f"Days with a directional break: {len(valid)} / {len(odf)}")
print(f"Continuation rate (close beyond ORB on breakout side): {valid['continuation'].mean()*100:.1f}%")
for direc in ['up','down']:
    sub = valid[valid['direction']==direc]
    print(f"  {direc} break: n={len(sub)}, continuation={sub['continuation'].mean()*100:.1f}%")

# ---------- 3. Round number (nearest 100) reaction ----------
round_events = []
for d in days:
    dd = daily[d]
    for i in range(1, len(dd)):
        prev_c = dd.iloc[i-1]['close']
        nearest100 = round(prev_c/100)*100
        dist = abs(prev_c - nearest100)
        if dist <= 3:
            future = dd.iloc[i:i+6]
            if len(future) < 2: continue
            move5 = future['close'].iloc[-1] - prev_c
            round_events.append({'date':d,'level':nearest100,'prev_close':prev_c,'move5':move5})
rdf = pd.DataFrame(round_events)
rdf = rdf.drop_duplicates(subset=['date','level'])
print("\n=== ROUND NUMBER (nearest 100) REACTION ===")
print(f"Approach events (deduped by day+level): {len(rdf)}")
if len(rdf):
    bounce = (rdf['move5'].abs() < 8).sum()
    through = (rdf['move5'].abs() >= 15).sum()
    print(f"Stalls/chop within 8pts after 5min: {bounce} ({bounce/len(rdf)*100:.1f}%)")
    print(f"Decisive move >=15pts within 5min: {through} ({through/len(rdf)*100:.1f}%)")

# ---------- 4. Post-lunch reversal (1:00-2:00 pm) ----------
lunch_events = []
for d in days:
    dd = daily[d]
    morn = dd[(dd['datetime'].dt.time >= pd.Timestamp('09:15').time()) & (dd['datetime'].dt.time < pd.Timestamp('13:00').time())]
    if len(morn) < 2: continue
    morn_dir = morn.iloc[-1]['close'] - morn.iloc[0]['open']
    aft = dd[(dd['datetime'].dt.time >= pd.Timestamp('13:00').time()) & (dd['datetime'].dt.time < pd.Timestamp('14:00').time())]
    if len(aft) < 2: continue
    aft_dir = aft.iloc[-1]['close'] - aft.iloc[0]['open']
    reversal = (np.sign(morn_dir) != np.sign(aft_dir)) and abs(morn_dir)>5 and abs(aft_dir)>3
    lunch_events.append({'date':d,'morn_dir':morn_dir,'aft_dir':aft_dir,'reversal':reversal})
ldf = pd.DataFrame(lunch_events)
print("\n=== POST-LUNCH REVERSAL (13:00-14:00 vs morning 9:15-13:00) ===")
print(f"Days analyzed: {len(ldf)}, reversal rate: {ldf['reversal'].mean()*100:.1f}%")

# ---------- 5. Last hour momentum continuation ----------
lasthr_events = []
for d in days:
    dd = daily[d]
    mid = dd[(dd['datetime'].dt.time >= pd.Timestamp('14:15').time()) & (dd['datetime'].dt.time < pd.Timestamp('15:00').time())]
    if len(mid) < 2: continue
    mid_dir = mid.iloc[-1]['close'] - mid.iloc[0]['open']
    last = dd[(dd['datetime'].dt.time >= pd.Timestamp('15:00').time())]
    if len(last) < 2: continue
    last_dir = last.iloc[-1]['close'] - last.iloc[0]['open']
    cont = (np.sign(mid_dir)==np.sign(last_dir)) and abs(mid_dir)>5
    lasthr_events.append({'date':d,'mid_dir':mid_dir,'last_dir':last_dir,'cont':cont, 'has_signal':abs(mid_dir)>5})
hdf = pd.DataFrame(lasthr_events)
sig = hdf[hdf['has_signal']]
print("\n=== LAST-HOUR MOMENTUM CONTINUATION (14:15-15:00 -> 15:00-15:30) ===")
print(f"Days with clear mid-afternoon direction (>5pt move): {len(sig)} / {len(hdf)}")
print(f"Continuation rate: {sig['cont'].mean()*100:.1f}%")

# ---------- 6. Daily inside/outside bars ----------
daily_ohlc = df.groupby('date').agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last')).reset_index()
daily_ohlc = daily_ohlc.sort_values('date').reset_index(drop=True)
inside_days = 0
outside_days = 0
inside_next_range = []
for i in range(1, len(daily_ohlc)):
    prev = daily_ohlc.iloc[i-1]
    cur = daily_ohlc.iloc[i]
    if cur['high'] <= prev['high'] and cur['low'] >= prev['low']:
        inside_days += 1
        if i+1 < len(daily_ohlc):
            nxt = daily_ohlc.iloc[i+1]
            brk_up = nxt['high'] > cur['high']
            brk_down = nxt['low'] < cur['low']
            inside_next_range.append({'brk_up':brk_up,'brk_down':brk_down})
    elif cur['high'] >= prev['high'] and cur['low'] <= prev['low']:
        outside_days += 1
print("\n=== DAILY INSIDE/OUTSIDE BARS ===")
print(f"Inside days: {inside_days} / {len(daily_ohlc)-1} ({inside_days/(len(daily_ohlc)-1)*100:.1f}%)")
print(f"Outside days: {outside_days} / {len(daily_ohlc)-1} ({outside_days/(len(daily_ohlc)-1)*100:.1f}%)")
if inside_next_range:
    indf = pd.DataFrame(inside_next_range)
    print(f"After inside day: breaks up {indf['brk_up'].mean()*100:.1f}%, breaks down {indf['brk_down'].mean()*100:.1f}%, both {((indf['brk_up'])&(indf['brk_down'])).mean()*100:.1f}%")

# ---------- 7. First 5-min candle direction predicts day direction ----------
first5_events = []
for d in days:
    dd = daily[d]
    first5 = dd[(dd['datetime'].dt.time >= pd.Timestamp('09:15').time()) & (dd['datetime'].dt.time < pd.Timestamp('09:20').time())]
    if len(first5)<2: continue
    f_dir = first5.iloc[-1]['close'] - first5.iloc[0]['open']
    day_dir = dd.iloc[-1]['close'] - dd.iloc[0]['open']
    if abs(f_dir) < 3: continue
    match = np.sign(f_dir)==np.sign(day_dir)
    first5_events.append({'date':d,'match':match})
fdf = pd.DataFrame(first5_events)
print("\n=== FIRST 5-MIN CANDLE DIRECTION -> EOD DIRECTION ===")
print(f"Days with clear first-5min direction: {len(fdf)} / {len(days)}")
print(f"Match rate with EOD direction: {fdf['match'].mean()*100:.1f}%")
