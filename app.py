import math
import time as _time
from datetime import datetime, date, time as dtime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Nifty Options Dashboard", layout="wide", page_icon="📈")

BASE = "https://api.dhan.co/v2"
NIFTY_ID, NIFTY_SEG = 13, "IDX_I"
IST = "Asia/Kolkata"
RATE = 0.07  # risk-free rate for Black-Scholes


# ----------------------------------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

default_cid = st.secrets.get("DHAN_CLIENT_ID", "") if hasattr(st, "secrets") else ""
default_tok = st.secrets.get("DHAN_ACCESS_TOKEN", "") if hasattr(st, "secrets") else ""
CLIENT_ID = st.sidebar.text_input("CLIENT_ID", value=default_cid)
ACCESS_TOKEN = st.sidebar.text_input("ACCESS_TOKEN", value=default_tok, type="password")

st.sidebar.markdown("### Candle range")
sel_date = st.sidebar.date_input("Date", value=date.today())
CANDLE_START = st.sidebar.time_input("CANDLE_START", value=dtime(9, 15), step=300)
CANDLE_END = st.sidebar.time_input("CANDLE_END", value=dtime(15, 30), step=300)
INTERVAL = st.sidebar.selectbox("Candle interval (min)", [1, 5, 15], index=1)

st.sidebar.markdown("### Contract")
EXPIRY_NUMBER = st.sidebar.number_input("EXPIRY_NUMBER (1 = nearest)", 1, 8, 1)
LOT_SIZE = st.sidebar.number_input("Lot size", 1, 1000, 65)
N_STRIKES = st.sidebar.slider("Strikes each side of ATM", 1, 6, 2)

st.sidebar.markdown("### Money (₹)")
BUDGET_RS = st.sidebar.number_input("BUDGET_RS", 1000, 10_000_000, 20000, step=1000)
TARGET_RS = st.sidebar.number_input("TARGET_RS", 100, 10_000_000, 3000, step=100)
MAX_LOSS_RS = st.sidebar.number_input("MAX_LOSS_RS", 100, 10_000_000, 1500, step=100)

st.sidebar.markdown("### Refresh")
auto = st.sidebar.checkbox("Auto refresh", value=False)
every = st.sidebar.number_input("Every (sec, min 5)", 5, 600, 10)
st.sidebar.button("🔄 Refresh now")

if not CLIENT_ID or not ACCESS_TOKEN:
    st.title("📈 Nifty Options Dashboard")
    st.info("Enter CLIENT_ID and ACCESS_TOKEN in the sidebar (or set them in Streamlit secrets).")
    st.stop()

HEAD = {"access-token": ACCESS_TOKEN, "client-id": CLIENT_ID, "Content-Type": "application/json"}


# ----------------------------------------------------------------------------
# DHAN API
# ----------------------------------------------------------------------------
def post(path, body):
    r = requests.post(BASE + path, headers=HEAD, json=body, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_expiries(cid, tok):
    j = post("/optionchain/expirylist", {"UnderlyingScrip": NIFTY_ID, "UnderlyingSeg": NIFTY_SEG})
    return j["data"]


@st.cache_data(ttl=4, show_spinner=False)
def get_chain(cid, tok, expiry):
    j = post("/optionchain", {"UnderlyingScrip": NIFTY_ID, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry})
    return j["data"]


def _frame(j):
    ts = pd.to_datetime(pd.Series(j["timestamp"]), unit="s", utc=True).dt.tz_convert(IST)
    return pd.DataFrame({"ts": ts, "open": j["open"], "high": j["high"], "low": j["low"],
                         "close": j["close"], "volume": j.get("volume", [0] * len(ts))})


@st.cache_data(ttl=20, show_spinner=False)
def get_intraday(cid, tok, frm, to, interval):
    j = post("/charts/intraday", {"securityId": str(NIFTY_ID), "exchangeSegment": NIFTY_SEG,
                                  "instrument": "INDEX", "interval": str(interval),
                                  "fromDate": frm, "toDate": to})
    return _frame(j)


@st.cache_data(ttl=3600, show_spinner=False)
def get_daily(cid, tok, frm, to):
    j = post("/charts/historical", {"securityId": str(NIFTY_ID), "exchangeSegment": NIFTY_SEG,
                                    "instrument": "INDEX", "expiryCode": 0,
                                    "fromDate": frm, "toDate": to})
    return _frame(j)


# ----------------------------------------------------------------------------
# MATH
# ----------------------------------------------------------------------------
def bs_price(S, K, T, sigma, cp):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if cp == "CE" else (K - S))
    d1 = (math.log(S / K) + (RATE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if cp == "CE":
        return S * norm.cdf(d1) - K * math.exp(-RATE * T) * norm.cdf(d2)
    return K * math.exp(-RATE * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, cp):
    try:
        return brentq(lambda s: bs_price(S, K, T, s, cp) - price, 1e-4, 5.0, xtol=1e-6)
    except Exception:
        return np.nan


def bs_greeks(S, K, T, sigma, cp):
    if T <= 0 or not sigma or np.isnan(sigma):
        return dict(delta=np.nan, gamma=np.nan, theta=np.nan, vega=np.nan)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (RATE + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * sq)
    vega = S * pdf * sq / 100  # per 1 vol point
    if cp == "CE":
        delta = norm.cdf(d1)
        th = -S * pdf * sigma / (2 * sq) - RATE * K * math.exp(-RATE * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        th = -S * pdf * sigma / (2 * sq) + RATE * K * math.exp(-RATE * T) * norm.cdf(-d2)
    return dict(delta=delta, gamma=gamma, theta=th / 252, vega=vega)  # theta per trading day


def trading_days_to_expiry(expiry: date, now: datetime):
    today = now.date()
    open_t, close_t = dtime(9, 15), dtime(15, 30)
    if today.weekday() >= 5:
        frac = 0.0
    elif now.time() < open_t:
        frac = 1.0
    elif now.time() > close_t:
        frac = 0.0
    else:
        mins = (datetime.combine(today, close_t) - now.replace(tzinfo=None)).total_seconds() / 60
        frac = mins / 375
    days = frac
    d = today + timedelta(days=1)
    while d <= expiry:
        if d.weekday() < 5:
            days += 1
        d += timedelta(days=1)
    return max(days, 0.02), (datetime.combine(expiry, close_t) - now.replace(tzinfo=None)).total_seconds() / 86400


def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def trade_cost(ltp, lot):
    v = ltp * lot
    brokerage = 40.0
    stt = 0.001 * v
    exch = 0.0003503 * 2 * v
    gst = 0.18 * (brokerage + exch)
    stamp = 0.00003 * v
    return brokerage + stt + exch + gst + stamp


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
st.title("📈 Nifty Options Dashboard")

try:
    expiries = get_expiries(CLIENT_ID, ACCESS_TOKEN)
    idx = min(int(EXPIRY_NUMBER) - 1, len(expiries) - 1)
    EXPIRY = expiries[idx]
    chain = get_chain(CLIENT_ID, ACCESS_TOKEN, EXPIRY)
except Exception as e:
    st.error(f"Dhan API error: {e}")
    st.stop()

now = datetime.now()
spot = float(chain["last_price"])
rows = []
for k, v in chain["oc"].items():
    K = float(k)
    for side, key in (("CE", "ce"), ("PE", "pe")):
        if key in v and v[key]:
            d = v[key]
            g = d.get("greeks", {}) or {}
            rows.append(dict(
                Strike=K, Type=side, LTP=d.get("last_price", 0), PrevClose=d.get("previous_close_price", 0),
                OI=d.get("oi", 0), PrevOI=d.get("previous_oi", 0), Vol=d.get("volume", 0),
                PrevVol=d.get("previous_volume", 0), IV_api=d.get("implied_volatility", np.nan),
                Delta_api=g.get("delta", np.nan), Bid=d.get("top_bid_price", 0), Ask=d.get("top_ask_price", 0),
                BidQty=d.get("top_bid_quantity", 0), AskQty=d.get("top_ask_quantity", 0)))
df = pd.DataFrame(rows)
strikes = sorted(df.Strike.unique())
step = np.median(np.diff(strikes))
atm = float(min(strikes, key=lambda x: abs(x - spot)))
T_days, T_cal = trading_days_to_expiry(datetime.strptime(EXPIRY, "%Y-%m-%d").date(), now)
T = T_days / 252
market_open = now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(15, 30)

# ---- candles
start_dt = datetime.combine(sel_date, CANDLE_START)
end_dt = datetime.combine(sel_date, CANDLE_END)
try:
    c = get_intraday(CLIENT_ID, ACCESS_TOKEN, start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                     end_dt.strftime("%Y-%m-%d %H:%M:%S"), INTERVAL)
except Exception as e:
    st.error(f"Candle fetch error: {e}")
    st.stop()
if c.empty:
    st.warning("No candles for that date/time range (holiday or market not open yet).")
    st.stop()

# previous-day levels + realised vol + 20 day average range
try:
    daily = get_daily(CLIENT_ID, ACCESS_TOKEN, (sel_date - timedelta(days=60)).strftime("%Y-%m-%d"),
                      sel_date.strftime("%Y-%m-%d"))
    daily["d"] = daily.ts.dt.date
    prev = daily[daily.d < sel_date].tail(21)
    pH, pL, pC = prev.iloc[-1][["high", "low", "close"]]
    rv = float(np.log(prev.close).diff().dropna().tail(20).std() * math.sqrt(252) * 100)
except Exception:
    pH = pL = pC = c.close.iloc[0]
    rv = np.nan

try:
    hist = get_intraday(CLIENT_ID, ACCESS_TOKEN, (sel_date - timedelta(days=35)).strftime("%Y-%m-%d 09:15:00"),
                        (sel_date - timedelta(days=1)).strftime("%Y-%m-%d 15:30:00"), INTERVAL)
    hist["d"] = hist.ts.dt.date
    ranges = []
    for d_, g in hist.groupby("d"):
        g = g[(g.ts.dt.time >= CANDLE_START) & (g.ts.dt.time <= CANDLE_END)]
        if len(g):
            ranges.append(g.high.max() - g.low.min())
    avg_range = float(np.mean(ranges[-20:])) if ranges else np.nan
except Exception:
    avg_range = np.nan

# ----------------------------------------------------------------------------
# CANDLE BLOCK
# ----------------------------------------------------------------------------
O, H, L, C = c.open.iloc[0], c.high.max(), c.low.min(), c.close.iloc[-1]
rng = H - L
body = abs(C - O)
body_pct = body / rng * 100 if rng else 0
close_pos = (C - L) / rng * 100 if rng else 50
upper_w = (H - max(O, C)) / rng * 100 if rng else 0
lower_w = (min(O, C) - L) / rng * 100 if rng else 0
co_pct = (C - O) / O * 100
if co_pct > 0.3 and body_pct > 50:
    candle_sig = "BULLISH"
elif co_pct < -0.3 and body_pct > 50:
    candle_sig = "BEARISH"
else:
    candle_sig = "SIDEWAYS"

# levels / trend
orb_n = max(1, 15 // INTERVAL)
orb_hi, orb_lo = c.high.iloc[:orb_n].max(), c.low.iloc[:orb_n].min()
orb_status = "ABOVE (breakout)" if spot > orb_hi else "BELOW (breakdown)" if spot < orb_lo else "INSIDE"
pivot = (pH + pL + pC) / 3
R1, S1 = 2 * pivot - pL, 2 * pivot - pH
R2, S2 = pivot + (pH - pL), pivot - (pH - pL)
ema9 = c.close.ewm(span=9, adjust=False).mean().iloc[-1]
ema21 = c.close.ewm(span=21, adjust=False).mean().iloc[-1]
tp = (c.high + c.low + c.close) / 3
vwap = (tp * c.volume.replace(0, 1)).cumsum().iloc[-1] / c.volume.replace(0, 1).cumsum().iloc[-1]
rsi_v = float(rsi(c.close).iloc[-1])
trend = "BULL" if (spot > ema9 > ema21 and spot > vwap) else "BEAR" if (spot < ema9 < ema21 and spot < vwap) else "MIXED"

# ----------------------------------------------------------------------------
# OPTION ANALYTICS
# ----------------------------------------------------------------------------
def leg(strike, side):
    r = df[(df.Strike == strike) & (df.Type == side)]
    return r.iloc[0] if len(r) else None


atm_ce, atm_pe = leg(atm, "CE"), leg(atm, "PE")
straddle = atm_ce.LTP + atm_pe.LTP
iv_atm = np.nanmean([implied_vol(atm_ce.LTP, spot, atm, T, "CE"), implied_vol(atm_pe.LTP, spot, atm, T, "PE")])
vol_prem = iv_atm * 100 - rv if not np.isnan(rv) else np.nan
imp_1d = spot * iv_atm / math.sqrt(252)
window_min = (end_dt - start_dt).total_seconds() / 60
imp_window = imp_1d * math.sqrt(min(window_min, 375) / 375)

band = [atm + i * step for i in range(-N_STRIKES, N_STRIKES + 1)]
band_df = df[df.Strike.isin(band)].copy()
spot_chg = spot - pC
days_elapsed = 1.0

recs = []
for _, r in band_df.sort_values(["Strike", "Type"]).iterrows():
    iv = implied_vol(r.LTP, spot, r.Strike, T, r.Type) if r.LTP > 0 else np.nan
    g = bs_greeks(spot, r.Strike, T, iv, r.Type)
    theo_atm = bs_price(spot, r.Strike, T, iv_atm, r.Type)
    theo_rv = bs_price(spot, r.Strike, T, rv / 100, r.Type) if not np.isnan(rv) else np.nan
    theo_api = bs_price(spot, r.Strike, T, r.IV_api / 100, r.Type) if r.IV_api and r.IV_api > 0 else np.nan
    pxchg = (r.LTP - r.PrevClose) - g["delta"] * spot_chg - g["theta"] * days_elapsed if r.PrevClose else 0.0
    oichg = r.OI - r.PrevOI
    thr = max(0.5, 0.01 * r.LTP)
    if oichg > 0:
        act = "LONG BUILDUP" if pxchg > thr else "SHORT BUILDUP" if pxchg < -thr else "OI UP (flat px)"
    else:
        act = "SHORT COVERING" if pxchg > thr else "LONG UNWINDING" if pxchg < -thr else "OI DOWN (flat px)"
    spread = (r.Ask - r.Bid) if r.Ask and r.Bid else np.nan
    cost = trade_cost(r.LTP, LOT_SIZE)
    be_pts = cost / LOT_SIZE
    mis = lambda th: (r.LTP - th) / th * 100 if th and not np.isnan(th) else np.nan
    lab = lambda m: "n/a" if np.isnan(m) else "OVERPRICED" if m > 3 else "UNDERPRICED" if m < -3 else "FAIR"
    recs.append(dict(
        Strike=int(r.Strike), Type=r.Type, LTP=r.LTP, Chg_pct=(r.LTP / r.PrevClose - 1) * 100 if r.PrevClose else np.nan,
        PxChg=pxchg, Vol=r.Vol, OI=r.OI, OIChg=oichg, OIChg_pct=oichg / r.PrevOI * 100 if r.PrevOI else np.nan,
        Vol_OI=r.Vol / r.OI if r.OI else np.nan, BidAskImb=(r.BidQty - r.AskQty) / (r.BidQty + r.AskQty) if (r.BidQty + r.AskQty) else 0,
        Action=act, IV_api=r.IV_api, IV_BS=iv * 100, Delta_api=r.Delta_api, Delta_BS=g["delta"], Gamma=g["gamma"],
        Theta_day=g["theta"], Th10m=g["theta"] / 37.5, Vega=g["vega"], G_T_e4=abs(g["gamma"] / g["theta"]) * 1e4 if g["theta"] else np.nan,
        Theo_apiIV=theo_api, Theo_ATMiv=theo_atm, Theo_RV=theo_rv,
        Mis_ATM=mis(theo_atm), Mis_RV=mis(theo_rv), Label_ATM=lab(mis(theo_atm)), Label_RV=lab(mis(theo_rv)),
        Spread=spread, Liq="WIDE" if (not np.isnan(spread) and spread / max(r.LTP, 1) > 0.005) else "OK",
        N_p5=r.LTP + g["delta"] * 5 + 0.5 * g["gamma"] * 25, N_m5=r.LTP - g["delta"] * 5 + 0.5 * g["gamma"] * 25,
        N_p10=r.LTP + g["delta"] * 10 + 0.5 * g["gamma"] * 100, N_m10=r.LTP - g["delta"] * 10 + 0.5 * g["gamma"] * 100,
        N_p20=r.LTP + g["delta"] * 20 + 0.5 * g["gamma"] * 400, N_m20=r.LTP - g["delta"] * 20 + 0.5 * g["gamma"] * 400,
        CostRs=cost, BE_prem_pts=be_pts, BE_nifty_pts=be_pts / abs(g["delta"]) if g["delta"] else np.nan))
opt = pd.DataFrame(recs)

# relative volume
med_v, med_vo = opt.Vol.median(), opt.Vol_OI.median()
opt["RelVol"] = opt.Vol / med_v
opt["VolOIRel"] = opt.Vol_OI / med_vo
opt["VolConf"] = (opt.RelVol >= 1.5) & (opt.VolOIRel >= 1.0)
def vol_read(r):
    if not r.VolConf:
        return "quiet"
    if r.Action in ("LONG BUILDUP", "SHORT COVERING"):
        return "STRONG BUY INTEREST"
    if r.Action in ("SHORT BUILDUP", "LONG UNWINDING"):
        return "HEAVY SELLING"
    return "active"
opt["VolRead"] = opt.apply(vol_read, axis=1)

# whole chain stats
ce_all, pe_all = df[df.Type == "CE"], df[df.Type == "PE"]
pcr_oi = pe_all.OI.sum() / ce_all.OI.sum()
pcr_vol = pe_all.Vol.sum() / max(ce_all.Vol.sum(), 1)
bnd = df[(df.Strike >= atm - 10 * step) & (df.Strike <= atm + 10 * step)]
pcr_vol_band = bnd[bnd.Type == "PE"].Vol.sum() / max(bnd[bnd.Type == "CE"].Vol.sum(), 1)
max_call_oi = ce_all.loc[ce_all.OI.idxmax(), "Strike"]
max_put_oi = pe_all.loc[pe_all.OI.idxmax(), "Strike"]
ce_add = ce_all.assign(d=ce_all.OI - ce_all.PrevOI)
pe_add = pe_all.assign(d=pe_all.OI - pe_all.PrevOI)
put_wall_df = pe_all[pe_all.Strike <= spot].sort_values("OI", ascending=False).head(1)
call_wall_df = ce_all[ce_all.Strike >= spot].sort_values("OI", ascending=False).head(1)
put_wall = put_wall_df.iloc[0] if len(put_wall_df) else None
call_wall = call_wall_df.iloc[0] if len(call_wall_df) else None

pain = {}
for K in strikes:
    pain[K] = (np.maximum(K - ce_all.Strike, 0) * ce_all.OI).sum() * -1 * -1 + (np.maximum(pe_all.Strike - K, 0) * pe_all.OI).sum()
max_pain = min(pain, key=pain.get)

# implied direction
fwd_parity = atm + math.exp(RATE * T) * (atm_ce.LTP - atm_pe.LTP)
fwd_fair = spot * math.exp(RATE * T)
fwd_diff = fwd_parity - fwd_fair
otm_put = [implied_vol(leg(atm - i * step, "PE").LTP, spot, atm - i * step, T, "PE") for i in range(1, 3) if leg(atm - i * step, "PE") is not None and leg(atm - i * step, "PE").LTP > 0]
otm_call = [implied_vol(leg(atm + i * step, "CE").LTP, spot, atm + i * step, T, "CE") for i in range(1, 3) if leg(atm + i * step, "CE") is not None and leg(atm + i * step, "CE").LTP > 0]
skew = (np.nanmean(otm_put) - np.nanmean(otm_call)) * 100 if otm_put and otm_call else 0.0

# ---- live flow (needs a previous snapshot)
snap = {(r.Strike, r.Type): (r.Vol, r.LTP) for r in df.itertuples()}
flow = None
if "prev_snap" in st.session_state and st.session_state.get("prev_expiry") == EXPIRY:
    ps = st.session_state["prev_snap"]
    buy = {"CE": 0.0, "PE": 0.0}
    sell = {"CE": 0.0, "PE": 0.0}
    for r in bnd.itertuples():
        pv, pl = ps.get((r.Strike, r.Type), (r.Vol, r.LTP))
        dv = r.Vol - pv
        if dv <= 0:
            continue
        mid = (r.Bid + r.Ask) / 2 if r.Bid and r.Ask else r.LTP
        side = 1 if r.LTP > mid else -1 if r.LTP < mid else (1 if r.LTP > pl else -1 if r.LTP < pl else 0)
        (buy if side > 0 else sell)[r.Type] += dv if side != 0 else 0
    flow = dict(buy=buy, sell=sell, bull=buy["CE"] + sell["PE"], bear=buy["PE"] + sell["CE"])
if "straddle_hist" not in st.session_state or st.session_state.get("prev_expiry") != EXPIRY:
    st.session_state["straddle_hist"] = []
st.session_state["straddle_hist"].append((datetime.now().strftime("%H:%M:%S"), straddle, iv_atm * 100))
st.session_state["straddle_hist"] = st.session_state["straddle_hist"][-300:]
st.session_state["prev_snap"], st.session_state["prev_expiry"] = snap, EXPIRY

# ----------------------------------------------------------------------------
# SCORE
# ----------------------------------------------------------------------------
cap = lambda x, m: max(-m, min(m, x))
sc = {}
sc["Opening candle"] = 1 if candle_sig == "BULLISH" else -1 if candle_sig == "BEARISH" else 0
sc["Close-Open %"] = 1 if co_pct > 0.3 else -1 if co_pct < -0.3 else 0
sc["ORB (confirmed)"] = 1 if spot > orb_hi else -1 if spot < orb_lo else 0
sc["Nifty EMA+VWAP trend"] = 1 if trend == "BULL" else -1 if trend == "BEAR" else 0
sc["Nifty RSI"] = 1 if rsi_v > 60 else -1 if rsi_v < 40 else 0
price_s = cap(sum(sc.values()), 2)

oi_sc = {}
oi_sc["PCR (OI)"] = 1 if pcr_oi > 1.2 else -1 if pcr_oi < 0.8 else 0
ce_d, pe_d = ce_add.d.sum(), pe_add.d.sum()
oi_sc["OI change (PE vs CE)"] = 1 if pe_d > 1.2 * ce_d and pe_d > 0 else -1 if ce_d > 1.2 * pe_d and ce_d > 0 else 0
dp = abs(spot - put_wall.Strike) if put_wall is not None else 1e9
dc = abs(call_wall.Strike - spot) if call_wall is not None else 1e9
oi_sc["Spot vs near OI walls"] = 1 if dp < 0.5 * dc else -1 if dc < 0.5 * dp else 0
def bias(r):
    a, t = r.Action, r.Type
    if t == "CE":
        return 1 if a in ("LONG BUILDUP", "SHORT COVERING") else -1 if a in ("SHORT BUILDUP", "LONG UNWINDING") else 0
    return 1 if a in ("SHORT BUILDUP", "LONG UNWINDING") else -1 if a in ("LONG BUILDUP", "SHORT COVERING") else 0
opt["Bias"] = opt.apply(bias, axis=1)
nb = opt.Bias.sum()
oi_sc["Buildup bias"] = 1 if nb >= 2 else -1 if nb <= -2 else 0
oi_s = cap(sum(oi_sc.values()), 2)

v_sc = {}
v_sc["Volume PCR (band)"] = 1 if pcr_vol_band > 1.2 else -1 if pcr_vol_band < 0.8 else 0
vb = opt[opt.VolConf].Bias.sum()
v_sc["Volume-confirmed buildup"] = 1 if vb >= 2 else -1 if vb <= -2 else 0
v_sc["Aggressor flow (live)"] = 0 if not flow else (1 if flow["bull"] > 1.3 * flow["bear"] else -1 if flow["bear"] > 1.3 * flow["bull"] else 0)
vol_s = cap(sum(v_sc.values()), 2)

i_sc = {}
i_sc["Parity forward vs fair"] = 1 if fwd_diff / spot * 100 > 0.1 else -1 if fwd_diff / spot * 100 < -0.1 else 0
i_sc["IV skew (put - call)"] = -1 if skew > 1.5 else 1 if skew < -1.5 else 0
imp_s = cap(sum(i_sc.values()), 1)

total = price_s + oi_s + vol_s + imp_s
signal = "BULLISH" if total >= 4 else "BEARISH" if total <= -4 else "SIDEWAYS"
icon = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS": "🟡"}[signal]

# ----------------------------------------------------------------------------
# DISPLAY
# ----------------------------------------------------------------------------
hdr = f"Expiry {EXPIRY} | Spot {spot:.2f} | ATM {int(atm)} | T = {T_days:.2f} trading days ({T_cal:.2f} calendar) | {now:%H:%M:%S} IST | {'LIVE' if market_open else 'CLOSED'}"
st.caption(hdr)
if not market_open:
    st.warning("MARKET CLOSED: this is a stale snapshot. Signals and the trade block are illustrative only.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Signal", f"{icon} {signal}", f"score {total:+d} / ±7")
k2.metric("Spot", f"{spot:.2f}", f"{spot - pC:+.2f} vs prev close")
k3.metric("ATM straddle", f"{straddle:.1f}", f"BE {atm - straddle:.0f} / {atm + straddle:.0f}")
k4.metric("ATM IV", f"{iv_atm * 100:.2f}%", f"RV {rv:.2f}%" if not np.isnan(rv) else "")
k5.metric("PCR (OI)", f"{pcr_oi:.2f}", f"Vol PCR {pcr_vol:.2f}")

t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
    ["🕯️ Candle & Levels", "📊 Vol & Straddle", "🔥 Price/Vol/OI", "🧮 Greeks & Pricing",
     "💧 Liquidity & Cost", "🧱 OI Walls & Direction", "🎯 Score", "💰 Trade Block"])

with t1:
    st.subheader(f"NIFTY {CANDLE_START:%H:%M}-{CANDLE_END:%H:%M} candle ({len(c)} bars of {INTERVAL}m) → {candle_sig}")
    a, b = st.columns(2)
    a.dataframe(pd.DataFrame({
        "Metric": ["Open", "High", "Low", "Close", "High-Open %", "Low-Open %", "Close-Open %", "Range (pts)", "Body % of range",
                   "Close position % of range", "Upper wick %", "Lower wick %"],
        "Value": [f"{O:.1f}", f"{H:.2f}", f"{L:.1f}", f"{C:.2f}", f"{(H - O) / O * 100:+.3f}%", f"{(L - O) / O * 100:+.3f}%",
                  f"{co_pct:+.3f}%", f"{rng:.1f} ({rng / O * 100:.3f}%)", f"{body_pct:.0f}%", f"{close_pos:.0f}%",
                  f"{upper_w:.0f}%", f"{lower_w:.0f}%"]}), hide_index=True, use_container_width=True)
    nr = [x for x in [("R1", R1), ("R2", R2), ("S1", S1), ("S2", S2)]]
    above = sorted([(v, n) for n, v in nr + [("Pivot", pivot)] if v > spot])
    below = sorted([(v, n) for n, v in nr + [("Pivot", pivot)] if v <= spot], reverse=True)
    b.dataframe(pd.DataFrame({
        "Level": ["ORB high", "ORB low", "ORB status", "Next resistance", "Next support", "Prev day H/L/C", "Pivot",
                  "R1 / R2", "S1 / S2", "EMA9 / EMA21", "VWAP", "RSI", "Trend", "VWAP distance"],
        "Value": [f"{orb_hi:.2f} ({orb_hi - spot:+.1f})", f"{orb_lo:.2f} ({orb_lo - spot:+.1f})", orb_status,
                  f"{above[0][1]} {above[0][0]:.1f} ({above[0][0] - spot:+.1f})" if above else "-",
                  f"{below[0][1]} {below[0][0]:.1f} ({below[0][0] - spot:+.1f})" if below else "-",
                  f"{pH:.1f} / {pL:.1f} / {pC:.1f}", f"{pivot:.1f}", f"{R1:.1f} / {R2:.1f}", f"{S1:.1f} / {S2:.1f}",
                  f"{ema9:.1f} / {ema21:.1f}", f"{vwap:.1f}", f"{rsi_v:.1f}", trend, f"{spot - vwap:+.1f} pts ({(spot - vwap) / vwap * 100:+.3f}%)"]}),
        hide_index=True, use_container_width=True)
    st.line_chart(c.set_index("ts")[["close"]])

with t2:
    a, b = st.columns(2)
    a.subheader("Volatility reference")
    a.write(f"ATM implied vol (from LTP, trading time): **{iv_atm * 100:.2f}%**")
    a.write(f"20-day realised vol: **{rv:.2f}%**")
    if not np.isnan(vol_prem):
        a.write(f"Vol premium (IV - RV): **{vol_prem:+.2f} pts** → " +
                ("options rich vs recent movement" if vol_prem > 2 else "options cheap vs recent movement" if vol_prem < -2 else "options roughly fairly priced vs recent movement"))
    a.write(f"Dhan IV minus BS IV (avg): **{(opt.IV_api - opt.IV_BS).mean():+.2f} vol pts**")
    b.subheader("Straddle + expected move")
    b.write(f"ATM straddle **{straddle:.1f}** pts → breakevens **{atm - straddle:.0f} / {atm + straddle:.0f}**")
    b.write(f"Implied 1-day 1-SD move: **{imp_1d:.0f}** pts")
    ratio = (rng / imp_window) if imp_window else np.nan
    b.write(f"Selected range: actual {rng:.1f} pts | options-implied {imp_window:.1f} (x{ratio:.2f})")
    if not np.isnan(avg_range):
        r2 = rng / avg_range
        b.write(f"vs 20-session average same-window range {avg_range:.1f}: x{r2:.2f} → " +
                ("WIDE" if r2 > 1.3 else "NARROW" if r2 < 0.7 else "NORMAL"))
    h = st.session_state["straddle_hist"]
    st.subheader("Straddle + ATM IV trend (this session)")
    if len(h) < 2:
        st.info("Needs at least 2 snapshots. Turn on Auto refresh.")
    else:
        hd = pd.DataFrame(h, columns=["time", "Straddle", "ATM IV %"]).set_index("time")
        st.line_chart(hd["Straddle"])
        st.line_chart(hd["ATM IV %"])

with t3:
    st.subheader("Price / Volume / OI / Buy-Sell read")
    cols = ["Strike", "Type", "LTP", "Chg_pct", "PxChg", "Vol", "Vol_OI", "OI", "OIChg", "OIChg_pct", "BidAskImb", "Action"]
    st.dataframe(opt[cols].round(2), hide_index=True, use_container_width=True)
    st.subheader("Volume analysis")
    a, b, c3 = st.columns(3)
    a.metric("Total volume CE", f"{ce_all.Vol.sum():,.0f}")
    a.metric("Total volume PE", f"{pe_all.Vol.sum():,.0f}")
    b.metric("Volume PCR (all)", f"{pcr_vol:.2f}")
    b.metric("Volume PCR (ATM ±10)", f"{pcr_vol_band:.2f}")
    c3.write("**Top CE volume strikes:** " + ", ".join(f"{int(r.Strike)} ({r.Vol:,.0f})" for r in ce_all.nlargest(3, "Vol").itertuples()))
    c3.write("**Top PE volume strikes:** " + ", ".join(f"{int(r.Strike)} ({r.Vol:,.0f})" for r in pe_all.nlargest(3, "Vol").itertuples()))
    st.dataframe(opt[["Strike", "Type", "LTP", "PxChg", "Vol", "Vol_OI", "RelVol", "VolOIRel", "VolConf", "Action", "VolRead"]].round(2),
                 hide_index=True, use_container_width=True)
    st.subheader("Live volume flow (quote + tick rule)")
    if flow is None:
        st.info("First snapshot stored. Flow needs at least one more refresh (turn on Auto refresh).")
    else:
        st.write(f"CE buy {flow['buy']['CE']:,.0f} / sell {flow['sell']['CE']:,.0f} | PE buy {flow['buy']['PE']:,.0f} / sell {flow['sell']['PE']:,.0f}")
        st.write(f"Bullish flow {flow['bull']:,.0f} vs bearish flow {flow['bear']:,.0f}")

with t4:
    st.subheader("Black-Scholes Greeks (IV backed out of LTP)")
    st.dataframe(opt[["Strike", "Type", "LTP", "IV_api", "IV_BS", "Delta_api", "Delta_BS", "Gamma", "Theta_day", "Th10m", "Vega", "G_T_e4"]].round(4),
                 hide_index=True, use_container_width=True)
    st.subheader("Over / under pricing vs Black-Scholes")
    st.caption("Mis% = (LTP - model)/model. Trust Label_ATM (self-consistent). Label_RV compares with 20-day realised vol.")
    st.dataframe(opt[["Strike", "Type", "LTP", "Theo_apiIV", "Theo_ATMiv", "Theo_RV", "Mis_ATM", "Mis_RV", "Label_ATM", "Label_RV"]].round(2),
                 hide_index=True, use_container_width=True)
    for nm, col in (("ATM IV", "Label_ATM"), ("realised vol", "Label_RV")):
        vc = opt[col].value_counts()
        st.write(f"vs {nm}: {vc.get('OVERPRICED', 0)} overpriced, {vc.get('UNDERPRICED', 0)} underpriced, {len(opt) - vc.get('OVERPRICED', 0) - vc.get('UNDERPRICED', 0)} fair/n.a.")

with t5:
    st.subheader("Spread / liquidity + premium-per-point + cost breakeven")
    st.caption(f"Lot size {LOT_SIZE}. Cost is an estimate (brokerage ₹20x2, STT, exchange, GST, stamp). Verify with your broker.")
    st.dataframe(opt[["Strike", "Type", "LTP", "Delta_BS", "Spread", "Liq", "N_p5", "N_m5", "N_p10", "N_m10", "N_p20", "N_m20",
                      "CostRs", "BE_prem_pts", "BE_nifty_pts", "Th10m", "G_T_e4"]].round(2), hide_index=True, use_container_width=True)
    wide = [f"{r.Strike}{r.Type}" for r in opt.itertuples() if r.Liq == "WIDE"]
    if wide:
        st.warning("WIDE spread (hard to scalp): " + ", ".join(wide))

with t6:
    st.subheader("Option-implied direction")
    st.write(f"Put-call parity forward vs fair forward: **{fwd_diff:+.1f} pts ({fwd_diff / spot * 100:+.3f}%)** → " +
             ("puts rich vs calls (bearish / hedging)" if fwd_diff < 0 else "calls rich vs puts (bullish)"))
    st.write(f"IV skew (OTM put - OTM call): **{skew:+.2f} vol pts** → " + ("puts bid over calls (fear / downside hedging)" if skew > 0 else "calls bid over puts"))
    st.subheader("Whole chain: where is the max OI?")
    mc, mp = ce_add.loc[ce_add.d.idxmax(), "Strike"], pe_add.loc[pe_add.d.idxmax(), "Strike"]
    st.dataframe(pd.DataFrame({"Item": [
        "Max CALL OI (resistance wall)", "Max PUT OI (support wall)", "Nearest PUT wall", "Nearest CALL wall",
        "Max CALL OI added", "Max PUT OI added", "Max CALL volume", "Max PUT volume", "Max pain", "PCR (OI)", "PCR (Volume)",
        "Total OI change (vs prev day)"],
        "Value": [f"{int(max_call_oi)}", f"{int(max_put_oi)}",
                  f"{int(put_wall.Strike)} (OI {put_wall.OI:,.0f}, {spot - put_wall.Strike:.0f} pts below)" if put_wall is not None else "-",
                  f"{int(call_wall.Strike)} (OI {call_wall.OI:,.0f}, {call_wall.Strike - spot:.0f} pts above)" if call_wall is not None else "-",
                  f"{int(mc)}", f"{int(mp)}", f"{int(ce_all.loc[ce_all.Vol.idxmax(), 'Strike'])}", f"{int(pe_all.loc[pe_all.Vol.idxmax(), 'Strike'])}",
                  f"{int(max_pain)}", f"{pcr_oi:.2f}", f"{pcr_vol:.2f}", f"CE {ce_d:+,.0f} | PE {pe_d:+,.0f}"]}),
        hide_index=True, use_container_width=True)
    oi_plot = df.assign(OI_=df.OI)[(df.Strike >= atm - 15 * step) & (df.Strike <= atm + 15 * step)].pivot(index="Strike", columns="Type", values="OI")
    st.bar_chart(oi_plot)

with t7:
    st.subheader(f"SCORE (>= +4 bullish | <= -4 bearish | else sideways)")
    def block(name, d, capv, s):
        st.markdown(f"**{name}** raw {sum(d.values()):+d} → capped {s:+d} (cap ±{capv})")
        st.dataframe(pd.DataFrame({"Factor": d.keys(), "Score": d.values()}), hide_index=True, use_container_width=True)
    block("Price", sc, 2, price_s)
    block("OI", oi_sc, 2, cap(sum(oi_sc.values()), 2))
    block("Volume", v_sc, 2, vol_s)
    block("Implied", i_sc, 1, imp_s)
    st.success(f"TOTAL {total:+d} of ±7 → {icon} {signal}") if signal != "SIDEWAYS" else st.warning(f"TOTAL {total:+d} of ±7 → {icon} SIDEWAYS")

with t8:
    st.subheader("Trade block (rule-based, not advice)")
    if not market_open:
        st.caption("ILLUSTRATIVE: market not live")
    if signal == "SIDEWAYS":
        st.info("No trade: signal is SIDEWAYS")
    else:
        side = "CE" if signal == "BULLISH" else "PE"
        pick = opt[(opt.Type == side) & (opt.Strike == int(atm))].iloc[0]
        entry = float(pick.LTP)
        cost_per_lot = entry * LOT_SIZE
        lots = int(BUDGET_RS // cost_per_lot)
        if lots < 1:
            st.error(f"Budget ₹{BUDGET_RS:,} is below one lot (₹{cost_per_lot:,.0f}).")
        else:
            qty = lots * LOT_SIZE
            sl_pts = MAX_LOSS_RS / qty
            tg_pts = TARGET_RS / qty
            sl, tg = entry - sl_pts, entry + tg_pts
            dlt = abs(pick.Delta_BS)
            c1, c2, c3 = st.columns(3)
            c1.metric("Buy", f"NIFTY {EXPIRY} {int(atm)} {side}")
            c2.metric("Entry (LTP)", f"{entry:.2f}")
            c3.metric("Lots × size", f"{lots} × {LOT_SIZE} = {qty}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Capital used", f"₹{qty * entry:,.0f}", f"budget ₹{BUDGET_RS:,}")
            d2.metric("Stop-loss premium", f"{max(sl, 0.05):.2f}", f"-{sl_pts:.2f} pts = -₹{MAX_LOSS_RS:,}")
            d3.metric("Target premium", f"{tg:.2f}", f"+{tg_pts:.2f} pts = +₹{TARGET_RS:,}")
            d4.metric("Reward : Risk", f"{TARGET_RS / MAX_LOSS_RS:.2f}")
            st.write(f"Approx. Nifty move needed → target **{tg_pts / dlt:.0f} pts**, stop **{sl_pts / dlt:.0f} pts** (delta {dlt:.2f}). "
                     f"Estimated round-trip cost ₹{trade_cost(entry, qty):,.0f}; cost breakeven ≈ {pick.BE_nifty_pts:.1f} Nifty pts.")
            if sl < 0:
                st.warning("Stop-loss is below zero premium, so your MAX_LOSS_RS exceeds capital at risk. The whole premium is the max loss.")
            if pick.Liq == "WIDE":
                st.warning("Wide spread on this strike, so expect slippage.")
            if pick.Mis_ATM > 3:
                st.warning("This option screens OVERPRICED vs ATM IV.")
    st.caption("Educational tool. Not investment advice. Options can lose the full premium.")

# ----------------------------------------------------------------------------
if auto:
    _time.sleep(int(every))
    st.rerun()
