import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="metric-container"] {
        background: #1e1e2e; border-radius: 10px; padding: 14px;
    }
    /* Mobile: stack dropdowns */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem 0.5rem; }
        div[data-testid="stDataFrame"] { font-size: 12px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_meta() -> pd.DataFrame:
    import requests, io
    url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    html    = requests.get(url, headers=headers, timeout=15).text
    df      = pd.read_html(io.StringIO(html))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={"Security": "Company", "GICS Sector": "Sector", "GICS Sub-Industry": "Industry"}
    )


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers: tuple) -> pd.DataFrame:
    return yf.download(
        list(tickers),
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )


def compute_one(ticker: str, raw, meta_row: dict) -> dict | None:
    try:
        df = raw[ticker].dropna(how="all").copy()
        if len(df) < 60:
            return None
        df.columns = [c.lower() for c in df.columns]
        close = df["close"]

        rsi_val   = RSIIndicator(close=close, window=14).rsi()
        macd_obj  = MACD(close=close)
        bb_obj    = BollingerBands(close=close, window=20, window_dev=2)
        sma50_s   = SMAIndicator(close=close, window=50).sma_indicator()
        sma200_s  = SMAIndicator(close=close, window=200).sma_indicator()

        price  = close.iloc[-1]
        score  = 0.0
        reasons: list[str] = []

        # RSI
        rsi = rsi_val.iloc[-1]
        if pd.notna(rsi):
            if rsi < 30:   score += 1;  reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70: score -= 1;  reasons.append(f"RSI overbought ({rsi:.1f})")

        # MACD
        mh  = macd_obj.macd_diff().iloc[-1]
        mhp = macd_obj.macd_diff().iloc[-2]
        if pd.notna(mh) and pd.notna(mhp):
            if   mh > 0 and mhp <= 0: score += 1;  reasons.append("MACD bullish cross")
            elif mh < 0 and mhp >= 0: score -= 1;  reasons.append("MACD bearish cross")
            elif mh > 0:              score += 0.5
            else:                     score -= 0.5

        # Bollinger Bands
        bbl = bb_obj.bollinger_lband().iloc[-1]
        bbu = bb_obj.bollinger_hband().iloc[-1]
        if pd.notna(bbl) and pd.notna(bbu):
            if   price < bbl: score += 1;  reasons.append("Below lower BB")
            elif price > bbu: score -= 1;  reasons.append("Above upper BB")

        # SMA 50 / 200
        sma50  = sma50_s.iloc[-1]
        sma200 = sma200_s.iloc[-1]
        if pd.notna(sma50) and pd.notna(sma200):
            if sma50 > sma200: score += 1;  reasons.append("Golden cross (50>200)")
            else:              score -= 1;  reasons.append("Death cross (50<200)")
        if pd.notna(sma50):
            score += 0.5 if price > sma50 else -0.5

        signal  = "BUY" if score >= 2 else ("SELL" if score <= -2 else "HOLD")
        chg1w   = (price / close.iloc[-6]  - 1) * 100 if len(close) >= 6  else np.nan
        chg1m   = (price / close.iloc[-22] - 1) * 100 if len(close) >= 22 else np.nan
        pct_from_sma50 = ((price - sma50) / sma50 * 100) if pd.notna(sma50) else np.nan

        return {
            "Ticker":       ticker,
            "Company":      meta_row.get("Company", ticker),
            "Sector":       meta_row.get("Sector",  "Unknown"),
            "Industry":     meta_row.get("Industry","Unknown"),
            "Price":        round(price, 2),
            "1W %":         round(chg1w, 2),
            "1M %":         round(chg1m, 2),
            "RSI":          round(rsi, 1) if pd.notna(rsi) else np.nan,
            "vs SMA50 %":   round(pct_from_sma50, 2),
            "Score":        round(score, 1),
            "Signal":       signal,
            "Reasons":      " | ".join(reasons),
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def run_screener() -> pd.DataFrame:
    meta     = get_sp500_meta()
    tickers  = tuple(meta["Symbol"].tolist())
    raw      = download_prices(tickers)
    meta_map = meta.set_index("Symbol").to_dict(orient="index")
    results  = [compute_one(t, raw, meta_map.get(t, {})) for t in tickers]
    results  = [r for r in results if r is not None]
    df       = pd.DataFrame(results)

    # ── Relative strength vs sector ───────────────────────────────────────────
    sector_median = df.groupby("Sector")["1M %"].transform("median")
    df["Rel Str %"] = (df["1M %"] - sector_median).round(2)

    # Adjust score: beating sector = +0.5, lagging = -0.5
    df["Score"] = (df["Score"] + df["Rel Str %"].apply(lambda x: 0.5 if x > 0 else -0.5)).round(1)

    # Add reason for relative strength
    def add_rel_reason(row):
        if row["Rel Str %"] > 2:
            return row["Reasons"] + " | Outperforming sector"
        if row["Rel Str %"] < -2:
            return row["Reasons"] + " | Underperforming sector"
        return row["Reasons"]
    df["Reasons"] = df.apply(add_rel_reason, axis=1)

    # Re-apply signal after score adjustment
    df["Signal"] = df["Score"].apply(lambda s: "BUY" if s >= 2 else ("SELL" if s <= -2 else "HOLD"))

    return df.sort_values("Score", ascending=False).reset_index(drop=True)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Stock Screener — S&P 500 📊")

# Refresh button
col_title, col_btn = st.columns([6, 1])
with col_btn:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()

with st.spinner("Downloading prices & computing signals (first run ~60s, then cached 1h) …"):
    df = run_screener()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  •  {len(df)} stocks scored")

# ── Metrics row ────────────────────────────────────────────────────────────────
n_buy  = (df["Signal"] == "BUY").sum()
n_sell = (df["Signal"] == "SELL").sum()
n_hold = (df["Signal"] == "HOLD").sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("📈 BUY",  n_buy)
m2.metric("📉 SELL", n_sell)
m3.metric("📊 HOLD", n_hold)
m4.metric("🔢 Total", len(df))

st.divider()

# ── Dropdowns ─────────────────────────────────────────────────────────────────
d1, d2, d3 = st.columns([2, 2, 1])

with d1:
    sort_options = {
        "Most Expected to BUY (↑ Score)":  ("Score", False),
        "Most Expected to SELL (↓ Score)": ("Score", True),
        "Alphabetical (A → Z)":             ("Ticker", False),
        "RSI — Lowest (most oversold)":     ("RSI",   True),
        "RSI — Highest (most overbought)":  ("RSI",   False),
        "1-Week % Change ↑":               ("1W %",  False),
        "1-Week % Change ↓":               ("1W %",  True),
    }
    sort_label = st.selectbox("🔽 Sort by", list(sort_options.keys()))

with d2:
    sectors    = ["All Sectors"] + sorted(df["Sector"].dropna().unique().tolist())
    chosen_sec = st.selectbox("🏭 Filter by Sector / Industry", sectors)

with d3:
    signal_filter = st.selectbox("📡 Signal", ["All", "BUY", "SELL", "HOLD"])

# ── Apply filters ──────────────────────────────────────────────────────────────
view = df.copy()

if chosen_sec != "All Sectors":
    view = view[view["Sector"] == chosen_sec]

if signal_filter != "All":
    view = view[view["Signal"] == signal_filter]

sort_col, sort_asc = sort_options[sort_label]
view = view.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

# ── Display ────────────────────────────────────────────────────────────────────
display_cols = ["Ticker", "Company", "Sector", "Price", "1W %", "1M %", "Rel Str %", "RSI", "vs SMA50 %", "Score", "Signal", "Reasons"]

SIGNAL_EMOJI = {"BUY": "📈 BUY", "SELL": "📉 SELL", "HOLD": "📊 HOLD"}

def color_signal_col(val):
    if "BUY"  in str(val): return "background-color:#16a34a;color:white;font-weight:bold;font-size:14px"
    if "SELL" in str(val): return "background-color:#dc2626;color:white;font-weight:bold;font-size:14px"
    if "HOLD" in str(val): return "background-color:#d97706;color:white;font-weight:bold;font-size:14px"
    return ""

def color_score(val):
    if val >= 2:  return "color:#4ade80;font-weight:bold"
    if val <= -2: return "color:#f87171;font-weight:bold"
    return ""

def color_rsi(val):
    if pd.isna(val): return ""
    if val < 30: return "color:#4ade80;font-weight:bold"
    if val > 70: return "color:#f87171;font-weight:bold"
    return ""

def color_relstr(val):
    if pd.isna(val): return ""
    if val > 0: return "color:#4ade80"
    if val < 0: return "color:#f87171"
    return ""

# Map signal to emoji version just for display
display_view = view[display_cols].copy()
display_view["Signal"] = display_view["Signal"].map(SIGNAL_EMOJI).fillna(display_view["Signal"])

styled = (
    display_view.style
    .map(color_signal_col, subset=["Signal"])
    .map(color_score,      subset=["Score"])
    .map(color_rsi,        subset=["RSI"])
    .map(color_relstr,     subset=["Rel Str %"])
    .format({
        "Price":      "${:.2f}",
        "1W %":       "{:+.2f}%",
        "1M %":       "{:+.2f}%",
        "Rel Str %":  "{:+.2f}%",
        "RSI":        "{:.1f}",
        "vs SMA50 %": "{:+.1f}%",
        "Score":      "{:+.1f}",
    }, na_rep="—")
    .set_properties(**{"font-size": "13px"})
)

st.dataframe(styled, use_container_width=True, height=600)

# ── Industry breakdown (shown when a sector is selected) ───────────────────────
if chosen_sec != "All Sectors":
    st.subheader(f"Industries within {chosen_sec}")
    ind_summary = (
        view.groupby("Industry")["Signal"]
        .value_counts()
        .unstack(fill_value=0)
        .assign(Total=lambda x: x.sum(axis=1))
        .sort_values("Total", ascending=False)
    )
    st.dataframe(ind_summary, use_container_width=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.divider()
csv = view[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered results as CSV",
    data=csv,
    file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)
