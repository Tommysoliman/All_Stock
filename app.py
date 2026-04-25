import warnings
warnings.filterwarnings("ignore")

import io, threading, os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

app = Flask(__name__)

# ── EGX30 master list (hardcoded — no Wikipedia needed) ───────────────────────
EGX_META = {
    "COMI.CA":  ("Commercial International Bank", "Banking",            "Commercial Banking"),
    "HRHO.CA":  ("EFG Hermes Holding",            "Financial Services", "Investment Banking"),
    "SWDY.CA":  ("Elsewedy Electric",              "Industrials",        "Electrical Equipment"),
    "TMGH.CA":  ("Talaat Moustafa Group",          "Real Estate",        "Real Estate Development"),
    "ORWE.CA":  ("Oriental Weavers",               "Consumer Discretionary", "Textiles"),
    "ABUK.CA":  ("Abu Kir Fertilizers",            "Materials",          "Fertilizers"),
    "AMOC.CA":  ("Alexandria Mineral Oils",        "Energy",             "Oil & Gas"),
    "PHDC.CA":  ("Palm Hills Developments",        "Real Estate",        "Real Estate Development"),
    "EGTS.CA":  ("Egyptian Transport (EGYTRANS)",  "Industrials",        "Transportation"),
    "EFIH.CA":  ("EFG Hermes",                     "Financial Services", "Investment Banking"),
    "OCDI.CA":  ("Orascom Construction",           "Industrials",        "Construction"),
    "CIEB.CA":  ("CIB Egypt",                      "Banking",            "Commercial Banking"),
    "ALCN.CA":  ("Alexandria Container",           "Industrials",        "Ports & Transportation"),
    "SKPC.CA":  ("Sidi Kerir Petrochemicals",      "Materials",          "Petrochemicals"),
    "CLHO.CA":  ("Cleopatra Hospital",             "Healthcare",         "Healthcare Facilities"),
    "HELI.CA":  ("Heliopolis Housing",             "Real Estate",        "Real Estate Development"),
    "ISPH.CA":  ("Ibnsina Pharma",                 "Healthcare",         "Pharmaceuticals"),
    "JUFO.CA":  ("Juhayna Food Industries",        "Consumer Staples",   "Food & Beverages"),
    "RAYA.CA":  ("Raya Holding",                   "Technology",         "IT Services"),
    "MCQE.CA":  ("Misr Chemical Industries",       "Materials",          "Chemicals"),
    "SPMD.CA":  ("Speed Medical",                  "Healthcare",         "Healthcare Services"),
    "ACGC.CA":  ("Arab Cotton Ginning",            "Materials",          "Textiles & Fibers"),
    "SUGR.CA":  ("Delta Sugar",                    "Consumer Staples",   "Food & Beverages"),
    "POUL.CA":  ("Cairo Poultry",                  "Consumer Staples",   "Poultry & Meat"),
}

# ── State: one entry per universe ─────────────────────────────────────────────
_state = {
    "sp500": {"df": None, "status": "computing", "updated": None},
    "egx":   {"df": None, "status": "computing", "updated": None},
}
_locks = {
    "sp500": threading.Lock(),
    "egx":   threading.Lock(),
}


def get_sp500_meta() -> pd.DataFrame:
    url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    html    = requests.get(url, headers=headers, timeout=15).text
    df      = pd.read_html(io.StringIO(html))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={"Security": "Company", "GICS Sector": "Sector", "GICS Sub-Industry": "Industry"}
    )


def get_egx_meta() -> pd.DataFrame:
    rows = [{"Symbol": k, "Company": v[0], "Sector": v[1], "Industry": v[2]}
            for k, v in EGX_META.items()]
    return pd.DataFrame(rows)


def compute_one(ticker: str, raw, meta_row: dict, multi: bool = True):
    try:
        df = raw[ticker].dropna(how="all").copy() if multi else raw.dropna(how="all").copy()
        if len(df) < 30:
            return None
        df.columns = [c.lower() for c in df.columns]
        close = df["close"]

        rsi_val  = RSIIndicator(close=close, window=14).rsi()
        macd_obj = MACD(close=close)
        bb_obj   = BollingerBands(close=close, window=20, window_dev=2)
        sma50_s  = SMAIndicator(close=close, window=50).sma_indicator()
        sma200_s = SMAIndicator(close=close, window=min(200, len(close) - 1)).sma_indicator()

        price   = float(close.iloc[-1])
        score   = 0.0
        reasons = []

        rsi = float(rsi_val.iloc[-1])
        if not np.isnan(rsi):
            if rsi < 30:   score += 1;  reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70: score -= 1;  reasons.append(f"RSI overbought ({rsi:.1f})")

        mh  = float(macd_obj.macd_diff().iloc[-1])
        mhp = float(macd_obj.macd_diff().iloc[-2])
        if not (np.isnan(mh) or np.isnan(mhp)):
            if   mh > 0 and mhp <= 0: score += 1;  reasons.append("MACD bullish cross")
            elif mh < 0 and mhp >= 0: score -= 1;  reasons.append("MACD bearish cross")
            elif mh > 0:              score += 0.5
            else:                     score -= 0.5

        bbl = float(bb_obj.bollinger_lband().iloc[-1])
        bbu = float(bb_obj.bollinger_hband().iloc[-1])
        if not (np.isnan(bbl) or np.isnan(bbu)):
            if   price < bbl: score += 1;  reasons.append("Below lower BB")
            elif price > bbu: score -= 1;  reasons.append("Above upper BB")

        sma50  = float(sma50_s.iloc[-1])
        sma200 = float(sma200_s.iloc[-1])
        if not (np.isnan(sma50) or np.isnan(sma200)):
            if sma50 > sma200: score += 1;  reasons.append("Golden cross (50>200)")
            else:              score -= 1;  reasons.append("Death cross (50<200)")
        if not np.isnan(sma50):
            score += 0.5 if price > sma50 else -0.5

        signal = "BUY" if score >= 2 else ("SELL" if score <= -2 else "HOLD")
        chg1w  = round((price / float(close.iloc[-6])  - 1) * 100, 2) if len(close) >= 6  else None
        chg1m  = round((price / float(close.iloc[-22]) - 1) * 100, 2) if len(close) >= 22 else None
        vs50   = round((price - sma50) / sma50 * 100, 2) if not np.isnan(sma50) else None

        return {
            "ticker":   ticker,
            "company":  meta_row.get("Company", ticker),
            "sector":   meta_row.get("Sector",  "Unknown"),
            "industry": meta_row.get("Industry","Unknown"),
            "price":    round(price, 2),
            "chg1w":    chg1w,
            "chg1m":    chg1m,
            "rsi":      round(rsi, 1) if not np.isnan(rsi) else None,
            "vs_sma50": vs50,
            "score":    round(score, 1),
            "signal":   signal,
            "reasons":  " · ".join(reasons),
        }
    except Exception:
        return None


def _build_df(meta: pd.DataFrame) -> pd.DataFrame:
    tickers  = tuple(meta["Symbol"].tolist())
    raw      = yf.download(list(tickers), period="6mo", interval="1d",
                           group_by="ticker", auto_adjust=True,
                           threads=True, progress=False)
    multi    = len(tickers) > 1
    meta_map = meta.set_index("Symbol").to_dict(orient="index")
    results  = [compute_one(t, raw, meta_map.get(t, {}), multi) for t in tickers]
    results  = [r for r in results if r is not None]
    df       = pd.DataFrame(results)

    sector_med   = df.groupby("sector")["chg1m"].transform("median")
    df["rel_str"] = (df["chg1m"] - sector_med).round(2)
    df["score"]   = (df["score"] + df["rel_str"].apply(lambda x: 0.5 if x > 0 else -0.5)).round(1)

    def upd_reason(row):
        suffix = ""
        if row["rel_str"] and row["rel_str"] > 2:   suffix = " · Outperforming sector"
        elif row["rel_str"] and row["rel_str"] < -2: suffix = " · Underperforming sector"
        return (row["reasons"] or "") + suffix
    df["reasons"] = df.apply(upd_reason, axis=1)
    df["signal"]  = df["score"].apply(lambda s: "BUY" if s >= 2 else ("SELL" if s <= -2 else "HOLD"))
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def run_screener(universe: str):
    lock = _locks[universe]
    with lock:
        _state[universe]["status"] = "computing"
    try:
        meta = get_sp500_meta() if universe == "sp500" else get_egx_meta()
        df   = _build_df(meta)
        with lock:
            _state[universe]["df"]      = df
            _state[universe]["status"]  = "ready"
            _state[universe]["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        with lock:
            _state[universe]["status"] = f"error: {e}"


# Start both on startup
threading.Thread(target=run_screener, args=("sp500",), daemon=True).start()
threading.Thread(target=run_screener, args=("egx",),   daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    universe = request.args.get("universe", "sp500")
    if universe not in _state:
        return jsonify({"status": "error: unknown universe"}), 400

    lock = _locks[universe]
    with lock:
        status  = _state[universe]["status"]
        df      = _state[universe]["df"]
        updated = _state[universe]["updated"]

    if status == "ready" and df is not None:
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        sectors = sorted(df["sector"].dropna().unique().tolist())
        return jsonify({
            "status":  "ready",
            "data":    records,
            "sectors": sectors,
            "updated": updated,
            "universe": universe,
            "counts": {
                "buy":   int((df["signal"] == "BUY").sum()),
                "sell":  int((df["signal"] == "SELL").sum()),
                "hold":  int((df["signal"] == "HOLD").sum()),
                "total": len(df),
            },
        })
    return jsonify({"status": status, "universe": universe})


@app.route("/api/refresh", methods=["POST"])
def refresh():
    universe = request.args.get("universe", "sp500")
    if universe not in _state:
        return jsonify({"status": "error"}), 400
    lock = _locks[universe]
    with lock:
        if _state[universe]["status"] != "computing":
            _state[universe]["status"] = "computing"
            threading.Thread(target=run_screener, args=(universe,), daemon=True).start()
    return jsonify({"status": "computing"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
