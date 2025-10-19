# app.py — Streamlit-versio (ei ipywidgetsia)

import os
import unicodedata
from functools import lru_cache
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Six Sigma Stock Plot", layout="wide")
st.title("Six Sigma Stock Plot (Streamlit + yfinance)")

# ---------- Fallback-listat ----------
try:
    from fallback_all import get_fallback_tickers  # HUOM: tiedosto fallback_all.py
except Exception:
    def get_fallback_tickers() -> pd.DataFrame:
        rows = [
            {"symbol":"NOKIA.HE","name":"Nokia","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
            {"symbol":"SAMPO.HE","name":"Sampo","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
            {"symbol":"ELISA.HE","name":"Elisa","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
            {"symbol":"AAPL","name":"Apple","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot","notes":""},
            {"symbol":"MSFT","name":"Microsoft","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot","notes":""},
        ]
        return pd.DataFrame(rows)

df_tickers = get_fallback_tickers().copy()
for c in ["name","exchange","country","asset_class","currency","segment","notes"]:
    if c not in df_tickers.columns:
        df_tickers[c] = ""

# ---------- Aput ----------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()

_SUFFIX_TO_EXCHANGE = {
    "HE":"Helsinki","ST":"Stockholm","CO":"Copenhagen","OL":"Oslo","DE":"Germany (Xetra/Frankfurt)",
    "PA":"Paris","AS":"Amsterdam","MI":"Milan","L":"London","SW":"Switzerland","TO":"Canada (TSX)",
    "V":"Canada (TSXV)","AX":"Australia","HK":"Hong Kong","SI":"Singapore","KS":"Korea (KSE)",
    "KQ":"Korea (KOSDAQ)","SA":"Brazil (B3)","NZ":"New Zealand","TA":"Tel Aviv","SS":"Shanghai",
    "SZ":"Shenzhen","BK":"Thailand","JK":"Indonesia","TW":"Taiwan","T":"Tokyo",
}

def infer_exchange_from_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        return "USA (NYSE/Nasdaq)"
    if "." in symbol:
        suf = symbol.rsplit(".", 1)[-1].upper()
        return _SUFFIX_TO_EXCHANGE.get(suf, f"Other (.{suf})")
    return "USA (NYSE/Nasdaq)"

def get_exchange_col(df: pd.DataFrame):
    for col in ["exchange","mic","MIC","Exchange","porssi","market"]:
        if col in df.columns:
            return col
    return None

def available_exchanges() -> list:
    ex_list = []
    if "symbol" in df_tickers.columns:
        exch_col = get_exchange_col(df_tickers)
        if exch_col:
            raw = df_tickers[exch_col].dropna().astype(str).str.strip()
            mic_map = {
                "XHEL":"Helsinki","XSTO":"Stockholm","XCSE":"Copenhagen","XOSL":"Oslo",
                "XETR":"Germany (Xetra/Frankfurt)","XFRA":"Germany (Frankfurt)","XPAR":"Paris",
                "XAMS":"Amsterdam","XMIL":"Milan","XLON":"London","XSWX":"Switzerland",
                "XTSE":"Canada (TSX)","XTSX":"Canada (TSXV)","XNAS":"USA (Nasdaq)","XNYS":"USA (NYSE)",
                "XHKG":"Hong Kong","XSES":"Singapore",
            }
            normed = raw.map(lambda x: mic_map.get(x.upper(), x))
            ex_list = sorted(set(normed.tolist()))
        else:
            ex_list = sorted(set(infer_exchange_from_symbol(s) for s in df_tickers["symbol"].dropna().astype(str)))
    return ["Kaikki"] + ex_list

@st.cache_data(ttl=600, show_spinner=False)
def dl(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in col if str(c)!=""]) for col in df.columns]
    return df

def pick_price_series(df: pd.DataFrame) -> pd.Series | None:
    low = {c.lower(): c for c in df.columns}
    if "adj close" in low: return df[low["adj close"]]
    if "close" in low:     return df[low["close"]]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): return df[c]
    return None

def ytd_slice(series: pd.Series) -> pd.Series:
    if series.empty: return series
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    start = pd.Timestamp(year=date.today().year, month=1, day=1, tz=series.index.tz)
    return series.loc[series.index >= start]

def classify_market_cap(mcap: float | None) -> str:
    if mcap is None: return "Tuntematon"
    if mcap >= 5_000_000_000: return "Isot"
    if mcap >= 1_000_000_000: return "Keskisuuret"
    return "Pienet"

@lru_cache(maxsize=256)
def get_market_cap(symbol: str) -> float | None:
    try:
        finfo = yf.Ticker(symbol).fast_info
        cap = getattr(finfo, "market_cap", None) if hasattr(finfo, "market_cap") else finfo.get("market_cap", None)
        return float(cap) if cap is not None else None
    except Exception:
        return None

def detect_market_segment_from_row(row: pd.Series) -> str:
    for c in ["segment","market_segment","market","list","listing","exchange","mic","porssi"]:
        if c in row and isinstance(row[c], str) and "first north" in row[c].lower():
            return "First North"
    if "symbol" in row:
        return classify_market_cap(get_market_cap(str(row["symbol"])))
    return "Tuntematon"

def build_full_options(asset_class="Kaikki", exchange_choice="Kaikki", market_size_choice="Kaikki"):
    df = df_tickers.copy()

    if asset_class != "Kaikki" and "asset_class" in df.columns:
        df = df[df["asset_class"] == asset_class]

    if exchange_choice != "Kaikki":
        exch_col = get_exchange_col(df)
        if exch_col:
            mask = df[exch_col].astype(str).str.lower().str.contains(exchange_choice.lower())
            df = df[mask]
        else:
            df = df[df["symbol"].astype(str).map(infer_exchange_from_symbol) == exchange_choice]

    if market_size_choice != "Kaikki":
        if "segment" in df.columns:
            df["_detected_segment"] = df["segment"].apply(lambda x: "First North" if isinstance(x,str) and "first north" in x.lower() else None)
        df["_detected_segment"] = df.get("_detected_segment") \
            .fillna(df.apply(detect_market_segment_from_row, axis=1))
        df = df[df["_detected_segment"] == market_size_choice]

    if "name" in df.columns:
        labels = (df["name"].fillna(df["symbol"]) + " (" + df["symbol"] + ")")
    else:
        labels = df["symbol"]
    options = list(zip(labels.tolist(), df["symbol"].tolist()))

    seen, dedup = set(), []
    for lab, val in options:
        if val not in seen:
            seen.add(val)
            dedup.append((lab, val))
    return sorted(dedup, key=lambda x: strip_accents(x[0]))

def filter_options(options, query: str):
    if not query.strip():
        return options
    q = strip_accents(query)
    return [(lab, val) for lab, val in options if q in strip_accents(lab) or q in strip_accents(val)]

# ---------- Sidebar UI ----------
with st.sidebar:
    st.subheader("Suodattimet")
    asset_classes = ["Kaikki"] + (sorted(df_tickers["asset_class"].dropna().unique().tolist()) if "asset_class" in df_tickers.columns else [])
    exchanges = available_exchanges()

    asset_dd = st.selectbox("Asset class", asset_classes, index=0)
    exchange_dd = st.selectbox("Pörssi", exchanges, index=0)
    market_size_dd = st.selectbox("Markkina", ["Kaikki","Isot","Keskisuuret","Pienet","First North"], index=0)
    search_txt = st.text_input("Haku (aksentiton)", "")

period_map = {"1kk":"1mo","3kk":"3mo","6kk":"6mo","YTD":"ytd","1v":"1y","5v":"5y","10v":"10y","15v":"15y"}
period_label = st.selectbox("Aikajakso", list(period_map.keys()), index=4)

base_opts = build_full_options(asset_dd, exchange_dd, market_size_dd)
opts = filter_options(base_opts, search_txt)
if not opts:
    st.info("Ei tuloksia. Muuta suodattimia tai hakua.")
    st.stop()

lab2sym = {lab: val for lab, val in opts}
choice = st.selectbox("Valitse symboli", [lab for lab,_ in opts], index=0)
sym = lab2sym[choice]

col_add, col_clear = st.columns([1,1])
if "selected_plots" not in st.session_state:
    st.session_state.selected_plots = []

if col_add.button("Lisää kuvaaja"):
    st.session_state.selected_plots.append((sym, period_label))
if col_clear.button("Tyhjennä kaikki"):
    st.session_state.selected_plots = []

# ---------- Piirto ----------
def plot_one(ax, symbol: str, period_label: str):
    per = period_map[period_label]
    load_period = "1y" if per == "ytd" else per
    df = dl(symbol, period=load_period, interval="1d")
    if df.empty:
        ax.text(0.5,0.5,f"Ei dataa:\n{symbol} ({period_label})", ha="center", va="center")
        ax.axis("off")
        return
    serie = pick_price_series(df)
    if serie is None or serie.dropna().empty:
        ax.text(0.5,0.5,f"Ei hintasarjaa:\n{symbol}", ha="center", va="center")
        ax.axis("off")
        return
    if per == "ytd":
        serie = ytd_slice(serie)

    mean_price = float(serie.mean())
    std_price  = float(serie.std(ddof=1)) if serie.std(ddof=1) != 0 else 0.0
    last_price = float(serie.iloc[-1])

    if std_price == 0:
        prob_tail = np.nan
        pct_within = np.nan
    else:
        z_last = (last_price - mean_price) / std_price
        prob_tail = 2 * (1 - norm.cdf(abs(z_last)))
        pct_within = (serie.between(mean_price - std_price, mean_price + std_price).mean() * 100)

    ax.plot(serie.index, serie.values, linewidth=1.5, label=f"{symbol}")
    ax.axhline(mean_price, linestyle='--', label='Mean')
    for k in [1,2,3]:
        ax.axhline(mean_price + k*std_price, linestyle='--')
        ax.axhline(mean_price - k*std_price, linestyle=':')
    ax.scatter(serie.index[-1], last_price, zorder=5)
    title = (f"{symbol} — {period_label}\nP(|Z|≥|zₙ|): {(prob_tail*100):.3f}% | % ±1σ: {pct_within:.1f}%"
             if std_price != 0 else f"{symbol} — {period_label}\nσ=0 (ei vaihtelua)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

sel = st.session_state.selected_plots
if not sel:
    st.info("Valitse symboli ja paina **Lisää kuvaaja**.")
else:
    cols = 3
    rows = int(np.ceil(len(sel)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.2*rows), squeeze=False)
    for idx, (s, per_lbl) in enumerate(sel):
        r, c = divmod(idx, cols)
        plot_one(axes[r][c], s, per_lbl)
    for k in range(len(sel), rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# Pieni taulu viimeisistä datapisteistä viimeiselle lisätylle
if sel:
    last_sym, last_per = sel[-1]
    df_last = dl(last_sym, period=("1y" if period_map[last_per]=="ytd" else period_map[last_per]), interval="1d")
    if not df_last.empty:
        st.dataframe(df_last.tail().reset_index(), use_container_width=True, hide_index=True)
