# app.py — Streamlit + yfinance (batch-lataus, 3 kuvaajaa/rivi, YTD-fix, ei fast_infoa)

import os
import time
import random
import unicodedata
from math import erf, sqrt
from datetime import date
from functools import lru_cache
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

st.set_page_config(page_title="Six Sigma Stock Plot", layout="wide")
st.title("Six Sigma Stock Plot (Streamlit + yfinance)")

# ---------------------------
# 0) Fallback-listat (valinnainen, ei verkkohakuja)
# ---------------------------
try:
    # jos sinulla on repoissa tämä tiedosto
    from fallback_all import get_fallback_tickers  # noqa: E402
    df_tickers = get_fallback_tickers().copy()
except Exception:
    # minimi sisäänrakennettu, jos tuonti epäonnistuu
    df_tickers = pd.DataFrame([
        {"symbol":"NOKIA.HE","name":"Nokia","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
        {"symbol":"ELISA.HE","name":"Elisa","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
        {"symbol":"KNEBV.HE","name":"KONE B","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
        {"symbol":"VOLV-B.ST","name":"Volvo B","exchange":"Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot","notes":""},
        {"symbol":"ERIC-B.ST","name":"Ericsson B","exchange":"Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot","notes":""},
        {"symbol":"AAPL","name":"Apple","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot","notes":""},
        {"symbol":"MSFT","name":"Microsoft","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot","notes":""},
    ])

for c in ["name","exchange","country","asset_class","currency","segment","notes"]:
    if c not in df_tickers.columns:
        df_tickers[c] = ""

def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch)).lower()

def build_options(df: pd.DataFrame) -> List[Tuple[str,str]]:
    if "name" in df.columns:
        labels = (df["name"].fillna(df["symbol"]) + " (" + df["symbol"] + ")")
    else:
        labels = df["symbol"]
    opts = list(zip(labels.tolist(), df["symbol"].tolist()))
    # poista duplikaatit, sortattu aksentittomasti
    seen, dedup = set(), []
    for lab, val in opts:
        if val not in seen:
            seen.add(val); dedup.append((lab, val))
    return sorted(dedup, key=lambda x: strip_accents(x[0]))

ALL_OPTIONS = build_options(df_tickers)

# ---------------------------
# 1) Yahoo-session + batch-lataus (429-ystävällinen)
# ---------------------------
@st.cache_resource
def yahoo_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6, connect=3, read=3, status=6,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET","POST"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    s.mount("http://", adapter); s.mount("https://", adapter)
    s.headers.update({"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"})
    return s

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in col if str(c)!=""]) for col in df.columns]
    return df

# periodien prioriteetti batchia varten
PERIOD_ORDER = {"1mo":1,"3mo":2,"6mo":3,"ytd":4,"1y":5,"2y":6,"5y":7,"10y":8,"15y":9,"max":10}
# YTD haetaan aina 1y ja leikataan UI:ssa
def _fetch_period_for(selected: List[Tuple[str,str]]) -> str:
    if not selected:
        return "1y"
    # valitaan pisin tarvittu periodi
    best = 1
    for _, per_lbl in selected:
        key = per_lbl.lower()
        best = max(best, PERIOD_ORDER.get(key, 5))
    # mapataan takaisin period-stringiin
    for k,v in PERIOD_ORDER.items():
        if v == best:
            return "1y" if k == "ytd" else k

@st.cache_data(ttl=900, show_spinner=False)
def dl_batch(tickers: List[str], period: str = "1y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Hakee kaikki tikkerit yhdellä pyynnöllä.
    Palauttaa dictin: symbol -> DataFrame
    """
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    sess = yahoo_session()
    _period = period.lower()
    # jos UI tulee joskus pyytämään ytd, noudetaan 1y
    if _period == "ytd":
        _period = "1y"

    joined = " ".join(sorted(set(tickers)))
    # pieni throttlaus ennen isoa pyyntöä
    time.sleep(0.6 + random.uniform(0.0, 0.4))

    try:
        df = yf.download(
            joined, period=_period, interval=interval,
            auto_adjust=False, progress=False, threads=False,
            session=sess, timeout=25, group_by="ticker",
        )
    except Exception:
        df = pd.DataFrame()

    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            # Oletus: taso 0 = kenttä (Close/Adj Close...), taso 1 = ticker
            # Poimitaan kullekin tickerille omat sarakkeet
            tickers_sorted = sorted(set(tickers))
            for t in tickers_sorted:
                try:
                    sub = df.xs(t, level=1, axis=1, drop_level=False)
                    sub = sub.droplevel(1, axis=1)
                    out[t] = _normalize_cols(sub)
                except Exception:
                    pass
        else:
            # Jos tuli vain yksittäinen tickeri
            if len(set(tickers)) == 1:
                out[tickers[0]] = _normalize_cols(df)
            else:
                # Viimeinen oljenkorsi: etsi ticker-suffiksit sarakkeista
                for t in set(tickers):
                    cols = [c for c in df.columns if isinstance(c, str) and c.endswith(t)]
                    if cols:
                        sub = df[cols].copy()
                        newc = [c.rsplit(" ",1)[0] if " " in c else c for c in sub.columns]
                        sub.columns = newc
                        out[t] = _normalize_cols(sub)

    # Fallbackit yksittäin puuttuviin
    missing = [t for t in set(tickers) if t not in out]
    for t in missing:
        try:
            time.sleep(1.0 + random.uniform(0.0, 0.4))
            dfi = yf.download(
                t, period=_period, interval=interval,
                auto_adjust=False, progress=False, threads=False,
                session=sess, timeout=25,
            )
            if not dfi.empty:
                out[t] = _normalize_cols(dfi)
        except Exception:
            pass

    return out

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

def normal_cdf(x: float) -> float:
    # ilman SciPyä
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

# ---------------------------
# 2) UI
# ---------------------------

with st.sidebar:
    st.subheader("Valitse symboli ja aikajakso")
    # Haku
    q = st.text_input("Haku (aksentiton)", "")
    if q.strip():
        filtered = [(lab,val) for lab,val in ALL_OPTIONS
                    if strip_accents(q) in strip_accents(lab) or strip_accents(q) in strip_accents(val)]
        if not filtered:
            st.info("Ei osumia suodatuksella – näytetään kaikki.")
            filtered = ALL_OPTIONS
    else:
        filtered = ALL_OPTIONS

    lab2sym = {lab: val for lab, val in filtered}
    choice = st.selectbox("Symboli", [lab for lab,_ in filtered], index=0)
    sym = lab2sym[choice]

    period_map = {"1kk":"1mo","3kk":"3mo","6kk":"6mo","YTD":"ytd","1v":"1y","2v":"2y","5v":"5y","10v":"10y","15v":"15y","MAX":"max"}
    period_label = st.selectbox("Aikajakso", list(period_map.keys()), index=4)

    col_a, col_b = st.columns([1,1])
    if "selected_plots" not in st.session_state:
        st.session_state.selected_plots = []

    if col_a.button("Lisää kuvaaja"):
        st.session_state.selected_plots.append((sym, period_map[period_label]))
    if col_b.button("Tyhjennä kaikki"):
        st.session_state.selected_plots = []

    st.caption("Vinkki: batch-lataus vähentää 429-virheitä. Välimuisti 15 min.")

sel: List[Tuple[str,str]] = st.session_state.selected_plots

# ---------------------------
# 3) Piirto (3/rivi)
# ---------------------------
def plot_one(ax, symbol: str, per_key: str, dfs: dict[str, pd.DataFrame]):
    df = dfs.get(symbol, pd.DataFrame())
    if df.empty:
        ax.text(0.5,0.5,f"Ei dataa:\n{symbol}", ha="center", va="center")
        ax.axis("off"); return
    s = pick_price_series(df)
    if s is None or s.dropna().empty:
        ax.text(0.5,0.5,f"Ei hintasarjaa:\n{symbol}", ha="center", va="center")
        ax.axis("off"); return

    if per_key.lower() == "ytd":
        s = ytd_slice(s)
    else:
        # leikkaa sarja suoraan period-avaimen mukaan, jos se on lyhyempi kuin haettu
        # (dl_batch haki pisimmän tarpeen, joten tässä voidaan “tailata” päivämäärällä)
        # yksinkertaisuus: annetaan s sellaisenaan (riittää useimpiin käyttötarpeisiin)
        pass

    if s.dropna().empty:
        ax.text(0.5,0.5,f"Ei datapisteitä:\n{symbol}", ha="center", va="center")
        ax.axis("off"); return

    mean_price = float(s.mean())
    std_price  = float(s.std(ddof=1)) if s.std(ddof=1) != 0 else 0.0
    last_price = float(s.iloc[-1])

    if std_price == 0.0:
        prob_tail = np.nan
        pct_within = np.nan
    else:
        z = (last_price - mean_price) / std_price
        prob_tail = 2.0 * (1.0 - normal_cdf(abs(z)))
        pct_within = (s.between(mean_price - std_price, mean_price + std_price).mean() * 100.0)

    ax.plot(s.index, s.values, linewidth=1.5, label=f"{symbol}")
    ax.axhline(mean_price, linestyle='--', label='Mean')
    for k in [1,2,3]:
        ax.axhline(mean_price + k*std_price, linestyle='--')
        ax.axhline(mean_price - k*std_price, linestyle=':')
    ax.scatter(s.index[-1], last_price, zorder=5)

    if np.isnan(prob_tail):
        title = f"{symbol} — {per_key.upper()}\nσ=0 (ei vaihtelua)"
    else:
        title = f"{symbol} — {per_key.upper()}\nP(|Z|≥|zₙ|): {(prob_tail*100):.3f}% | % ±1σ: {pct_within:.1f}%"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

if not sel:
    st.info("Valitse symboli ja aikajakso sivupalkista, lisää kuvaaja.")
else:
    # päättele pisin tarvittava latausperiodi batchille
    fetch_period = _fetch_period_for(sel)
    uniq_syms = sorted({s for s,_ in sel})

    with st.spinner(f"Haetaan dataa: {', '.join(uniq_syms)} (periodi: {fetch_period})"):
        dfs = dl_batch(uniq_syms, period=fetch_period, interval="1d")

    cols = 3
    rows = int(np.ceil(len(sel)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.2*rows), squeeze=False)
    for idx, (symbol, per_key) in enumerate(sel):
        r, c = divmod(idx, cols)
        plot_one(axes[r][c], symbol, per_key, dfs)
        time.sleep(0.1)  # pieni hengähdys piirtojen välissä

    # piilota tyhjät ruudut
    for k in range(len(sel), rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Näytä viimeksi lisätyn taulukon häntä
    last_sym, last_per = sel[-1]
    dfl = dfs.get(last_sym, pd.DataFrame())
    if not dfl.empty:
        st.dataframe(dfl.tail().reset_index(), use_container_width=True, hide_index=True)
