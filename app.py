# app.py — Streamlit + yfinance (batch-lataus, 3 kuvaajaa/rivi, interval-valitsin, YTD-fix)

import time
import random
import unicodedata
from datetime import date, timedelta
from math import erf, sqrt
from functools import lru_cache
from typing import List, Tuple, Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

st.set_page_config(page_title="Six Sigma Stock Plot", layout="wide")
st.title("Six Sigma Stock Plot (Streamlit + yfinance)")

# ---------------------------
# 0) Fallback-tickerlista (ei verkkohakuja)
# ---------------------------
try:
    # Jos sinulla on tämä tiedosto repossa, käytä sitä
    from fallback_all import get_fallback_tickers  # type: ignore
    df_tickers = get_fallback_tickers().copy()
except Exception:
    # Minimi sisäinen fallback
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

def fetch_period_for(selected: List[Tuple[str,str,str]]) -> str:
    """Valitse pisin tarvittava periodi annetulle listalle (symbol, period, interval)."""
    if not selected:
        return "1y"
    best = 1
    for _, per_key, _ in selected:
        best = max(best, PERIOD_ORDER.get(per_key.lower(), 5))
    # Palauta vastaava period-string; YTD:n kohdalla haetaan 1y ja leikataan myöhemmin
    for k,v in PERIOD_ORDER.items():
        if v == best:
            return "1y" if k == "ytd" else k

@st.cache_data(ttl=900, show_spinner=False)
def dl_batch(tickers: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Hakee kaikki tikkerit yhdellä pyynnöllä. Palauttaa dict: symbol -> DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    sess = yahoo_session()
    _period = period.lower()
    if _period == "ytd":
        _period = "1y"

    joined = " ".join(sorted(set(tickers)))
    time.sleep(0.5 + random.uniform(0.0, 0.4))  # pieni hengähdys

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
            # MultiIndex: taso 0 = kenttä (Close...), taso 1 = ticker
            for t in sorted(set(tickers)):
                try:
                    sub = df.xs(t, level=1, axis=1, drop_level=False)
                    sub = sub.droplevel(1, axis=1)
                    out[t] = _normalize_cols(sub)
                except Exception:
                    pass
        else:
            if len(set(tickers)) == 1:
                out[tickers[0]] = _normalize_cols(df)
            else:
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
            time.sleep(0.8 + random.uniform(0.0, 0.4))
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

def cut_series_to_period(s: pd.Series, per_key: str) -> pd.Series:
    """Leikkaa sarjan loppupää halutun periodin pituuteen (paitsi max)."""
    if s.empty:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.to_datetime(s.index)

    per = per_key.lower()
    if per == "max":
        return s
    if per == "ytd":
        return ytd_slice(s)

    days_map = {
        "1mo": 31, "3mo": 93, "6mo": 186,
        "1y": 365, "2y": 2*365, "5y": 5*365,
        "10y": 10*365, "15y": 15*365
    }
    n_days = days_map.get(per, 365)
    cutoff = s.index.max() - timedelta(days=n_days)
    return s.loc[s.index >= cutoff]

def normal_cdf(x: float) -> float:
    # ilman SciPyä
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

# ---------------------------
# 2) SIDEBAR
# ---------------------------
with st.sidebar:
    st.subheader("Valitse symboli ja aikajakso")

    # --- Haku ---
    q = st.text_input("Haku (aksentiton)", "")
    if q.strip():
        filtered = [
            (lab, val) for lab, val in ALL_OPTIONS
            if strip_accents(q) in strip_accents(lab) or strip_accents(q) in strip_accents(val)
        ]
        if not filtered:
            st.info("Ei osumia suodatuksella – näytetään kaikki.")
            filtered = ALL_OPTIONS
    else:
        filtered = ALL_OPTIONS

    # --- Symboli ---
    lab2sym = {lab: val for lab, val in filtered}
    choice = st.selectbox("Symboli", [lab for lab, _ in filtered], index=0)
    sym = lab2sym[choice]

    # --- Aikajakso ---
    period_map = {
        "1kk": "1mo",
        "3kk": "3mo",
        "6kk": "6mo",
        "YTD": "ytd",
        "1v": "1y",
        "2v": "2y",
        "5v": "5y",
        "10v": "10y",
        "15v": "15y",
        "MAX": "max",
    }
    period_label_ui = st.selectbox("Aikajakso", list(period_map.keys()), index=4)
    period_key = period_map[period_label_ui]

    # --- Aikasarja (resoluutio) ---
    interval = st.selectbox(
        "Aikasarja (resoluutio)",
        ["1d (päivä)", "1wk (viikko)", "1mo (kuukausi)"],
        index=2,  # oletuksena kuukausi (vähemmän 429-virheitä)
    )
    interval_map = {"1d (päivä)": "1d", "1wk (viikko)": "1wk", "1mo (kuukausi)": "1mo"}
    interval_key = interval_map[interval]

    # --- Napit ---
    col_a, col_b = st.columns([1, 1])
    if "selected_plots" not in st.session_state:
        # lista tupleja: (symbol, period_key, interval_key)
        st.session_state.selected_plots: List[Tuple[str,str,str]] = []

    if col_a.button("Lisää kuvaaja"):
        st.session_state.selected_plots.append((sym, period_key, interval_key))
    if col_b.button("Tyhjennä kaikki"):
        st.session_state.selected_plots = []

    st.caption("Vinkki: käytä kuukausitolppia (1mo) jos 429-virheitä ilmenee. Välimuisti 15 min.")

# ---------------------------
# 3) Piirto
# ---------------------------
def plot_one(ax, symbol: str, per_key: str, df: pd.DataFrame):
    if df.empty:
        ax.text(0.5,0.5,f"Ei dataa:\n{symbol}", ha="center", va="center")
        ax.axis("off"); return
    s = pick_price_series(df)
    if s is None or s.dropna().empty:
        ax.text(0.5,0.5,f"Ei hintasarjaa:\n{symbol}", ha="center", va="center")
        ax.axis("off"); return

    s = cut_series_to_period(s, per_key)
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

sel: List[Tuple[str,str,str]] = st.session_state.selected_plots
if not sel:
    st.info("Valitse symboli, aikajakso ja resoluutio sivupalkista, lisää kuvaaja.")
else:
    # Ryhmitellään valinnat intervalin mukaan -> yksi batch/interval
    by_interval: Dict[str, List[Tuple[str,str,str]]] = {}
    for item in sel:
        by_interval.setdefault(item[2], []).append(item)

    # Haetaan data jokaiselle interval-ryhmälle
    dfs_by_interval: Dict[str, Dict[str, pd.DataFrame]] = {}
    for interv, items in by_interval.items():
        fetch_period = fetch_period_for(items)
        uniq_syms = sorted({s for s,_,_ in items})
        with st.spinner(f"Haetaan dataa ({interv}): {', '.join(uniq_syms)} | periodi: {fetch_period}"):
            dfs_by_interval[interv] = dl_batch(uniq_syms, period=fetch_period, interval=interv)

    # Piirretään 3/rivi
    cols = 3
    rows = int(np.ceil(len(sel)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.2*rows), squeeze=False)
    for idx, (symbol, per_key, interv) in enumerate(sel):
        r, c = divmod(idx, cols)
        dfs = dfs_by_interval.get(interv, {})
        plot_one(axes[r][c], symbol, per_key, dfs.get(symbol, pd.DataFrame()))
        time.sleep(0.05)  # pieni hengähdys

    # piilota tyhjät ruudut
    for k in range(len(sel), rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Näytä viimeksi lisätyn taulukon häntä
    last_sym, last_per, last_interv = sel[-1]
    dflast = dfs_by_interval.get(last_interv, {}).get(last_sym, pd.DataFrame())
    if not dflast.empty:
        st.dataframe(dflast.tail().reset_index(), use_container_width=True, hide_index=True)
