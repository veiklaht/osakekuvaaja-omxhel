# app.py — Minimal 1-call smoke test (1mo, exactly one value)

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

st.set_page_config(page_title="Yahoo 1mo Smoke Test", layout="centered")
st.title("Yahoo 1mo Smoke Test (single call)")

# --- optional ADR/OTC alias for Helsinki tickers ---
ALIAS_MAP = {
    "NOKIA.HE": ["NOK"],      # Nokia ADR
    "ELISA.HE": ["ELMUF"],    # Elisa OTC
    "SAMPO.HE": ["SAXPF","SAXPY"],
    "KNEBV.HE": ["KNYJF"],
}

@st.cache_resource
def yahoo_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, connect=2, read=2, status=3,
        backoff_factor=0.8,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET","POST"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    s.mount("http://", adapter); s.mount("https://", adapter)
    s.headers.update({"User-Agent":"Mozilla/5.0"})
    return s

def pick_price_series(df: pd.DataFrame) -> pd.Series | None:
    low = {c.lower(): c for c in df.columns}
    if "adj close" in low: return df[low["adj close"]]
    if "close" in low:     return df[low["close"]]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): return df[c]
    return None

def one_call_1mo_last_value(symbol: str) -> tuple[pd.DataFrame, float | None, str]:
    """
    Tee tasan yksi kutsu yfinanceen (period='1mo', interval='1mo').
    Palauta (df, value, used_symbol). Jos df tyhjä, value = None.
    """
    sess = yahoo_session()
    # EXACTLY ONE CALL:
    df = yf.download(
        symbol,
        period="1mo",
        interval="1mo",
        auto_adjust=False,
        progress=False,
        threads=False,
        session=sess,
        timeout=20,
        group_by="ticker",
    )
    if df is None or df.empty:
        return pd.DataFrame(), None, symbol

    # Jos MultiIndex -> normalisoi
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in col if str(c)!=""]) for col in df.columns]

    s = pick_price_series(df)
    if s is None or s.dropna().empty:
        return df, None, symbol

    # Tästä saadaan tasan yksi arvo
    return df, float(s.dropna().iloc[-1]), symbol

st.write("Testaa ensin **NOKIA.HE**. Sovellus tekee **vain yhden** 1mo-haun.")
symbol = st.text_input("Symboli", "NOKIA.HE")
run = st.button("Hae (1 kutsu)")

if run:
    st.info("Suoritetaan tasan yksi 1mo-kutsu...")
    df, value, used = one_call_1mo_last_value(symbol.strip())
    if value is not None:
        st.success(f"Onnistui: {used} → viimeisin kuukausi = **{value}**")
        st.dataframe(df.tail().reset_index(), use_container_width=True, hide_index=True)
    else:
        st.warning(f"Ei dataa symbolille {used} (1 kutsu). Kokeillaan alias...")
        alias_list = ALIAS_MAP.get(symbol.strip().upper(), [])
        tried_any = False
        for alias in alias_list:
            st.info(f"Alias-yritys: {alias} (taas vain yksi 1mo-kutsu)")
            df2, value2, used2 = one_call_1mo_last_value(alias)
            tried_any = True
            if value2 is not None:
                st.success(f"Alias onnistui: {alias} → viimeisin kuukausi = **{value2}** "
                           f"(mapataan alkuperäiseen {symbol.strip()})")
                st.dataframe(df2.tail().reset_index(), use_container_width=True, hide_index=True)
                break
        else:
            if tried_any:
                st.error("Alias-yrityksetkin palauttivat tyhjää / 429. "
                         "Tämä vahvistaa, että ongelma on Yahoo-päässä (IP-rate limit).")
            else:
                st.error("Ei alias-reittiä tälle symbolille eikä dataa tullut yhdellä haulla.")
