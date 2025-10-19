# app.py — Single-month value via Finnhub (works on Streamlit Cloud free)

import os
import time
import math
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="1-month Smoke Test (Finnhub)", layout="centered")
st.title("1 kk viimeisin arvo (Finnhub) — yksi kutsu")

# ---- API key ----
TOKEN = st.secrets.get("FINNHUB_TOKEN") or os.getenv("FINNHUB_TOKEN", "")
if not TOKEN:
    st.warning(
        "Aseta FINNHUB_TOKEN:\n"
        "• Streamlit Cloud: Settings → Secrets → FINNHUB_TOKEN\n"
        "• Paikallisesti: export FINNHUB_TOKEN=... tai .env\n"
    )

# ---- Pieni symboliapu: kokeillaan suoraan pörssitunnusta ja tarvittaessa ADR:ää ----
ADR_MAP = {
    "NOKIA.HE": ["NOK"],        # Nokia ADR NYSE
    "ELISA.HE": ["ELMUF"],      # Elisa OTC
    "SAMPO.HE": ["SAXPF","SAXPY"],
    "KNEBV.HE": ["KNYJF"],
}

st.write("Tämä tekee **tasan yhden** haku-kutsun (kuukausiresoluutio).")
symbol = st.text_input("Symboli", "NOKIA.HE")
run = st.button("Hae (1 kutsu)")

# ---- Finnhub candle endpoint (M = monthly) ----
# Docs: https://finnhub.io/docs/api/stock-candles
def finnhub_month_last_close(sym: str, token: str) -> tuple[pd.DataFrame, float | None, str]:
    """
    Palauta (df, last_close, used_symbol).
    Tekee yhden haun: resolution=M, viimeiset ~3 kk varmuuden vuoksi.
    """
    if not token:
        return pd.DataFrame(), None, sym

    now = int(time.time())
    three_months = 93 * 24 * 3600
    params = {
        "symbol": sym,
        "resolution": "M",
        "from": now - three_months,
        "to": now,
        "token": token,
    }
    r = requests.get("https://finnhub.io/api/v1/stock/candle", params=params, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame(), None, sym
    data = r.json()
    if not data or data.get("s") != "ok":
        return pd.DataFrame(), None, sym

    # Rakennetaan DataFrame
    df = pd.DataFrame({
        "t": pd.to_datetime(data["t"], unit="s"),
        "o": data["o"],
        "h": data["h"],
        "l": data["l"],
        "c": data["c"],
        "v": data["v"],
    })
    df = df.sort_values("t").reset_index(drop=True)
    last_close = float(df["c"].iloc[-1]) if not df.empty else None
    return df, last_close, sym

def try_symbol_then_aliases(sym: str, token: str):
    # 1) Yritä pyydetty symboli
    df, val, used = finnhub_month_last_close(sym, token)
    if val is not None:
        return df, val, used
    # 2) Yritä aliaset (ADR/OTC)
    for alias in ADR_MAP.get(sym.upper(), []):
        df2, val2, used2 = finnhub_month_last_close(alias, token)
        if val2 is not None:
            return df2, val2, used2
    # 3) Ei löytynyt
    return pd.DataFrame(), None, sym

if run:
    if not TOKEN:
        st.stop()
    with st.spinner("Haetaan Finnhubista (M = monthly)…"):
        df, value, used = try_symbol_then_aliases(symbol.strip(), TOKEN)

    if value is not None:
        st.success(f"Onnistui: {used} → viimeisin kuukausi = **{value}**")
        st.dataframe(df.tail().reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.error(
            "Ei dataa tälle symbolille Finnhubista (myös alias-yritykset epäonnistuivat). "
            "Tarkista tikkeri
