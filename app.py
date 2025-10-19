import os, time, requests, pandas as pd, streamlit as st

st.set_page_config(page_title="Finnhub diag (1mo)", layout="centered")
st.title("Finnhub diagnostiikka — 1 kuukauden kynttilä (yksi kutsu)")

# --- Lue token ---
TOKEN = st.secrets.get("FINNHUB_TOKEN") or os.getenv("FINNHUB_TOKEN", "")
masked = (TOKEN[:4] + "…" + TOKEN[-4:]) if TOKEN else "(tyhjä)"
st.write(f"FINNHUB_TOKEN: **{masked}**")

if not TOKEN:
    st.error("Token puuttuu. Lisää se Secretsiin avaimella **FINNHUB_TOKEN**.")
    st.stop()

# --- Alias-apu ADR/OTC: ---
ADR_MAP = {
    "NOKIA.HE": ["NOK"],        # Nokia ADR NYSE
    "ELISA.HE": ["ELMUF"],      # Elisa OTC
    "SAMPO.HE": ["SAXPF","SAXPY"],
    "KNEBV.HE": ["KNYJF"],
}

# --- Suffix -> exchange Finnhubissa ---
EX_MAP = {
    "HE":"HE", "ST":"ST", "CO":"CO", "OL":"OL",
    "AS":"AS", "PA":"PA", "MI":"MI", "L":"L", "SW":"SW",
    "TO":"TO", "V":"V", "DE":"DE", "T":"T",
}

symbol_in = st.text_input("Symboli", "NOKIA.HE").strip()
run = st.button("Testaa")

def finnhub_get(url, params):
    r = requests.get(url, params=params, timeout=20)
    return r.status_code, (r.json() if "application/json" in r.headers.get("content-type","") else r.text)

def resolve_symbol(sym: str) -> str | None:
    """Vahvista symboli Finnhubin symbolilistasta (jos sisältää .SUFFIX)."""
    if "." not in sym:
        return sym  # US/ADR: käytä sellaisenaan (esim. AAPL, NOK)
    base, suf = sym.rsplit(".", 1)
    ex = EX_MAP.get(suf.upper())
    if not ex:
        return sym
    code, data = finnhub_get("https://finnhub.io/api/v1/stock/symbol",
                             {"exchange": ex, "token": TOKEN})
    st.write(f"Symbolilistan haku exchange={ex} → HTTP {code}")
    if code != 200 or not isinstance(data, list):
        return sym
    # Finhubin 'symbol' kenttä on muodossa esim. 'NOKIA.HE'
    wanted = sym.upper()
    for row in data:
        if isinstance(row, dict) and row.get("symbol","").upper() == wanted:
            return row["symbol"]
    # ei löytynyt listasta → palautetaan alkuperäinen
    return sym

def last_month_close(sym: str):
    now = int(time.time())
    three_months = 93 * 24 * 3600
    code, data = finnhub_get("https://finnhub.io/api/v1/stock/candle",
                             {"symbol": sym, "resolution":"M",
                              "from": now-three_months, "to": now, "token": TOKEN})
    return code, data

if run:
    # 1) varmista symboliformaatti
    resolved = resolve_symbol(symbol_in)
    st.write(f"Käytettävä symboli Finnhubille: **{resolved}**")

    # 2) yritä 1mo-kutsu
    code, data = last_month_close(resolved)
    st.write(f"Candles (M) → HTTP {code}")
    if isinstance(data, dict):
        st.write(f"Vastaus s-kenttä: **{data.get('s')}**")

    ok = isinstance(data, dict) and data.get("s") == "ok" and "c" in data and data["c"]
    if ok:
        df = pd.DataFrame({
            "t": pd.to_datetime(data["t"], unit="s"),
            "o": data["o"], "h": data["h"], "l": data["l"],
            "c": data["c"], "v": data["v"],
        }).sort_values("t")
        st.success(f"Onnistui: viimeisin kuukausi = **{float(df['c'].iloc[-1]):.4f}**")
        st.dataframe(df.tail(), use_container_width=True)
    else:
        st.warning("Ei dataa tällä symbolilla. Kokeillaan alias(-t), jos tunnettu HE/ST-lappu.")
        for alias in ADR_MAP.get(symbol_in.upper(), []):
            st.write(f"Alias-yritys: **{alias}**")
            code2, data2 = last_month_close(alias)
            st.write(f"Candles (M) alias → HTTP {code2} | s={getattr(data2,'get',lambda *_:None)('s') if isinstance(data2,dict) else '(ei json)'}")
            ok2 = isinstance(data2, dict) and data2.get("s") == "ok" and data2.get("c")
            if ok2:
                df2 = pd.DataFrame({
                    "t": pd.to_datetime(data2["t"], unit="s"),
                    "o": data2["o"], "h": data2["h"], "l": data2["l"],
                    "c": data2["c"], "v": data2["v"],
                }).sort_values("t")
                st.success(f"Alias **{alias}** onnistui: viimeisin kuukausi = **{float(df2['c'].iloc[-1]):.4f}**")
                st.dataframe(df2.tail(), use_container_width=True)
                break
        else:
            st.error("Ei dataa: symboli ei ehkä kuulu ilmaiseen datapakettiin TAI token on virheellinen/vanhentunut.")
            st.info("Varmista token: Settings → Secrets → FINNHUB_TOKEN (sama NIMI isolla). "
                    "Testaa myös puhdas US-tikkeri, esim. **AAPL**.")
