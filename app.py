# @title
# --- Six Sigma Stock Plot + CSV + dropdown + hakukenttä (aksentiton) ---

# --- (valinnainen) Jos fallback-providerit ovat Google Drivessa, aja nämä ennen importteja ---
# from google.colab import drive
# drive.mount('/content/drive')
# import sys, os
# BASE_DIR = '/content/drive/MyDrive/Osakekuvaaja'  # <-- muuta polku omasi mukaan
# if BASE_DIR not in sys.path:
#     sys.path.append(BASE_DIR)

import os
import unicodedata
from functools import lru_cache

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ipywidgets import Dropdown, Text, HBox, VBox, Layout, Button, Output
from IPython.display import display

# ---------------- 0) Valitse fallback-providerin lähde ----------------
# Vaihda tähän haluamasi provider: fallback_all / fallback_helsinki / fallback_stockholm / fallback_sp500 / fallback_nasdaq_us
try:
    from fallback_allx import get_fallback_tickers  # yhdistetty fallback (suositus)
except Exception:
    # Jos yhdistettyä ei löydy, yritetään Helsingin fallbackia
    try:
        from fallback_helsinki import get_fallback_tickers
    except Exception:
        # Viimesijainen pieni sisäänrakennettu fallback (ettei kaadu kokonaan)
        def get_fallback_tickers() -> pd.DataFrame:
            rows = [
                {"symbol":"NOKIA.HE","name":"Nokia","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
                {"symbol":"SAMPO.HE","name":"Sampo","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
                {"symbol":"ELISA.HE","name":"Elisa","exchange":"Nasdaq Helsinki","country":"FI","asset_class":"Equity","currency":"EUR","segment":"Isot","notes":""},
                {"symbol":"AAPL","name":"Apple","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot","notes":""},
                {"symbol":"MSFT","name":"Microsoft","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot","notes":""},
            ]
            return pd.DataFrame(rows)

# ---------------- 1) Lataa CSV tai käytä provideriä ----------------

CSV_PATHS = ["/content/tickers_base.csv", "tickers_base.csv"]

def clean_tickers_df(df: pd.DataFrame) -> pd.DataFrame:
    """Korjaa yleiset CSV-ongelmat ja varmistaa minimi-sarakkeet."""
    # Jos koko rivi on yhdessä solussa, splitataan pilkulla
    if df.shape[1] == 1 and df.iloc[:, 0].astype(str).str.contains(",").any():
        parts = df.iloc[:, 0].astype(str).str.split(",", expand=True)
        header = parts.iloc[0].str.strip().str.lower()
        if {"symbol", "name"}.issubset(set(header)):
            parts.columns = header
            parts = parts.iloc[1:]
        else:
            parts.columns = [f"col{i+1}" for i in range(parts.shape[1])]
        df = parts

    # Poista mahdollinen header-rivi datasta ja trimmaa
    df.columns = [c.strip() for c in df.columns]
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.lower() != "symbol"]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # Varmista minimi-sarakkeet
    must = ["symbol","name","exchange","country","asset_class","currency"]
    for c in must:
        if c not in df.columns:
            df[c] = ""
    if "segment" not in df.columns:
        df["segment"] = ""
    if "notes" not in df.columns:
        df["notes"] = ""

    # Poista tyhjät ja selvästi rikki rivit
    df = df[df["symbol"].notna() & df["symbol"].str.len().gt(0)]
    df = df[~df["symbol"].str.contains(",")]
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return df

df_tickers = None
for p in CSV_PATHS:
    if os.path.exists(p):
        try:
            df_tickers = pd.read_csv(p)
            df_tickers = clean_tickers_df(df_tickers)
            break
        except Exception as e:
            print(f"[WARN] CSV-lataus epäonnistui {p}: {e}")

if df_tickers is None:
    # Ei CSV:tä → käytä fallback-provideriä
    df_tickers = get_fallback_tickers().copy()
    # varmista minimi-sarakkeet
    for c in ["name","exchange","country","asset_class","currency","segment","notes"]:
        if c not in df_tickers.columns:
            df_tickers[c] = ""

# Fallback-symbolit (vain lista) tarvittaessa
fallback_symbols = df_tickers["symbol"].tolist()

# ---------------- 2) Aikajaksot ----------------

period_options = {
    "1kk": "1mo",
    "3kk": "3mo",
    "6kk": "6mo",
    "YTD": "ytd",
    "1v": "1y",
    "5v": "5y",
    "10v": "10y",
    "15v": "15y"
}

# ---------------- 3) Aputyökalut ----------------

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s))
                   if not unicodedata.combining(c)).lower()

_SUFFIX_TO_EXCHANGE = {
    "HE": "Helsinki",
    "ST": "Stockholm",
    "CO": "Copenhagen",
    "OL": "Oslo",
    "DE": "Germany (Xetra/Frankfurt)",
    "PA": "Paris",
    "AS": "Amsterdam",
    "MI": "Milan",
    "L":  "London",
    "SW": "Switzerland",
    "TO": "Canada (TSX)",
    "V":  "Canada (TSXV)",
    "AX": "Australia",
    "HK": "Hong Kong",
    "SI": "Singapore",
    "KS": "Korea (KSE)",
    "KQ": "Korea (KOSDAQ)",
    "SA": "Brazil (B3)",
    "NZ": "New Zealand",
    "TA": "Tel Aviv",
    "SS": "Shanghai",
    "SZ": "Shenzhen",
    "BK": "Thailand",
    "JK": "Indonesia",
    "TW": "Taiwan",
    "T":  "Tokyo",
}

def infer_exchange_from_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        return "USA (NYSE/Nasdaq)"
    if "." in symbol:
        suf = symbol.rsplit(".", 1)[-1].upper()
        return _SUFFIX_TO_EXCHANGE.get(suf, f"Other (.{suf})")
    return "USA (NYSE/Nasdaq)"

def get_exchange_col(df: pd.DataFrame):
    for col in ["exchange", "mic", "MIC", "Exchange", "porssi", "market"]:
        if col in df.columns:
            return col
    return None

def available_exchanges() -> list:
    ex_list = []
    if df_tickers is not None and "symbol" in df_tickers.columns:
        exch_col = get_exchange_col(df_tickers)
        if exch_col:
            raw = df_tickers[exch_col].dropna().astype(str).str.strip()
            mic_map = {
                "XHEL": "Helsinki",
                "XSTO": "Stockholm",
                "XCSE": "Copenhagen",
                "XOSL": "Oslo",
                "XETR": "Germany (Xetra/Frankfurt)",
                "XFRA": "Germany (Frankfurt)",
                "XPAR": "Paris",
                "XAMS": "Amsterdam",
                "XMIL": "Milan",
                "XLON": "London",
                "XSWX": "Switzerland",
                "XTSE": "Canada (TSX)",
                "XTSX": "Canada (TSXV)",
                "XNAS": "USA (Nasdaq)",
                "XNYS": "USA (NYSE)",
                "XHKG": "Hong Kong",
                "XSES": "Singapore",
            }
            normed = raw.map(lambda x: mic_map.get(x.upper(), x))
            ex_list = sorted(set(normed.tolist()))
        else:
            ex_list = sorted(set(infer_exchange_from_symbol(s)
                                 for s in df_tickers["symbol"].dropna().astype(str)))
    else:
        ex_list = sorted(set(infer_exchange_from_symbol(s) for s in fallback_symbols))
    return ["Kaikki"] + ex_list

# ---------------- 4) Markkina-luokitus ----------------

market_size_dd = Dropdown(
    options=["Kaikki", "Isot", "Keskisuuret", "Pienet", "First North"],
    value="Kaikki",
    description='Markkina:',
    layout=Layout(width="240px")
)

@lru_cache(maxsize=256)
def get_market_cap(symbol: str) -> float | None:
    try:
        finfo = yf.Ticker(symbol).fast_info
        cap = getattr(finfo, "market_cap", None) if hasattr(finfo, "market_cap") else finfo.get("market_cap", None)
        return float(cap) if cap is not None else None
    except Exception:
        return None

def classify_market_cap(mcap: float | None) -> str:
    if mcap is None:
        return "Tuntematon"
    if mcap >= 5_000_000_000:
        return "Isot"
    elif mcap >= 1_000_000_000:
        return "Keskisuuret"
    else:
        return "Pienet"

def detect_market_segment_from_row(row: pd.Series) -> str:
    for c in ["segment", "market_segment", "market", "list", "listing"]:
        if c in row and isinstance(row[c], str) and "first north" in row[c].lower():
            return "First North"
    for c in ["exchange", "mic", "porssi"]:
        if c in row and isinstance(row[c], str) and "first north" in row[c].lower():
            return "First North"
    if "symbol" in row:
        cap = get_market_cap(str(row["symbol"]))
        return classify_market_cap(cap)
    return "Tuntematon"

# ---------------- 5) Optiolistan rakennus ----------------

def build_full_options(asset_class="Kaikki", exchange_choice="Kaikki", market_size_choice="Kaikki"):
    df = df_tickers.copy()

    # asset class -filtteri
    if asset_class != "Kaikki" and "asset_class" in df.columns:
        df = df[df["asset_class"] == asset_class]

    # pörssi-filtteri
    if exchange_choice != "Kaikki":
        exch_col = get_exchange_col(df)
        if exch_col:
            mask = df[exch_col].astype(str).str.lower().str.contains(exchange_choice.lower())
            df = df[mask]
        else:
            df = df[df["symbol"].astype(str).map(infer_exchange_from_symbol) == exchange_choice]

    # markkina/segmentti-filtteri
    if market_size_choice != "Kaikki":
        if "segment" in df.columns or "market_cap" in df.columns:
            if "market_cap" in df.columns:
                df["_detected_segment"] = df["market_cap"].apply(
                    lambda x: classify_market_cap(float(x)) if pd.notnull(x) else "Tuntematon")
            else:
                df["_detected_segment"] = df.apply(detect_market_segment_from_row, axis=1)
        else:
            df["_detected_segment"] = df["symbol"].apply(lambda s: classify_market_cap(get_market_cap(s)))
        df = df[df["_detected_segment"] == market_size_choice]

    # labelit dropdowniin
    if "name" in df.columns:
        labels = (df["name"].fillna(df["symbol"]) + " (" + df["symbol"] + ")")
    else:
        labels = df["symbol"]

    options = list(zip(labels.tolist(), df["symbol"].tolist()))

    # duplikaattien poisto + aksentiton sorttaus
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
    return [(lab, val) for lab, val in options
            if q in strip_accents(lab) or q in strip_accents(val)]

def label_to_symbol(label_or_symbol: str):
    txt = label_or_symbol.strip()
    if txt.endswith(")") and "(" in txt:
        return txt.split("(")[-1][:-1].strip()
    return txt

# ---------------- 6) Piirtologiikka ----------------

@lru_cache(maxsize=512)
def _download_price(symbol: str, period_key: str) -> pd.Series | None:
    try:
        data = yf.download(symbol, period=period_key, interval="1d",
                           auto_adjust=False, progress=False)
    except Exception:
        return None
    if data is None or data.empty:
        return None
    price = data.get("Adj Close", data.get("Close"))
    if isinstance(price, pd.DataFrame):
        price = price.squeeze()
    return price.dropna() if price is not None else None

def _plot_to_axes(ax, symbol: str, period_label: str, price: pd.Series):
    mean_price = float(price.mean())
    std_price = float(price.std(ddof=1)) if price.std(ddof=1) != 0 else 0.0
    last_price = float(price.iloc[-1])
    if std_price == 0:
        prob_tail = np.nan
    else:
        z_last = (last_price - mean_price) / std_price
        prob_tail = 2 * (1 - norm.cdf(abs(z_last)))
    pct_within = ((price.between(mean_price - std_price, mean_price + std_price))
                  .mean() * 100 if std_price != 0 else np.nan)

    ax.plot(price.index, price.values, linewidth=1.5, label=f"{symbol}")
    ax.axhline(mean_price, linestyle='--', color='green', label='Mean')
    for k, col in zip([1, 2, 3], ['blue', 'orange', 'red']):
        ax.axhline(mean_price + k*std_price, linestyle='--', color=col)
        ax.axhline(mean_price - k*std_price, linestyle=':', color=col)
    ax.scatter(price.index[-1], last_price, zorder=5)
    title = (f"{symbol} — {period_label}\nP(|Z|≥|zₙ|): {(prob_tail*100):.3f}% | % ±1σ: {pct_within:.1f}%"
             if std_price != 0 else f"{symbol} — {period_label}\nσ=0 (ei vaihtelua)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

# ---------------- 7) Widgetit (+ TraitError-turva) ----------------

asset_classes = ["Kaikki"] + (sorted(df_tickers["asset_class"].dropna().unique().tolist())
                              if "asset_class" in df_tickers.columns else [])
exchanges = available_exchanges()

asset_dd = Dropdown(options=asset_classes, value="Kaikki",
                    description='Asset class:', layout=Layout(width="240px"))
exchange_dd = Dropdown(options=exchanges, value="Kaikki",
                       description='Pörssi:', layout=Layout(width="240px"))
market_size_dd = market_size_dd  # jo luotu yllä


# Turvallinen alustus symboli-dropdownille
all_opts = build_full_options(asset_dd.value, exchange_dd.value, market_size_dd.value)
if not all_opts:
    all_opts = [("— ei tuloksia —", None)]

sym_dd = Dropdown(
    options=all_opts,
    value=(all_opts[0][1] if all_opts and all_opts[0][1] is not None else None),
    description='Symboli:',
    layout=Layout(width="460px")
)


search_txt = Text(value="", placeholder='Hae esim. "Metsä", "Nokia", "VWCE"...',
                  description='Haku:', layout=Layout(width="360px"))
period_dd = Dropdown(options=list(period_options.keys()), value='1v', description='Aikajakso:')

# Napit (luodaan ennen refresh-funktiota, koska sitä disabloidaan tarvittaessa)
add_btn = Button(description="Lisää kuvaaja", button_style="success", layout=Layout(width="150px"))
clear_btn = Button(description="Tyhjennä kaikki", button_style="danger", layout=Layout(width="150px"))

_selected_plots = []
grid_out = Output()

def _render_grid():
    with grid_out:
        grid_out.clear_output(wait=True)
        n = len(_selected_plots)
        if n == 0:
            display(pd.DataFrame({"Vinkki": ["Valitse pörssi/markkina, hae symboli ja paina 'Lisää kuvaaja'"]}))
            return
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.2*rows), squeeze=False)
        for idx, (sym, per_lbl) in enumerate(_selected_plots):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            price = _download_price(sym, period_options[per_lbl])
            if price is None or price.empty:
                ax.text(0.5, 0.5, f"Ei dataa:\n{sym} ({per_lbl})", ha="center", va="center")
                ax.axis("off")
                continue
            _plot_to_axes(ax, sym, per_lbl, price)
        # tyhjät ruudut pois näkyvistä
        for k in range(n, rows*cols):
            r, c = divmod(k, cols)
            axes[r][c].axis("off")
        fig.tight_layout()
        plt.show()

def refresh_symbol_options(*_):
    # Rakenna ja suodata lista
    base = build_full_options(asset_dd.value, exchange_dd.value, market_size_dd.value)
    filtered = filter_options(base, search_txt.value)

    # Päättele uusi value
    if not filtered:
        new_options = [("— ei tuloksia —", None)]
        new_value = None
        add_btn.disabled = True
    else:
        new_options = filtered
        current_value = sym_dd.value
        values = [v for _, v in new_options]
        new_value = current_value if current_value in values else values[0]
        add_btn.disabled = False

    # Päivitä atomisesti ilman TraitErroria
    try:
        # irrota mahdolliset value-observerit väliaikaisesti
        for cb in list(sym_dd._trait_notifiers.get('value', {}).get('change', [])):
            sym_dd.unobserve(cb['handler'], names='value')
    except Exception:
        pass

    sym_dd.value = None
    sym_dd.options = new_options
    sym_dd.value = new_value

    try:
        # kytke observerit takaisin
        for cb in list(sym_dd._trait_notifiers.get('value', {}).get('change', [])):
            sym_dd.observe(cb['handler'], names='value')
    except Exception:
        pass

# Päivityskoukut
asset_dd.observe(lambda _: refresh_symbol_options(), names='value')
exchange_dd.observe(lambda _: refresh_symbol_options(), names='value')
market_size_dd.observe(lambda _: refresh_symbol_options(), names='value')
search_txt.observe(lambda _: refresh_symbol_options(), names='value')

# Nappien toiminnot
def _add_plot_clicked(_):
    sym = sym_dd.value
    if sym is None:
        return
    _selected_plots.append((label_to_symbol(sym), period_dd.value))
    _render_grid()


def _clear_clicked(_):
    _selected_plots.clear()
    _render_grid()

add_btn.on_click(_add_plot_clicked)
clear_btn.on_click(_clear_clicked)

# Ensimmäinen täyttö ja UI
refresh_symbol_options()

ui = VBox([
    HBox([asset_dd, exchange_dd, market_size_dd]),
    HBox([sym_dd, period_dd, add_btn, clear_btn]),
    HBox([search_txt]),
    grid_out
])

display(ui)
_render_grid()
