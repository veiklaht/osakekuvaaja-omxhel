# @title
%%writefile fallback_all.py
import importlib
import pandas as pd

# Muokkaa tätä listaa sen mukaan, mitkä fallback-tiedostot sinulla on.
# Jokaisessa moduulissa täytyy olla funktio: get_fallback_tickers() -> pd.DataFrame
PROVIDER_MODULES = [
    "fallback_helsinki",
    "fallback_stockholm",
    "fallback_sp500",
    "fallback_nasdaq_us",
]

REQUIRED_COLS = ["symbol","name","exchange","country","asset_class","currency","segment","notes"]

def _safe_import(name: str):
    """Yritä importoida moduuli. Palauta (moduuli, None) tai (None, error_str)."""
    try:
        mod = importlib.import_module(name)
        return mod, None
    except Exception as e:
        return None, f"[WARN] import failed: {name}: {e}"

def _safe_fetch_df(mod):
    """Yritä kutsua get_fallback_tickers(). Palauta (df, None) tai (None, error_str)."""
    try:
        fn = getattr(mod, "get_fallback_tickers", None)
        if fn is None:
            return None, f"[WARN] module '{mod.__name__}' missing get_fallback_tickers()"
        df = fn()
        if not isinstance(df, pd.DataFrame):
            return None, f"[WARN] {mod.__name__}.get_fallback_tickers() did not return DataFrame"
        return df.copy(), None
    except Exception as e:
        return None, f"[WARN] provider failed: {mod.__name__}: {e}"

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Täydennä puuttuvat pakolliset sarakkeet ja siivoa perusasiat."""
    df = df.copy()
    # Trimmaa string-sarakkeet
    for c in df.columns:
        try:
            df[c] = df[c].astype(str).str.strip()
        except Exception:
            pass
    # Lisää puuttuvat pakolliset sarakkeet
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""
    # Poista tyhjät/virheelliset symbolit
    df = df[df["symbol"].astype(str).str.len() > 0]
    # Vakioi “segment” arvot kevyesti (ei pakollinen)
    seg_map = {
        "large": "Isot", "large-cap": "Isot", "megacap": "Isot",
        "mid": "Keskisuuret", "mid-cap": "Keskisuuret",
        "small": "Pienet", "small-cap": "Pienet",
        "first north": "First North",
    }
    df["segment"] = df["segment"].astype(str)
    df.loc[df["segment"].str.lower().isin(seg_map.keys()),
           "segment"] = df["segment"].str.lower().map(seg_map)
    return df

def get_fallback_tickers(providers: list[str] | None = None) -> pd.DataFrame:
    """
    Yhdistä useiden fallback-moduulien listat yhdeksi DataFrameksi.
    - providers: lista modulien nimiä. Jos None -> käytä PROVIDER_MODULES -listaa.
    """
    modules = providers if providers is not None else PROVIDER_MODULES
    frames = []
    warnings = []

    for name in modules:
        mod, err = _safe_import(name)
        if err:
            warnings.append(err)
            continue
        df, err = _safe_fetch_df(mod)
        if err:
            warnings.append(err)
            continue
        frames.append(_normalize_cols(df))

    if not frames:
        # Ei yhtään provideriä onnistunut → nosta virhe, mutta kerro syyt
        hints = "\n".join(warnings) if warnings else "No provider frames."
        raise SystemExit(f"[ERROR] No fallback providers succeeded.\n{hints}")

    out = pd.concat(frames, ignore_index=True)
    # Poista duplikaatti-tickerit, säilytä ensimmäinen (ensimmäisen providerin prioriteetti)
    out = out.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    # Lopuksi järjestä nimen mukaan (aksentteja huomioimatta olisi UI-taso, mutta tässä perus)
    if "name" in out.columns:
        out = out.sort_values("name").reset_index(drop=True)

    # Lisää vielä varmistus, että kaikki pakolliset sarakkeet ovat mukana
    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = ""

    # (Valinnainen) — voit halutessasi täyttää tyhjiä 'segment'-arvoja kokeilemalla yksinkertaista heuristiikkaa:
    # if (out["segment"] == "").any():
    #     out.loc[out["segment"] == "", "segment"] = "Pienet"

    return out
