# tickers_providers.py
import pandas as pd
import os

from fallback_helsinki import get_fallback_tickers as helsinki
from fallback_sp500 import get_fallback_tickers as sp500
from fallback_stockholm import get_fallback_tickers as stockholm
from fallback_nasdaq_us import get_fallback_tickers as nasdaq_us

PROVIDERS = {
    "helsinki": helsinki,
    "sp500": sp500,
    "stockholm": stockholm,
    "nasdaq_us": nasdaq_us,
}

def load_tickers_or_fallback(csv_paths: list[str], default_provider="helsinki") -> pd.DataFrame:
    for p in csv_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    # jos ei löydy CSV:tä, käytä oletus-provideria
    return PROVIDERS[default_provider]()
