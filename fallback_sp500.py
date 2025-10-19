# fallback_sp500.py
import pandas as pd

def get_fallback_tickers() -> pd.DataFrame:
    rows = [
        # --- Isot (megacap/large) ---
        {"symbol":"AAPL","name":"Apple","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"MSFT","name":"Microsoft","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"AMZN","name":"Amazon","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"GOOGL","name":"Alphabet Class A","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"META","name":"Meta Platforms","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"NVDA","name":"NVIDIA","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"BRK-B","name":"Berkshire Hathaway B","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"LLY","name":"Eli Lilly","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"JPM","name":"JPMorgan Chase","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"V","name":"Visa","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"UNH","name":"UnitedHealth","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"PG","name":"Procter & Gamble","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"XOM","name":"Exxon Mobil","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"MA","name":"Mastercard","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"HD","name":"Home Depot","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},

        # --- Keskisuuret (edustavasti, S&P 500:ssa paljon large/mid) ---
        {"symbol":"ADBE","name":"Adobe","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"PFE","name":"Pfizer","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"MCD","name":"McDonald's","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"NKE","name":"Nike","exchange":"NYSE","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"ABNB","name":"Airbnb","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},

        # --- ETF esimerkkejä (jos käytät myös ETF:iä valikossa) ---
        {"symbol":"SPY","name":"SPDR S&P 500 ETF","exchange":"NYSE Arca","country":"US","asset_class":"ETF","currency":"USD","segment":"Isot"},
        {"symbol":"VOO","name":"Vanguard S&P 500 ETF","exchange":"NYSE Arca","country":"US","asset_class":"ETF","currency":"USD","segment":"Isot"},
    ]
    return pd.DataFrame(rows)
