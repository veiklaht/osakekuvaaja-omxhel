# fallback_nasdaq_us.py
import pandas as pd

def get_fallback_tickers() -> pd.DataFrame:
    rows = [
        # --- Isot (Nasdaq megacap) ---
        {"symbol":"AAPL","name":"Apple","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"MSFT","name":"Microsoft","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"AMZN","name":"Amazon","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"GOOGL","name":"Alphabet Class A","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"META","name":"Meta Platforms","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"NVDA","name":"NVIDIA","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"TSLA","name":"Tesla","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"AVGO","name":"Broadcom","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"COST","name":"Costco","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},
        {"symbol":"PEP","name":"PepsiCo","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Isot"},

        # --- Keskisuuret ---
        {"symbol":"NFLX","name":"Netflix","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"AMD","name":"Advanced Micro Devices","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"INTC","name":"Intel","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"CSCO","name":"Cisco Systems","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"QCOM","name":"Qualcomm","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"ADBE","name":"Adobe","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"AMAT","name":"Applied Materials","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"TXN","name":"Texas Instruments","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"INTU","name":"Intuit","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},
        {"symbol":"SBUX","name":"Starbucks","exchange":"NASDAQ","country":"US","asset_class":"Equity","currency":"USD","segment":"Keskisuuret"},

        # --- ETF (valinnainen) ---
        {"symbol":"QQQ","name":"Invesco QQQ","exchange":"NASDAQ","country":"US","asset_class":"ETF","currency":"USD","segment":"Isot"},
    ]
    return pd.DataFrame(rows)
