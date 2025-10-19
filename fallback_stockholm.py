# fallback_stockholm.py
import pandas as pd

def get_fallback_tickers() -> pd.DataFrame:
    rows = [
        # --- Isot ---
        {"symbol":"VOLV-B.ST","name":"Volvo B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"ERIC-B.ST","name":"Ericsson B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"HMB.ST","name":"H&M","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"SAND.ST","name":"Sandvik","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"ATCO-A.ST","name":"Atlas Copco A","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"ATCO-B.ST","name":"Atlas Copco B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"ASSA-B.ST","name":"Assa Abloy B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"INVE-B.ST","name":"Investor B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"BOL.ST","name":"Boliden","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"ESSITY-B.ST","name":"Essity B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"SCA-B.ST","name":"SCA B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},
        {"symbol":"HEXA-B.ST","name":"Hexagon B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Isot"},

        # --- Keskisuuret ---
        {"symbol":"SKF-B.ST","name":"SKF B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Keskisuuret"},
        {"symbol":"ELECTROLUX-B.ST","name":"Electrolux B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Keskisuuret"},
        {"symbol":"TEL2-B.ST","name":"Tele2 B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Keskisuuret"},
        {"symbol":"ALFA.ST","name":"Alfa Laval","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Keskisuuret"},
        {"symbol":"SEB-A.ST","name":"SEB A","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Keskisuuret"},
        {"symbol":"SWED-A.ST","name":"Swedbank A","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Keskisuuret"},

        # --- Pienet ---
        {"symbol":"EQT.ST","name":"EQT AB","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Pienet"},
        {"symbol":"SINCH.ST","name":"Sinch AB","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Pienet"},
        {"symbol":"NIBE-B.ST","name":"NIBE Industrier B","exchange":"Nasdaq Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"Pienet"},

        # --- (valinnainen) First North – lisää tähän kun haluat varmuudella FN-yhtiöitä ---
        # {"symbol":"AAC.ST","name":"AAC Clyde Space","exchange":"First North Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"First North"},
        # {"symbol":"KAMBI.ST","name":"Kambi Group","exchange":"First North Stockholm","country":"SE","asset_class":"Equity","currency":"SEK","segment":"First North"},
    ]
    return pd.DataFrame(rows)
