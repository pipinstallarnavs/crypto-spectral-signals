import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
ASSETS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'TRXUSDT', 'DOTUSDT',
    'LINKUSDT', 'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'BCHUSDT', 
    'ATOMUSDT', 'XMRUSDT', 'ETCUSDT', 'XLMUSDT', 'FILUSDT', 
    'ICPUSDT', 'HBARUSDT', 'APTUSDT', 'VETUSDT', 'MKRUSDT', 
    'AAVEUSDT', 'GRTUSDT', 'ALGOUSDT', 'EGLDUSDT', 'SANDUSDT'
]

YEAR = '2025'
# Jan to Dec
MONTHS = [f"{i:02d}" for i in range(1, 13)] 

BASE_URL = "https://data.binance.vision/data/spot/monthly/trades"
OUTPUT_DIR = "data"
OUTPUT_FILE = f"{OUTPUT_DIR}/1s_data_{YEAR}_30.parquet"

# FAKE BROWSER HEADER (Crucial for Server IPs)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def download_and_process_month(symbol, year, month):
    filename = f"{symbol}-trades-{year}-{month}"
    url = f"{BASE_URL}/{symbol}/{filename}.zip"
    
    print(f"   Downloading {symbol} {year}-{month}...")
    
    try:
        response = requests.get(url, stream=True, headers=HEADERS)
        
        if response.status_code == 404:
            print(f"   ⚠️  File not found: {url}")
            return None
        elif response.status_code != 200:
            print(f"   ❌ HTTP Error {response.status_code}")
            return None

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = f"{filename}.csv"
            with z.open(csv_name) as f:
                df = pd.read_csv(f, 
                                 header=None, 
                                 usecols=[1, 2, 4, 5], 
                                 names=['price', 'qty', 'time', 'is_buyer_maker'])

        # --- FOOLPROOF TIMESTAMP FIX ---
        # Strategy: Try 'ms' (Milliseconds). 
        # If it crashes (Year 56971 error), catch it and use 'us' (Microseconds).
        try:
            # Test first value to fail fast
            pd.to_datetime(df['time'].iloc[0], unit='ms')
            
            # If safe, apply to column
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            
            # Double check for future dates (e.g. Year 5000+)
            if df['time'].iloc[0].year > 2030:
                raise ValueError("Year too far in future, switching units.")
                
        except (pd.errors.OutOfBoundsDatetime, ValueError, OverflowError):
            # If MS failed, it MUST be Microseconds
            # print("   (Switching to Microseconds...)")
            df['time'] = pd.to_datetime(df['time'], unit='us')
        # -------------------------------

        df.set_index('time', inplace=True)

        vol_sell = np.where(df['is_buyer_maker'] == True, df['qty'], 0)
        vol_buy = np.where(df['is_buyer_maker'] == False, df['qty'], 0)

        df['vol_sell'] = vol_sell
        df['vol_buy'] = vol_buy

        df_1s = df.resample('1s').agg({
            'price': 'last',
            'vol_buy': 'sum',
            'vol_sell': 'sum'
        })

        # Fill Gaps: Price forward fills, Volume zero fills
        df_1s['price'] = df_1s['price'].ffill()
        df_1s.fillna(0, inplace=True)
        
        # Calculate TI
        df_1s['ti'] = df_1s['vol_buy'] - df_1s['vol_sell']

        return df_1s[['price', 'ti']]

    except Exception as e:
        print(f"   ❌ Error processing {symbol} {month}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    final_data = {}
    start_time = time.time()
    
    print(f"🚀 STARTING PIPELINE: {len(ASSETS)} Assets, Year {YEAR}")
    print("-------------------------------------------------------")

    for symbol in ASSETS:
        asset_name = symbol.replace('USDT', '')
        print(f"\n[Processing {asset_name}]")
        
        yearly_parts = []
        for month in MONTHS:
            # Skip future dates if running in mid-2025
            if int(YEAR) == datetime.now().year and int(month) > datetime.now().month:
                continue
                
            df_month = download_and_process_month(symbol, YEAR, month)
            if df_month is not None:
                yearly_parts.append(df_month)
        
        if yearly_parts:
            full_year = pd.concat(yearly_parts)
            full_year = full_year[~full_year.index.duplicated(keep='first')]
            full_year.columns = [f"{asset_name}_price", f"{asset_name}_ti"]
            final_data[asset_name] = full_year
            print(f"   ✅ {asset_name} Complete. Rows: {len(full_year)}")
        else:
            print(f"   ⚠️ SKIPPING {asset_name} (No data found)")

    print("\n-------------------------------------------------------")
    
    if not final_data:
        print("❌ CRITICAL ERROR: No data was downloaded for ANY asset.")
        return

    print("Merging into Master DataFrame...")
    master_df = pd.concat(final_data.values(), axis=1)
    master_df.sort_index(inplace=True)
    
    price_cols = [c for c in master_df.columns if '_price' in c]
    ti_cols = [c for c in master_df.columns if '_ti' in c]
    
    master_df[price_cols] = master_df[price_cols].ffill()
    master_df[ti_cols] = master_df[ti_cols].fillna(0)
    master_df.dropna(subset=price_cols, inplace=True)
    
    print(f"Final Dataset Shape: {master_df.shape}")
    print(f"Saving to {OUTPUT_FILE}...")
    master_df.to_parquet(OUTPUT_FILE, compression='snappy')
    
    elapsed = (time.time() - start_time) / 60
    print(f"🎉 DONE in {elapsed:.2f} minutes.")

if __name__ == "__main__":
    main()