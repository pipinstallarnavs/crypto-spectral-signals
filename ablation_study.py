import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
import warnings

warnings.filterwarnings("ignore")

DATA_FILE = "data/1s_data_2025.parquet"
TRAIN_RATIO = 0.6
TIMESCALE = "1h" 

# STABLECOINS TO REMOVE
STABLECOINS = ['USDT', 'USDC', 'FDUSD', 'USDP', 'TUSD', 'DAI', 'BUSD']

def run_strategy(returns, strategy_name, cov_estimator):
    # 1. Split
    split = int(len(returns) * TRAIN_RATIO)
    train = returns.iloc[:split]
    test = returns.iloc[split:]
    
    # 2. Fit Model
    cov_estimator.fit(train)
    S = cov_estimator.covariance_
    
    # 3. Optimize (Correct Minimum Variance Formula)
    # w = (inv(S) * 1) / (1.T * inv(S) * 1)
    try:
        inv_cov = np.linalg.pinv(S)
        ones = np.ones(len(S))
        denom = ones.T @ inv_cov @ ones
        w = (inv_cov @ ones) / denom
    except:
        # Fallback if matrix is singular (rare with LedoitWolf)
        w = np.ones(len(S)) / len(S)
    
    # 4. Portfolio Returns
    port_ret = test.dot(w)
    
    # 5. Annualize Sharpe (Correct Logic)
    # Hourly data -> 24 * 365 = 8760 periods
    ann_factor = np.sqrt(365 * 24)
    
    mean_ret = port_ret.mean()
    std_ret = port_ret.std()
    
    if std_ret == 0:
        sharpe = 0
    else:
        # Annualized Sharpe = (Mean / Std) * sqrt(T)
        sharpe = (mean_ret / std_ret) * ann_factor
    
    return sharpe

def run_experiment(df, exclude_stable=False):
    print(f"\n--- EXPERIMENT: {'RISKY ASSETS ONLY' if exclude_stable else 'FULL UNIVERSE'} ---")
    
    # 1. Filter Assets (Robust String Matching)
    price_cols = [c for c in df.columns if '_price' in c]
    
    if exclude_stable:
        # Extract asset name from "BTC_price" -> "BTC"
        filtered_cols = []
        for c in price_cols:
            asset_name = c.replace('_price', '').upper()
            # Check if any stablecoin string is strictly in the asset name
            if not any(s in asset_name for s in STABLECOINS):
                filtered_cols.append(c)
        price_cols = filtered_cols

    print(f"Universe Size: {len(price_cols)} Assets")
    
    # 2. Resample (Last Price)
    rule = {c: 'last' for c in price_cols}
    df_res = df.resample(TIMESCALE).agg(rule).ffill().dropna()
    returns = np.log(df_res / df_res.shift(1)).fillna(0)
    
    # 3. Strategy A: Ledoit-Wolf
    lw = LedoitWolf()
    sharpe_lw = run_strategy(returns, "LW", lw)
    
    # 4. Strategy B: Graph Lasso 
    # Using CV to auto-tune alpha for the risky universe
    gl = GraphicalLassoCV(cv=3) 
    sharpe_gl = run_strategy(returns, "Graph", gl)
    
    print(f"Ledoit-Wolf Sharpe: {sharpe_lw:.4f}")
    print(f"Graph Lasso Sharpe: {sharpe_gl:.4f}")
    
    return sharpe_lw, sharpe_gl

def main():
    df = pd.read_parquet(DATA_FILE)
    
    s_lw_full, s_gl_full = run_experiment(df, exclude_stable=False)
    s_lw_risk, s_gl_risk = run_experiment(df, exclude_stable=True)
    
    print("\n\n=== FINAL VERDICT ===")
    print(f"FULL UNIVERSE GAP: {s_lw_full - s_gl_full:.2f}")
    print(f"RISKY ONLY GAP:    {s_lw_risk - s_gl_risk:.2f}")
    
    if s_lw_risk > s_gl_risk:
        print("CONCLUSION: Robustness holds! LW wins even on pure risky assets.")
    else:
        print("CONCLUSION: Graph wins on risky assets.")

if __name__ == "__main__":
    main()