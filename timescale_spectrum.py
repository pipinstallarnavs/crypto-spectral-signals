import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_FILE = "data/1s_data_2025.parquet"
TRAIN_RATIO = 0.6

TIMESCALES = ['1s', '5s', '15s', '1min', '5min', '30min', '1h']

# Map timescale string to number of seconds (for annualization)
SCALE_MAP = {
    '1s': 1, '5s': 5, '15s': 15, '1min': 60, 
    '5min': 300, '30min': 1800, '1h': 3600
}

def get_sharpe(returns, weights, seconds_per_step):
    port_ret = returns.dot(weights)
    if port_ret.std() == 0: return -999
    
    # Annualization Factor = sqrt(Seconds in Year / Seconds in Step)
    seconds_per_year = 365 * 24 * 60 * 60
    ann_factor = np.sqrt(seconds_per_year / seconds_per_step)
    
    return (port_ret.mean() / port_ret.std()) * ann_factor

def compute_minvar_weights(S):
    # w = (inv(S) * 1) / (1.T * inv(S) * 1)
    try:
        inv = np.linalg.pinv(S)
        ones = np.ones(len(S))
        denom = ones.T @ inv @ ones
        w = (inv @ ones) / denom
        return w
    except:
        return np.ones(len(S)) / len(S)

def run_experiment(df, timescale):
    print(f"\n--- Testing Timescale: {timescale} ---")
    
    # 1. Resample Data
    # FIX: Use 'mean' for TI to keep scale consistent across timescales
    rule = {c: 'mean' if '_ti' in c else 'last' for c in df.columns}
    df_res = df.resample(timescale).agg(rule)
    
    # Clean Data
    price_cols = [c for c in df.columns if '_price' in c]
    ti_cols = [c for c in df.columns if '_ti' in c]
    df_res[price_cols] = df_res[price_cols].ffill()
    df_res[ti_cols] = df_res[ti_cols].fillna(0)
    df_res.dropna(inplace=True)
    
    # 2. Split
    split = int(len(df_res) * TRAIN_RATIO)
    train = df_res.iloc[:split]
    test = df_res.iloc[split:]
    
    # 3. Calculate Returns
    prices = df_res[price_cols]
    returns = np.log(prices / prices.shift(1)).fillna(0)
    train_rets = returns.iloc[:split]
    test_rets = returns.iloc[split:]
    
    # 4. Strategy A: Ledoit-Wolf (Benchmark)
    lw = LedoitWolf().fit(train_rets)
    S_lw = lw.covariance_
    w_lw = compute_minvar_weights(S_lw)
    
    # 5. Strategy B: Graph Filter (The Challenger)
    tis_train = train[ti_cols]
    ti_std = (tis_train - tis_train.mean()) / tis_train.std()
    
    try:
        # FIX: Use CV to adapt alpha to the timescale's noise level
        gl = GraphicalLassoCV(alphas=[0.01, 0.05, 0.1, 0.2, 0.5], cv=3).fit(ti_std)
        
        prec = gl.precision_
        adj = np.abs(prec)
        np.fill_diagonal(adj, 0)
        edges = np.sum(adj > 1e-5) / 2
        
        # Spectral Filter
        degs = adj.sum(axis=1)
        degs[degs==0] = 1
        L = np.diag(degs) - adj
        D_inv = np.diag(1/np.sqrt(degs))
        L_norm = D_inv @ L @ D_inv
        
        evals, evecs = np.linalg.eigh(L_norm)
        
        # Keep 50%
        k = len(evals) // 2
        U = evecs[:, :k]
        S_filt = U @ (U.T @ S_lw @ U) @ U.T + np.eye(len(S_lw))*1e-6
        
        w_filt = compute_minvar_weights(S_filt)
        
    except Exception as e:
        print(f"Graph Fitting Failed: {e}")
        edges = 0
        w_filt = w_lw # Fallback
        
    # 6. Compare
    sec_step = SCALE_MAP[timescale]
    sharpe_lw = get_sharpe(test_rets, w_lw, sec_step)
    sharpe_filt = get_sharpe(test_rets, w_filt, sec_step)
    
    print(f"Edges Found: {int(edges)}")
    print(f"Sharpe LW:   {sharpe_lw:.4f}")
    print(f"Sharpe FILT: {sharpe_filt:.4f}")
    
    return {'scale': timescale, 'edges': edges, 'lw': sharpe_lw, 'filt': sharpe_filt}

def main():
    df = pd.read_parquet(DATA_FILE)
    results = []
    
    for scale in TIMESCALES:
        res = run_experiment(df, scale)
        results.append(res)
        
    print("\n\n=== FINAL RESULTS TABLE ===")
    print(f"{'SCALE':<10} {'EDGES':<10} {'LW (Bench)':<12} {'GRAPH (Us)':<12} {'WINNER'}")
    for r in results:
        winner = "GRAPH 🏆" if r['filt'] > r['lw'] else "LW"
        print(f"{r['scale']:<10} {int(r['edges']):<10} {r['lw']:.4f}       {r['filt']:.4f}       {winner}")

if __name__ == "__main__":
    main()