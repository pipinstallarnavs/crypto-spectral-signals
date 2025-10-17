import pandas as pd
import numpy as np
import networkx as nx
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DATA_FILE = "data/1s_data_2025.parquet"
TRAIN_RATIO = 0.6

# 25 RUNS (5 Alphas x 5 Filters)
ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.4] 
FILTERS = [0.1, 0.3, 0.5, 0.7, 0.9] # % of eigenvalues to KEEP

def load_data(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_parquet(filepath)
    price_cols = [c for c in df.columns if '_price' in c]
    ti_cols = [c for c in df.columns if '_ti' in c]
    
    prices = df[price_cols]
    tis = df[ti_cols]
    
    assets = [c.replace('_price', '') for c in price_cols]
    prices.columns = assets
    tis.columns = assets
    
    returns = np.log(prices / prices.shift(1)).fillna(0)
    return returns, tis, assets

def get_graph_covariance(tis_train, returns_train, alpha, keep_frac):
    # 1. Learn Graph (Force Alpha)
    # Standardize TI
    ti_std = (tis_train - tis_train.mean()) / tis_train.std()
    
    try:
        gl = GraphicalLasso(alpha=alpha, max_iter=100)
        gl.fit(ti_std)
        precision = gl.precision_
        adjacency = np.abs(precision)
        np.fill_diagonal(adjacency, 0)
    except:
        # If it fails to converge, return None
        return None, 0

    n_edges = np.sum(adjacency > 1e-5) / 2
    
    # 2. Filter Covariance
    lw = LedoitWolf().fit(returns_train)
    S_noisy = lw.covariance_
    
    degrees = np.sum(adjacency, axis=1)
    # Handle disconnected nodes (degree 0) to avoid div by zero
    degrees[degrees == 0] = 1 
    
    D = np.diag(degrees)
    L = D - adjacency
    
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    evals, evecs = np.linalg.eigh(L_norm)
    
    # Keep bottom k eigenvalues (Smoothness)
    k = int(len(evals) * keep_frac)
    k = max(1, k) # Keep at least 1
    
    U_k = evecs[:, :k]
    S_filtered = U_k @ (U_k.T @ S_noisy @ U_k) @ U_k.T
    S_filtered += np.eye(len(S_filtered)) * 1e-6
    
    return S_filtered, n_edges

def backtest_sharpe(returns, split_idx, S_filtered):
    test_returns = returns.iloc[split_idx:]
    inv_cov = np.linalg.pinv(S_filtered)
    ones = np.ones(len(S_filtered))
    weights = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
    
    port_ret = test_returns.dot(weights)
    
    # Annualized Sharpe (approx)
    ann_factor = np.sqrt(365*24*60*60)
    mean = port_ret.mean() * (ann_factor**2)
    vol = port_ret.std() * ann_factor
    
    if vol == 0: return 0
    return mean / vol

def main():
    returns, tis, assets = load_data(DATA_FILE)
    split_idx = int(len(returns) * TRAIN_RATIO)
    
    train_returns = returns.iloc[:split_idx]
    train_tis = tis.iloc[:split_idx]
    
    # Baseline: Ledoit Wolf
    lw = LedoitWolf().fit(train_returns)
    S_lw = lw.covariance_
    base_sharpe = backtest_sharpe(returns, split_idx, S_lw)
    print(f"\nBASE SHARPE (Ledoit-Wolf): {base_sharpe:.4f}")
    print("-" * 60)
    print(f"{'ALPHA':<10} {'KEEP_FRAC':<12} {'EDGES':<10} {'SHARPE':<10} {'VS BASE':<10}")
    print("-" * 60)
    
    results = []

    for alpha in ALPHAS:
        for frac in FILTERS:
            S_filt, edges = get_graph_covariance(train_tis, train_returns, alpha, frac)
            
            if S_filt is None:
                print(f"{alpha:<10} {frac:<12} FAILED")
                continue
                
            sharpe = backtest_sharpe(returns, split_idx, S_filt)
            diff = sharpe - base_sharpe
            
            # Simple indicator
            flag = "WIN 🏆" if diff > 0 else ""
            
            print(f"{alpha:<10} {frac:<12} {int(edges):<10} {sharpe:.4f}     {flag}")
            
            results.append({
                'alpha': alpha, 
                'frac': frac, 
                'sharpe': sharpe, 
                'edges': edges
            })

if __name__ == "__main__":
    main()