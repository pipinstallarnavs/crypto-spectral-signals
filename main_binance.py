import pandas as pd
import numpy as np
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DATA_FILE = "data/1s_data_2025.parquet"
TRAIN_RATIO = 0.6  # First 60% for training graph, rest for testing
WINDOW_SIZE = 3600 # Rebalance every hour (3600 seconds)

def load_data(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_parquet(filepath)
    
    # Separate Price and TI columns
    price_cols = [c for c in df.columns if '_price' in c]
    ti_cols = [c for c in df.columns if '_ti' in c]
    
    prices = df[price_cols]
    tis = df[ti_cols]
    
    # Rename columns to just asset names (e.g. "BTC", "ETH")
    assets = [c.replace('_price', '') for c in price_cols]
    prices.columns = assets
    tis.columns = assets
    
    # Calculate Log Returns
    returns = np.log(prices / prices.shift(1)).fillna(0)
    
    return returns, tis, assets

def learn_graph(tis_train, assets):
    print("Learning Graph Topology from Trade Imbalance...")
    
    # Standardize TI
    ti_std = (tis_train - tis_train.mean()) / tis_train.std()
    
    # Use GLASSO to find sparse precision matrix
    # alpha is regularization strength. Higher = Sparser graph.
    gl = GraphicalLassoCV(alphas=[0.1, 0.2, 0.5], cv=3)
    gl.fit(ti_std)
    
    precision = gl.precision_
    adjacency = np.abs(precision)
    np.fill_diagonal(adjacency, 0) # No self-loops
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adjacency)
    mapping = {i: asset for i, asset in enumerate(assets)}
    G = nx.relabel_nodes(G, mapping)
    
    print(f"Graph Learned. Edges: {G.number_of_edges()}")
    return G, adjacency

def filter_covariance(returns_train, adjacency):
    print("Applying Graph Spectral Filtering to Covariance...")
    
    # 1. Compute Standard Covariance (Ledoit-Wolf for stability)
    lw = LedoitWolf().fit(returns_train)
    S_noisy = lw.covariance_
    
    # 2. Compute Graph Laplacian
    degrees = np.sum(adjacency, axis=1)
    D = np.diag(degrees)
    L = D - adjacency
    
    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    # 3. Eigendecomposition of Laplacian
    evals, evecs = np.linalg.eigh(L_norm)
    
    # 4. Filter: Keep only "smooth" modes (low eigenvalues)
    # This removes high-frequency noise that doesn't align with the graph
    # We keep the bottom 50% of modes (the "structure")
    k = len(evals) // 2
    U_k = evecs[:, :k]
    
    # Project Covariance onto Graph Basis
    S_filtered = U_k @ (U_k.T @ S_noisy @ U_k) @ U_k.T
    
    # Add small regularization to ensure positive definite
    S_filtered += np.eye(len(S_filtered)) * 1e-6
    
    return S_filtered, S_noisy

def minimum_variance_portfolio(cov_matrix):
    inv_cov = np.linalg.pinv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    weights = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
    return weights

def backtest(returns, split_idx, S_filtered, S_noisy, window=3600):
    print("Running Backtest...")
    
    test_returns = returns.iloc[split_idx:]
    n_assets = returns.shape[1]
    
    # Strategy 1: Graph Filtered MVP
    w_graph = minimum_variance_portfolio(S_filtered)
    
    # Strategy 2: Standard Ledoit-Wolf MVP
    w_bench = minimum_variance_portfolio(S_noisy)
    
    # Strategy 3: Equal Weight (Naive Benchmark)
    w_eq = np.ones(n_assets) / n_assets
    
    # Calculate Portfolio Returns
    # Note: In a real HFT backtest, you'd rebalance dynamically. 
    # For this paper, we test if the STATIC covariance estimate holds up out-of-sample.
    
    port_graph = test_returns.dot(w_graph)
    port_bench = test_returns.dot(w_bench)
    port_eq = test_returns.dot(w_eq)
    
    return port_graph, port_bench, port_eq

def analyze_results(p_graph, p_bench, p_eq):
    print("\n--- PERFORMANCE RESULTS (Annualized) ---")
    
    # Assuming 1-second data, approx 31.5 million seconds per year
    # But crypto trades 24/7.
    SECONDS_PER_YEAR = 365 * 24 * 60 * 60
    
    res = {}
    for name, series in [("Graph Filter", p_graph), ("Ledoit-Wolf", p_bench), ("Equal Weight", p_eq)]:
        mean_ret = series.mean() * SECONDS_PER_YEAR
        vol = series.std() * np.sqrt(SECONDS_PER_YEAR)
        sharpe = mean_ret / vol if vol > 0 else 0
        
        print(f"{name}: Return={mean_ret:.2%}, Vol={vol:.2%}, Sharpe={sharpe:.2f}")
        res[name] = sharpe
        
    return res

def main():
    # 1. Load Data
    returns, tis, assets = load_data(DATA_FILE)
    n = len(returns)
    split_idx = int(n * TRAIN_RATIO)
    
    print(f"Data Loaded. Assets: {len(assets)}. Total Seconds: {n}")
    print(f"Training on first {split_idx} seconds...")
    
    train_returns = returns.iloc[:split_idx]
    train_tis = tis.iloc[:split_idx]
    
    # 2. Learn Graph from Trade Imbalance (TI)
    G, adjacency = learn_graph(train_tis, assets)
    
    # 3. Filter Return Covariance using TI Graph
    S_filtered, S_noisy = filter_covariance(train_returns, adjacency)
    
    # 4. Backtest Out-of-Sample
    p_graph, p_bench, p_eq = backtest(returns, split_idx, S_filtered, S_noisy)
    
    # 5. Analyze
    analyze_results(p_graph, p_bench, p_eq)
    
    print("\n✅ Experiment Complete.")

if __name__ == "__main__":
    main()