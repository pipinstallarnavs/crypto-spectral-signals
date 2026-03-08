import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from scipy.linalg import eigh
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# --- CORE ALGORITHMS ---

def learn_latent_topology(data):
    """Learns the graph structure (adjacency)"""
    try:
        # Normalize
        data_std = (data - data.mean()) / data.std()
        data_std = data_std.fillna(0)
        
        # Robust Graph Lasso (Fixed max_iter)
        model = GraphicalLassoCV(cv=3, assume_centered=False, n_jobs=-1, max_iter=1000)
        model.fit(data_std)
        
        precision = model.precision_
        adjacency = np.abs(precision)
        np.fill_diagonal(adjacency, 0)
        return adjacency
    except:
        # Fallback for convergence failures
        return np.zeros((data.shape[1], data.shape[1]))

def apply_spectral_filter(sample_cov, adjacency, cutoff_pct=0.5):
    """The 'Guillotine' Filter"""
    try:
        # Laplacian
        degs = adjacency.sum(axis=1)
        degs[degs == 0] = 1e-10
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degs))
        L = np.eye(len(degs)) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        # Eigen-decomp
        evals, U = eigh(L)
        
        # Mask creation
        k = int(len(degs) * cutoff_pct)
        k = max(1, min(k, len(degs)-1)) # Bounds check
        mask = np.diag([1]*k + [0]*(len(degs)-k))
        
        # Filter
        S_clean = U @ (mask @ (U.T @ sample_cov @ U)) @ U.T
        return S_clean
    except:
        return sample_cov

# --- ROBUSTNESS ENGINE ---

def run_robustness_check(returns_df, ti_df, window=60):
    print("--- Running Full Robustness Suite ---")
    results = {'LW': [], 'GSP_0.3': [], 'GSP_0.5': [], 'GSP_0.7': []}
    dates = []
    
    # Step size: Every 10th window for speed (adjust as needed)
    step = 10 
    
    for i in range(window, len(returns_df), step):
        t = returns_df.index[i]
        
        # Data Slices
        ret_win = returns_df.iloc[i-window:i]
        ti_win = ti_df.iloc[i-window:i]
        next_ret = returns_df.iloc[i]
        
        # 1. Ledoit-Wolf Benchmark
        try:
            lw = LedoitWolf().fit(ret_win)
            w_lw = np.linalg.inv(lw.covariance_) @ np.ones(len(lw.covariance_))
            results['LW'].append(np.dot(w_lw/w_lw.sum(), next_ret))
        except:
            results['LW'].append(0)
            
        # 2. GSP Variants
        # Learn Topology ONCE per window
        adj = learn_latent_topology(ti_win)
        cov_sample = ret_win.cov().values
        
        for cut in [0.3, 0.5, 0.7]:
            # Apply different filter strengths
            cov_gsp = apply_spectral_filter(cov_sample, adj, cutoff_pct=cut)
            try:
                w_gsp = np.linalg.inv(cov_gsp) @ np.ones(len(cov_gsp))
                results[f'GSP_{cut}'].append(np.dot(w_gsp/w_gsp.sum(), next_ret))
            except:
                results[f'GSP_{cut}'].append(0)
        
        dates.append(t)
        
        if i % 1000 == 0: print(f"Processed {i}/{len(returns_df)}")
            
    return pd.DataFrame(results, index=dates)

if __name__ == "__main__":
    # Mock Data
    np.random.seed(42)
    df = pd.DataFrame(np.random.normal(0,1, (2000, 5)), columns=[f'A{i}' for i in range(5)])
    
    # Run
    res = run_robustness_check(df, df, window=50) # Using df as TI for mock
    
    # Plot Cumulative Wealth
    (1 + res).cumprod().plot(title="Robustness Check: Wealth Curve")
    plt.savefig("wealth_curve.png")
    print("Robustness check complete. Saved 'wealth_curve.png'")