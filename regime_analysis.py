import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
WINDOW_SIZE = 3600       # 1 Hour of data (in seconds)
ALGO_TIMEOUT = 1000      # Max iterations for Glasso (Fail fast on noise)

def get_regime_metrics(returns_window):
    """Calculates the 'Regime' of the market (Density & Volatility)"""
    # 1. Correlation Density (How coupled is the market?)
    # We take the absolute average of off-diagonal elements
    corr_matrix = returns_window.corr().values
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    avg_corr = np.abs(corr_matrix[mask]).mean()
    
    # 2. Volatility Regime (Is the market panicking?)
    # Average standard deviation across all assets
    avg_vol = returns_window.std().mean()
    
    return avg_corr, avg_vol

def get_strategy_return(cov_matrix, next_returns):
    """Calculates return of GMVP portfolio for a given covariance matrix"""
    try:
        # Inverse variance portfolio (Simple GMVP)
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        weights = inv_cov @ ones
        weights = weights / weights.sum() # Normalize to sum to 1
        return np.dot(weights, next_returns)
    except:
        return 0.0 # Return flat if matrix is singular

def run_regime_scan(returns_df, step_size):
    results = []
    
    print(f"--- Starting Regime Scan (N={len(returns_df)}) ---")
    print(f"Step Size: {step_size} rows (scanning every ~{step_size/3600:.1f} hours)")
    
    # Iterate through the dataset
    for i in range(WINDOW_SIZE, len(returns_df), step_size):
        
        # 1. Define Window
        window = returns_df.iloc[i-WINDOW_SIZE:i]
        if i + 1 >= len(returns_df): break
        next_ret = returns_df.iloc[i] # One-step ahead out-of-sample
        
        # 2. Identify Regime
        avg_corr, avg_vol = get_regime_metrics(window)
        
        # 3. Run Strategies
        # A. Ledoit-Wolf (Robust Benchmark)
        try:
            lw = LedoitWolf().fit(window)
            ret_lw = get_strategy_return(lw.covariance_, next_ret)
        except:
            ret_lw = 0.0
        
        # B. Graphical Lasso (Sparse/Structural Candidate)
        try:
            # We use a fixed cross-validation (cv=3) to find structure
            gl = GraphicalLassoCV(cv=3, max_iter=ALGO_TIMEOUT, assume_centered=False, n_jobs=-1)
            gl.fit(window)
            ret_gsp = get_strategy_return(gl.covariance_, next_ret)
            
            # Metric: Graph Density (Did it find a sparse structure?)
            prec = gl.precision_
            n_features = len(prec)
            # Count edges (non-zero off-diagonals)
            n_edges = np.count_nonzero(np.abs(prec) > 1e-5) - n_features
            max_edges = n_features**2 - n_features
            graph_density = n_edges / max_edges if max_edges > 0 else 1.0
            
        except Exception:
            # If Glasso fails (common in high noise), we assume it failed to find structure
            ret_gsp = ret_lw 
            graph_density = 1.0 
            
        # 4. Store Data
        results.append({
            'timestamp': window.index[-1],
            'avg_correlation': avg_corr,
            'volatility': avg_vol,
            'graph_density': graph_density,
            'return_lw': ret_lw,
            'return_gsp': ret_gsp,
            'diff_gsp_minus_lw': ret_gsp - ret_lw # Positive = Glasso Wins
        })
        
        # Progress Tracker
        if len(results) % 50 == 0:
            print(f"Scanned {i}/{len(returns_df)} windows...")
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    
    # --- 1. LOAD REAL DATA ---
    # Path based on your screenshot
    file_path = "data/1s_data_2025.parquet"
    
    print(f"Loading data from: {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: File not found at {file_path}")
        print("Make sure you are running this from the 'Signal_SOP' directory.")
        exit()

    # Load Parquet
    returns_df = pd.read_parquet(file_path)
    print("Data loaded. Preprocessing...")

    # --- 2. DATA CLEANING ---
    # Ensure Index is Datetime
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        if 'timestamp' in returns_df.columns:
            returns_df = returns_df.set_index('timestamp')
        elif 'Date' in returns_df.columns:
            returns_df = returns_df.set_index('Date')
            
    # Forward Fill gaps (Standard crypto practice)
    returns_df = returns_df.ffill()
    
    # Check if data is Prices (Values > 1.0) and convert to Returns
    # This prevents the "2 seconds" error where you process prices as returns
    first_val = returns_df.iloc[0, 0]
    if first_val > 1.0: 
        print(f"Detected Prices (First value: {first_val}). Converting to Returns...")
        returns_df = returns_df.pct_change()
    else:
        print(f"Detected Returns (First value: {first_val}). Proceeding...")
        
    # Drop NaNs created by pct_change
    returns_df = returns_df.dropna()
    
    # Clean Infinite values
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Ready for Scan. Total Rows: {len(returns_df)}")
    
    # --- 3. RUN THE SCAN ---
    # 7200 rows = 2 Hours. 
    # For 31M rows, this generates ~4,300 data points. Perfect for scatter plot.
    STEP_SIZE_REAL = 7200 
    
    regime_df = run_regime_scan(returns_df, step_size=STEP_SIZE_REAL)
    
    # --- 4. PLOTTING THE "NOVELTY" ---
    plt.figure(figsize=(12, 8))
    
    # Scatter Plot: Correlation vs Performance Gap
    sc = plt.scatter(
        regime_df['avg_correlation'], 
        regime_df['diff_gsp_minus_lw'], 
        c=regime_df['volatility'], 
        cmap='viridis', 
        alpha=0.6,
        edgecolors='none',
        s=20
    )
    
    # Zero Line (Where performance is equal)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label="Equal Performance")
    
    # Add Trendline (The "Proof")
    try:
        # Filter out extreme outliers for cleaner trendline
        clean_df = regime_df[np.abs(regime_df['diff_gsp_minus_lw']) < 0.05]
        z = np.polyfit(clean_df['avg_correlation'], clean_df['diff_gsp_minus_lw'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(clean_df['avg_correlation'].min(), clean_df['avg_correlation'].max(), 100)
        plt.plot(x_range, p(x_range), "r-", linewidth=2.5, label="Regime Trend")
    except Exception as e:
        print(f"Trendline skipped: {e}")

    # Labels and Style
    plt.colorbar(sc, label='Market Volatility (Std Dev)')
    plt.title("The Decision Framework: Estimator Performance vs. Market Correlation", fontsize=14)
    plt.xlabel("Average Pairwise Correlation (Regime)", fontsize=12)
    plt.ylabel("Advantage: Glasso vs Ledoit-Wolf (Positive = Glasso Wins)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Save
    save_path = "regime_decision_boundary.png"
    plt.savefig(save_path, dpi=300)
    print(f"Analysis Complete. 'Findings' plot saved to: {save_path}")