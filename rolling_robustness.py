import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
import warnings

warnings.filterwarnings("ignore")

DATA_FILE = "data/1s_data_2025.parquet"
TIMESCALE = "1h"

# ROLLING CONFIG
TRAIN_MONTHS = 3
TEST_MONTHS = 1

def get_weights(returns, method='lw'):
    try:
        if method == 'lw':
            cov = LedoitWolf().fit(returns).covariance_
        elif method == 'graph':
            # Use CV to allow adaptivity
            gl = GraphicalLassoCV(cv=3).fit(returns)
            cov = np.linalg.pinv(gl.precision_)
        
        # Min Variance Weights
        inv = np.linalg.pinv(cov)
        ones = np.ones(len(cov))
        w = (inv @ ones) / (ones.T @ inv @ ones)
        return w
    except:
        return np.ones(returns.shape[1]) / returns.shape[1]

def run_rolling_analysis(df):
    print(f"Loading data & resampling to {TIMESCALE}...")
    # Clean & Resample
    price_cols = [c for c in df.columns if '_price' in c]
    rule = {c: 'last' for c in price_cols}
    df_res = df.resample(TIMESCALE).agg(rule).ffill().dropna()
    returns = np.log(df_res / df_res.shift(1)).fillna(0)
    
    # Define Windows (Monthly steps)
    # Approx 720 hours per month
    MONTH_HOURS = 24 * 30
    window_size = TRAIN_MONTHS * MONTH_HOURS
    step_size = TEST_MONTHS * MONTH_HOURS
    
    results = []
    weight_history_lw = []
    weight_history_graph = []
    dates = []
    
    # Rolling Walk-Forward
    for start_idx in range(0, len(returns) - window_size - step_size, step_size):
        train_end = start_idx + window_size
        test_end = train_end + step_size
        
        train_data = returns.iloc[start_idx:train_end]
        test_data = returns.iloc[train_end:test_end]
        test_dates = returns.index[train_end:test_end]
        
        # 1. Get Weights (Static for this test month)
        w_lw = get_weights(train_data, 'lw')
        w_graph = get_weights(train_data, 'graph')
        
        # 2. Store Weights for Stability Plot
        weight_history_lw.append(w_lw)
        weight_history_graph.append(w_graph)
        dates.append(test_dates[0])
        
        # 3. Test Performance
        r_lw = test_data.dot(w_lw)
        r_graph = test_data.dot(w_graph)
        
        # Annualize Sharpe for this month
        ann = np.sqrt(24 * 365)
        s_lw = (r_lw.mean() / r_lw.std()) * ann if r_lw.std() > 0 else 0
        s_graph = (r_graph.mean() / r_graph.std()) * ann if r_graph.std() > 0 else 0
        
        print(f"Period {dates[-1].date()}: LW={s_lw:.2f} | Graph={s_graph:.2f}")
        
        results.append({
            'date': dates[-1],
            'LW': s_lw,
            'Graph': s_graph
        })

    # --- PLOT 1: ROLLING SHARPE ---
    res_df = pd.DataFrame(results).set_index('date')
    print("\nAverage Rolling Sharpe:")
    print(res_df.mean())
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_df.index, res_df['LW'], label='Ledoit-Wolf', marker='o')
    plt.plot(res_df.index, res_df['Graph'], label='Graph Lasso', marker='x', color='red')
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Rolling Out-of-Sample Performance ({TRAIN_MONTHS}m Train / {TEST_MONTHS}m Test)")
    plt.ylabel("Annualized Sharpe")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rolling_sharpe.png")
    
    # --- PLOT 2: WEIGHT STABILITY (Fixed) ---
    # Switch to Line Plot to handle negative weights (Short Positions)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # LW Weights
    w_lw_df.plot(ax=axes[0], cmap='tab10', linewidth=2)
    axes[0].set_title("Ledoit-Wolf Weight Stability (Robust)", fontsize=14)
    axes[0].set_ylabel("Weight Allocation")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Graph Weights
    w_graph_df.plot(ax=axes[1], cmap='tab10', linewidth=2, legend=False)
    axes[1].set_title("Graph Lasso Weight Stability (Volatile)", fontsize=14)
    axes[1].set_ylabel("Weight Allocation")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("weight_stability.png")
    print("Saved rolling_sharpe.png and weight_stability.png")