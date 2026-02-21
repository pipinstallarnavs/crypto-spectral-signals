import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV
import warnings

warnings.filterwarnings("ignore")

DATA_FILE = "data/1h_data_2025_19.parquet"
LOOKBACK = 720
TIME_BW = 5
TAU_VALUES = np.linspace(0.0, 3.0, 20)

def learn_graph_from_ti(ti_window):
    ti_std = (ti_window - ti_window.mean()) / (ti_window.std() + 1e-8)
    ti_std = ti_std.fillna(0)
    try:
        gl = GraphicalLassoCV(cv=3, max_iter=500).fit(ti_std)
        adj = np.abs(gl.precision_)
    except Exception:
        adj = np.abs(ti_std.corr().values)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(adj, 0)
    degrees = adj.sum(axis=1)
    degrees[degrees == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = np.eye(len(degrees)) - (D_inv_sqrt @ adj @ D_inv_sqrt)
    return (L_norm + L_norm.T) / 2.0

def gaussian_time_kernel(T, bandwidth):
    t = np.arange(T)
    K = np.exp(-0.5 * ((t[:, None] - t[None, :]) / bandwidth) ** 2)
    return K / K.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    df = pd.read_parquet(DATA_FILE)
    price_cols = [c for c in df.columns if '_price' in c]
    ti_cols    = [c for c in df.columns if '_ti' in c]
    
    returns_df = np.log(df[price_cols] / df[price_cols].shift(1)).fillna(0).replace([np.inf, -np.inf], 0)
    returns_df += np.random.normal(0, 1e-10, returns_df.shape)
    ti_df = df[ti_cols].copy()

    ret_train = returns_df.iloc[-LOOKBACK:]
    ti_train = ti_df.iloc[-LOOKBACK:]

    T = len(ti_train)
    ti_smoothed = pd.DataFrame(gaussian_time_kernel(T, TIME_BW) @ ti_train.values)
    L_norm = learn_graph_from_ti(ti_smoothed)
    X_raw = ret_train.values  
    evals, evecs = np.linalg.eigh(L_norm)

    condition_numbers = []

    print("Running Tau Sweep...")
    for tau in TAU_VALUES:
        G_filter = evecs @ np.diag(np.exp(-tau * evals)) @ evecs.T
        X_projected = X_raw @ G_filter
        S_filt = pd.DataFrame(X_projected).cov().values
        S_filt += np.eye(len(S_filt)) * 1e-10
        
        c_evals = np.linalg.eigvalsh(S_filt)
        cond = c_evals[-1] / (c_evals[0] if c_evals[0] > 0 else 1e-10)
        condition_numbers.append(cond)
        print(f"Tau: {tau:.2f} | Cond: {cond:,.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(TAU_VALUES, condition_numbers, 'r-', linewidth=2, marker='o')
    plt.yscale('log')
    plt.title("Condition Number Divergence vs. Filter Strength (Tau)")
    plt.xlabel("Heat Kernel Filter Strength (Tau)")
    plt.ylabel("Condition Number (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.savefig("tau_divergence_curve.png", dpi=300)
    print("Saved 'tau_divergence_curve.png'.")