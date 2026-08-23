import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_FILE = "data/1h_data_2025_19.parquet" 
LOOKBACK     = 720    # 30 days of hourly bars
STEP         = 72     # 3 days OOS
ALGO_TIMEOUT = 500
TIME_BW        = 5      
GRAPH_TAU      = 1.5    
TIKHONOV_DELTA = 0.05   

# ==============================================================================
# GRAPH & FILTER CORE
# ==============================================================================
def learn_graph_from_ti(ti_window):
    ti_std = (ti_window - ti_window.mean()) / (ti_window.std() + 1e-8)
    ti_std = ti_std.fillna(0)
    
    try:
        gl = GraphicalLassoCV(cv=3, max_iter=ALGO_TIMEOUT).fit(ti_std)
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

def get_jtvf_covariances(returns_window, ti_window):
    T = len(ti_window)
    ti_smoothed = pd.DataFrame(gaussian_time_kernel(T, TIME_BW) @ ti_window.values)
    L_norm = learn_graph_from_ti(ti_smoothed)
    
    X_raw = returns_window.values  
    evals, evecs = np.linalg.eigh(L_norm)   
    
    G_filter = evecs @ np.diag(np.exp(-GRAPH_TAU * evals)) @ evecs.T
    X_projected = X_raw @ G_filter
    
    S_filt = pd.DataFrame(X_projected).cov().values
    
    S_unconstrained = S_filt + np.eye(len(S_filt)) * 1e-8 
    S_stabilized = S_filt + np.eye(len(S_filt)) * TIKHONOV_DELTA
    
    return S_unconstrained, S_stabilized

# ==============================================================================
# DIAGNOSTICS (Updated: Removed Marchenko-Pastur)
# ==============================================================================
def get_spectral_metrics(cov_matrix):
    evals = np.linalg.eigvalsh(cov_matrix)
    lam_max, lam_min = evals[-1], evals[0]
    condition_number = lam_max / (lam_min if lam_min > 0 else 1e-10)
    return condition_number, lam_min

def get_portfolio_metrics(cov_matrix, next_returns):
    try:
        inv_cov = np.linalg.pinv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        w = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
    except Exception:
        w = np.ones(len(cov_matrix)) / len(cov_matrix)
        
    # Calculate realized Out-Of-Sample return for this window
    ret = float(np.dot(w, next_returns))
    l1_norm = np.sum(np.abs(w)) 
    return ret, l1_norm

# ==============================================================================
# PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("Loading hourly data...")
    df = pd.read_parquet(DATA_FILE)
    
    price_cols = [c for c in df.columns if '_price' in c]
    ti_cols    = [c for c in df.columns if '_ti' in c]
    N_assets = len(price_cols)

    returns_df = np.log(df[price_cols] / df[price_cols].shift(1)).fillna(0).replace([np.inf, -np.inf], 0)
    np.random.seed(42)
    returns_df += np.random.normal(0, 1e-10, returns_df.shape)
    ti_df = df[ti_cols].copy()

    results = []
    print(f"--- Running Spectral Paradox Diagnostics ({N_assets} assets) ---")
    
    for i in range(LOOKBACK, len(returns_df), STEP):
        if i + STEP > len(returns_df): break

        ret_train = returns_df.iloc[i - LOOKBACK : i]
        ti_train  = ti_df.iloc[i - LOOKBACK : i]
        # Calculate the actual OOS holding period return for each asset
        ret_test  = returns_df.iloc[i : i + STEP].sum(axis=0) 

        # A: Ledoit-Wolf
        S_lw = LedoitWolf().fit(ret_train).covariance_
        cond_lw, min_lw = get_spectral_metrics(S_lw)
        ret_lw, lev_lw = get_portfolio_metrics(S_lw, ret_test)

        # B: JTVF Unconstrained
        S_jtvf, S_jtvf_reg = get_jtvf_covariances(ret_train, ti_train)
        cond_jtvf, min_jtvf = get_spectral_metrics(S_jtvf)
        ret_jtvf, lev_jtvf = get_portfolio_metrics(S_jtvf, ret_test)

        # C: JTVF Stabilized
        cond_reg, min_reg = get_spectral_metrics(S_jtvf_reg)
        ret_reg, lev_reg = get_portfolio_metrics(S_jtvf_reg, ret_test)

        results.append({
            'Window': i,
            'Cond_LW': cond_lw, 'Cond_JTVF': cond_jtvf, 'Cond_Reg': cond_reg,
            'Lev_LW': lev_lw,   'Lev_JTVF': lev_jtvf,   'Lev_Reg': lev_reg,
            'Ret_LW': ret_lw,   'Ret_JTVF': ret_jtvf,   'Ret_Reg': ret_reg
        })
        
        if len(results) % 5 == 0:
            print(f"  Processed {len(results)} windows...")

    res_df = pd.DataFrame(results)
    
    print("\n=== EMPIRICAL RESULTS (UNBIASED) ===")
    print(f"Ledoit-Wolf Cond: {res_df['Cond_LW'].median():,.2f} | Lev: {res_df['Lev_LW'].median():.2f}x")
    print(f"JTVF Raw Cond   : {res_df['Cond_JTVF'].median():,.2f} | Lev: {res_df['Lev_JTVF'].median():.2f}x")
    print(f"JTVF Reg Cond   : {res_df['Cond_Reg'].median():,.2f} | Lev: {res_df['Lev_Reg'].median():.2f}x")
    
    # Calculate Out-Of-Sample Volatility (Risk)
    vol_lw = res_df['Ret_LW'].std()
    vol_jtvf = res_df['Ret_JTVF'].std()
    vol_reg = res_df['Ret_Reg'].std()
    
    print("\n=== OUT-OF-SAMPLE RISK (VOLATILITY) ===")
    print(f"Ledoit-Wolf OOS Vol: {vol_lw:.6f}")
    print(f"JTVF Raw OOS Vol   : {vol_jtvf:.6f}")
    print(f"JTVF Reg OOS Vol   : {vol_reg:.6f}")
    
    res_df.to_csv("empirical_spectral_results.csv", index=False)
    print("Saved to empirical_spectral_results.csv")