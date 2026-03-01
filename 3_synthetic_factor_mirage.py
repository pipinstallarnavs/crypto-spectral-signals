import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings("ignore")

N_ASSETS = 19
T_OBS = 720
TAU = 1.5

def generate_synthetic_market(base_corr):
    """Generates a known true covariance matrix and samples from it."""
    # True Covariance: Base correlation + diagonal variance
    true_cov = np.full((N_ASSETS, N_ASSETS), base_corr)
    np.fill_diagonal(true_cov, 1.0)
    
    # Generate noisy sample returns from this true matrix
    returns = np.random.multivariate_normal(np.zeros(N_ASSETS), true_cov, size=T_OBS)
    return returns, true_cov

def calculate_metrics(S_est, S_true):
    frob_norm = np.linalg.norm(S_est - S_true, ord='fro')
    evals = np.linalg.eigvalsh(S_est)
    cond = evals[-1] / (evals[0] if evals[0] > 0 else 1e-10)
    return frob_norm, cond

def run_synthetic_test(name, base_corr):
    print(f"\n--- {name} (Base Correlation: {base_corr}) ---")
    returns, true_cov = generate_synthetic_market(base_corr)
    
    # 1. Ledoit-Wolf
    S_lw = LedoitWolf().fit(returns).covariance_
    frob_lw, cond_lw = calculate_metrics(S_lw, true_cov)
    
    # 2. GSP Vertex Filter (Assuming fully connected graph for synthetic)
    adj = np.ones((N_ASSETS, N_ASSETS)) - np.eye(N_ASSETS)
    degrees = adj.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = np.eye(N_ASSETS) - (D_inv_sqrt @ adj @ D_inv_sqrt)
    
    evals, evecs = np.linalg.eigh(L_norm)
    G_filter = evecs @ np.diag(np.exp(-TAU * evals)) @ evecs.T
    X_projected = returns @ G_filter
    S_jtvf = pd.DataFrame(X_projected).cov().values
    
    frob_jtvf, cond_jtvf = calculate_metrics(S_jtvf, true_cov)
    
    print(f"Ledoit-Wolf | Frob Error: {frob_lw:.2f} | Cond: {cond_lw:,.2f}")
    print(f"JTVF Filter | Frob Error: {frob_jtvf:.2f} | Cond: {cond_jtvf:,.2f}")

if __name__ == "__main__":
    np.random.seed(42)
    run_synthetic_test("World A: High Systemic Risk", base_corr=0.8)
    run_synthetic_test("World B: Low Systemic Risk (Stock Picker Market)", base_corr=0.2)