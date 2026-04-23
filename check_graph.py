import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLassoCV

# Load just a chunk of data to test
df = pd.read_parquet("data/1s_data_2025.parquet")
ti_cols = [c for c in df.columns if '_ti' in c]
tis = df[ti_cols].iloc[:100000] # Just first 100k seconds

# Standardize
tis_std = (tis - tis.mean()) / tis.std()

# Fit Graph
print("Fitting Graph...")
gl = GraphicalLassoCV(alphas=[0.1, 0.2, 0.5], cv=3)
gl.fit(tis_std)

precision = gl.precision_
adjacency = np.abs(precision)
np.fill_diagonal(adjacency, 0)

# Count Edges
n_edges = np.sum(adjacency > 1e-5) / 2
print(f"Number of Edges found: {n_edges}")
print(f"Precision Matrix Density: {n_edges / 45:.2%}")