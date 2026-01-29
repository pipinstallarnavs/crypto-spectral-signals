import matplotlib.pyplot as plt
import numpy as np

# --- DATA INPUT (UPDATE THESE TOMORROW) ---
# Timescales: 1s, 5s, 15s, 1min, 5min, 30min, 1h
timescales = ['1s', '5s', '15s', '1min', '5min', '30min', '1h']
x_pos = np.arange(len(timescales))

# REPLACE THESE LISTS WITH THE OUTPUT FROM timescale_spectrum.py
# Example format based on your previous runs (Update these!)
edges =        [0,    0,    0,    8,     33,    36,     36] 
sharpe_lw =    [-0.49, -0.01, 0.64, 2.41,  5.96,  15.47,  23.69]
sharpe_graph = [-2.30, -4.43, -7.04, -9.98, -5.28,  5.86,  -14.10]

# --- PLOT 1: THE PHASE TRANSITION (Topology) ---
plt.figure(figsize=(10, 6))
# Plot line with markers
plt.plot(x_pos, edges, marker='o', linestyle='-', linewidth=3, color='#2ca02c', label='Graph Density')

# Aesthetics
plt.title("The Epps Effect: Emergence of Network Topology", fontsize=16)
plt.ylabel("Number of Learned Edges", fontsize=14)
plt.xlabel("Sampling Frequency", fontsize=14)
plt.xticks(x_pos, timescales, fontsize=12)
plt.grid(True, alpha=0.3)

# Add "Phase Transition" Annotation (Adjust x-coord if needed)
# Assuming transition happens around 1min (index 3)
plt.axvline(x=3, color='gray', linestyle='--', alpha=0.7)
plt.text(3.1, max(edges)*0.5, 'Phase Transition\n(Structure Emerges)', fontsize=12, color='#555')

plt.tight_layout()
plt.savefig("figure1_topology.png", dpi=300)
print("Saved figure1_topology.png")

# --- PLOT 2: THE COMPLEXITY PENALTY (Performance) ---
plt.figure(figsize=(10, 6))

# Bar configuration
width = 0.35
plt.bar(x_pos - width/2, sharpe_lw, width, label='Robust Shrinkage (Ledoit-Wolf)', color='#1f77b4')
plt.bar(x_pos + width/2, sharpe_graph, width, label='Graph Spectral Filter', color='#d62728')

# Aesthetics
plt.title("The Complexity Penalty: Robustness vs. Structure", fontsize=16)
plt.ylabel("Annualized Sharpe Ratio", fontsize=14)
plt.xlabel("Sampling Frequency", fontsize=14)
plt.xticks(x_pos, timescales, fontsize=12)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, axis='y', alpha=0.3)
plt.axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig("figure2_performance.png", dpi=300)
print("Saved figure2_performance.png")