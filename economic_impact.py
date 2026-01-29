import numpy as np
import pandas as pd

def calculate_economic_impact(sharpe_lw, sharpe_graph, vol_target=0.15, aum=10_000_000):
    """
    Calculates the dollar value of the performance gap.
    
    Params:
        sharpe_lw: Annualized Sharpe of Benchmark (e.g., 1.46)
        sharpe_graph: Annualized Sharpe of Graph Method (e.g., 1.06)
        vol_target: Target volatility for the fund (15% is standard for hedge funds)
        aum: Assets Under Management ($10M)
    """
    
    print(f"\n--- ECONOMIC IMPACT REPORT (AUM: ${aum/1e6}M) ---")
    
    # 1. Alpha Generation (The value of the gap)
    # Excess Return = Sharpe * Volatility
    # Alpha Diff = (S_lw - S_graph) * Vol
    alpha_diff_bps = (sharpe_lw - sharpe_graph) * vol_target * 10000
    dollar_value = (sharpe_lw - sharpe_graph) * vol_target * aum
    
    print(f"1. Performance Gap: {alpha_diff_bps:.0f} basis points (bps) per year")
    print(f"2. Dollar Value of Alpha: ${dollar_value:,.2f}")
    
    # 2. The "Complexity Tax"
    # If a fund used the Graph method, how much would they lose?
    print(f"3. The 'Complexity Tax':")
    print(f"   Using the Graph method costs a ${aum/1e6}M fund approx.")
    print(f"   ${dollar_value:,.2f} per year in lost risk-adjusted returns.")
    
    # 3. Fee Equivalent
    # Hedge funds charge 2% management fee. 
    # Is the gap larger than the fee?
    fee_equiv = (dollar_value / aum) * 100
    print(f"4. Management Fee Equivalent: {fee_equiv:.2f}%")
    if fee_equiv > 2.0:
        print("   -> CRITICAL: The underperformance exceeds typical management fees.")

# USE YOUR REAL ROLLING NUMBERS HERE
# Example based on your rolling test results:
calculate_economic_impact(sharpe_lw=1.46, sharpe_graph=1.06)