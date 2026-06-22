![Crypto Spectral Signals banner](assets/banner.svg)

# Crypto Spectral Signals

Crypto Spectral Signals studies whether cross-asset trade-imbalance graphs improve covariance estimation and minimum-variance allocation in cryptocurrency markets.

## Research question

Market dependencies become easier to estimate as observations are aggregated, but added structure can also amplify estimation error. This project compares graph-based spectral filters with Ledoit-Wolf shrinkage across sampling frequencies, rolling windows, and market regimes.

## Pipeline

1. Download public Binance trade archives.
2. Aggregate prices and trade imbalance for a multi-asset universe.
3. Estimate sparse cross-asset structure with graphical lasso.
4. Filter returns in the graph spectral domain.
5. Construct global minimum-variance portfolios.
6. Compare conditioning, risk, Sharpe ratio, turnover, and regime dependence.

## Main experiments

- `1_emperical_spectral_analysis.py`: rolling covariance comparison
- `2_tau_sensitivity_sweep.py`: graph-filter strength sensitivity
- `3_synthetic_factor_mirage.py`: controlled covariance recovery
- `timescale_spectrum.py`: sampling-frequency study
- `rolling_robustness.py`: rolling train and test windows
- `regime_analysis.py`: performance by correlation and volatility regime
- `ablation_study.py`: component-level comparison
- `robustness_checks.py`: stability diagnostics

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python get_binance_data.py
```

Run an experiment from the repository root, for example:

```bash
python timescale_spectrum.py
```

## Data policy

Downloaded market data is excluded from Git. The downloader and experiment code define the collection and transformation process. Users remain responsible for the source provider's terms.

## Scope

The project is an empirical study of covariance estimators. Backtest statistics are sensitive to sampling, transaction costs, and market selection, and should not be interpreted as evidence of a deployable trading edge.
