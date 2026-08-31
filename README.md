# Adaptive Markets Lab

> Markets change. How quickly should a quantitative model forget the past?

Adaptive Markets Lab is a point-in-time research platform for testing whether a
model that changes its memory length can outperform fixed-parameter benchmarks
out of sample. The first strategy is exponentially weighted momentum.

This repository is being built as a research project, not a claim of a profitable
live trading system. Results include costs and must be interpreted alongside
robustness tests, regime behaviour, and limitations.

## Current engine

- Configurable momentum half-life and signal threshold
- Signal shifted by one period to prevent look-ahead bias
- Turnover-based transaction costs
- `spot_long_only`: cash or a fully funded long position; no shorting or leverage
- `long_short`: signed positions with configurable gross leverage
- Total/annual return, volatility, Sharpe ratio, drawdown, and turnover metrics
- Deterministic tests for timing and trading constraints

`spot_long_only` is a trading-constraint preset rather than a religious ruling.
It prevents leverage and short-selling, but users must still decide whether the
asset and the wider strategy satisfy their own Sharia-screening requirements.

## Project structure

```text
adaptive-markets-lab/
├── README.md
├── pyproject.toml
├── scripts/
│   └── run_momentum.py
├── src/
│   └── adaptive_markets_lab/
│       ├── __init__.py
│       ├── backtest.py
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       └── momentum.py
└── tests/
    └── test_backtest.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python scripts/run_momentum.py --ticker SPY --half-life 20 --cost-bps 5
```

An explicitly short-enabled experiment is separate:

```bash
python scripts/run_momentum.py --ticker SPY --mode long_short --leverage 2
```

## Research roadmap

1. Fixed half-life sensitivity experiments
2. Walk-forward selection using training data only
3. Fixed-versus-adaptive out-of-sample comparison
4. Regime and robustness diagnostics
5. Streamlit/Plotly interface backed by this independent engine
