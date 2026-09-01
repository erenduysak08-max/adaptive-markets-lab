# Adaptive Markets Lab

[![tests](https://github.com/erenduysak08-max/adaptive-markets-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/erenduysak08-max/adaptive-markets-lab/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://erenduysak08-max-adaptive-markets-lab-app-nogckh.streamlit.app)

## Try the app

The easiest way to use the project is through the live dashboard:

### [Open Adaptive Markets Lab](https://erenduysak08-max-adaptive-markets-lab-app-nogckh.streamlit.app)

There is nothing to install. In the sidebar:

1. Choose **Adaptive momentum** or **Pairs trading**.
2. Use the reproducible offline demo or enter Yahoo Finance ticker symbols.
3. Change the model parameters and trading constraints.
4. Press **Run research study**.
5. Use the tabs to inspect performance, exposure, parameter heatmaps,
   diagnostics and the strategy code.

The offline demo is selected by default, so the app still works if live market
data is temporarily unavailable.

## Why I made this

I built this project while learning more about quantitative research before
starting my Mathematics degree at Warwick. I wanted to make something more
useful than a backtest which only shows one profitable-looking graph.

The main question I wanted to investigate was:

> If market behaviour changes, can a model improve by changing how much past
> data it remembers?

I also wanted to understand pairs trading, rolling regression and how easy it
is to accidentally introduce look-ahead bias into a backtest.

This project is not intended to claim that either strategy will make money. It
is a research and learning project which makes the assumptions, model choices
and unsuccessful results visible.

## Strategies

### Adaptive momentum

The momentum strategy calculates an exponentially weighted average of past
returns. A short half-life reacts quickly to recent prices, while a long
half-life changes more slowly.

For each walk-forward fold, the program:

1. Tests several half-lives using the training period.
2. Selects the best training Sharpe ratio.
3. Freezes that choice for the next unseen test period.
4. Repeats the process and joins only the unseen periods together.

This result is compared with fixed 20-day momentum and buy-and-hold over the
same dates.

### Rolling-beta pairs trading

The pairs strategy estimates a rolling relationship between two log-price
series:

```text
Asset A = intercept + beta × Asset B + residual
```

The latest residual is converted into a z-score. The strategy enters when the
spread is far enough from its rolling average and exits when it moves back
towards it.

Long-short mode trades both legs and normalises their combined exposure. The
spot long-only option never shorts: it rotates into the relatively undervalued
asset. I describe this as long-only rotation rather than market-neutral pairs
trading.

## What the dashboard includes

- Offline synthetic data that is reproducible from a fixed seed
- Live adjusted price data for one ticker or a ticker pair
- Long-only and explicitly enabled long-short modes
- Adjustable transaction costs and gross leverage
- Walk-forward model selection
- Equity curves and comparable performance statistics
- Invested and out-of-market timelines
- Rolling hedge ratio, spread z-score and portfolio weights
- Two-parameter colour-scale return tables
- Moving-block bootstrap confidence intervals
- The Python source code used by each strategy
- Explanations beside the main model controls

## Avoiding common backtesting mistakes

I tried to make the timing of the program clear and testable:

- A signal calculated after close on day `t` is shifted before it earns a return.
- Training data always ends before its related test period begins.
- Transaction costs are charged when portfolio weights change.
- Competing strategies are compared over identical dates.
- Pair turnover includes changes in both legs and the rolling hedge ratio.
- A future-mutation test changes later prices and checks that earlier results
  remain unchanged.
- Heatmaps use a common evaluation window so cells remain comparable.

The full equations and assumptions are in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Running it locally

The hosted app above is the quickest way to use the project. To run the code
locally instead:

```bash
git clone https://github.com/erenduysak08-max/adaptive-markets-lab.git
cd adaptive-markets-lab
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install and start the app:

```bash
python -m pip install -e ".[app]"
streamlit run app.py
```

The command-line demo can also be run without downloading market data:

```bash
aml-research --demo
aml-research --strategy pairs --demo --output results/pairs-latest
```

## Testing

```bash
python -m pip install -e ".[dev,app]"
python -m ruff check .
python -m pytest -q
```

GitHub Actions runs the tests and command-line smoke studies after every push.

## Repository layout

```text
adaptive-markets-lab/
├── app.py                         # Streamlit dashboard
├── src/adaptive_markets_lab/
│   ├── backtest.py                # Lagged and costed backtest engine
│   ├── walk_forward.py            # Past-only model selection
│   ├── momentum.py                # Momentum score and signals
│   ├── pairs.py                   # Rolling regression pairs strategy
│   ├── sensitivity.py             # Two-parameter surfaces
│   ├── research.py                # Comparisons and saved results
│   ├── metrics.py                 # Performance and bootstrap statistics
│   ├── data.py                    # Live and synthetic data
│   └── cli.py                     # Command-line interface
├── tests/                         # Unit and leakage tests
├── results/                       # Reproducible demonstration outputs
└── docs/METHODOLOGY.md            # Equations and assumptions
```

## Current limitations

The project works with daily closing prices and studies one asset or one pair
at a time. It does not simulate intraday fills, bid-ask spread changes, market
impact, borrowing availability, taxes or portfolio-level allocation. Yahoo
Finance is useful for a student project but is not an institutional
point-in-time dataset.

The pairs strategy also assumes the user has already chosen a sensible pair. A
future version could separate pair formation from trading and test
cointegration stability using only past data.

## What I learned

This project helped me learn how to:

- Organise Python research code separately from the interface
- Work with pandas time series and rolling regressions
- Build a walk-forward experiment
- Test for timing and information leakage
- Compare parameter choices without hiding the full surface
- Create a Streamlit dashboard and command-line interface from the same engine
- Use automated tests, GitHub Actions and Docker

Feedback and methodological criticism are welcome.

MIT licensed.
