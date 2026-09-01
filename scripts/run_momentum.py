import argparse

from adaptive_markets_lab import (
    BacktestConfig,
    MomentumConfig,
    TradingMode,
    run_backtest,
)
from adaptive_markets_lab.data import download_adjusted_close


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an adaptive momentum experiment")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--half-life", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TradingMode],
        default=TradingMode.SPOT_LONG_ONLY.value,
    )
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = download_adjusted_close(args.ticker, args.start, args.end)
    result = run_backtest(
        prices,
        MomentumConfig(half_life=args.half_life, threshold=args.threshold),
        BacktestConfig(
            trading_mode=TradingMode(args.mode),
            gross_leverage=args.leverage,
            transaction_cost_bps=args.cost_bps,
        ),
    )
    print(
        f"Experiment: {args.ticker.upper()} | {args.mode} | "
        f"half-life={args.half_life:g}"
    )
    for name, value in result.metrics.items():
        print(f"{name:>20}: {value: .4f}")


if __name__ == "__main__":
    main()
