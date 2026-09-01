"""Command-line entry point for a reproducible research run."""

import argparse
from datetime import date

from .config import BacktestConfig, PairsConfig, TradingMode, WalkForwardConfig
from .data import (
    download_adjusted_close,
    download_adjusted_pair,
    generate_cointegrated_pair,
    generate_regime_prices,
)
from .research import (
    run_pairs_research,
    run_research,
    save_pairs_artifacts,
    save_research_artifacts,
)


def _half_lives(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "half-lives must be comma-separated numbers"
        ) from error
    if not parsed:
        raise argparse.ArgumentTypeError("at least one half-life is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible momentum or pairs research study"
    )
    parser.add_argument("--strategy", choices=["momentum", "pairs"], default="momentum")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--demo", action="store_true", help="use offline synthetic data"
    )
    source.add_argument("--ticker", help="download one ticker from Yahoo Finance")
    parser.add_argument("--ticker-b", help="second Yahoo ticker for pairs trading")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--demo-periods", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--half-lives", type=_half_lives, default=(5.0, 10.0, 20.0, 40.0, 80.0)
    )
    parser.add_argument("--fixed-half-life", type=float, default=20.0)
    parser.add_argument("--train-periods", type=int, default=504)
    parser.add_argument("--test-periods", type=int, default=63)
    parser.add_argument(
        "--expanding",
        action="store_true",
        help="use expanding rather than rolling training",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--pair-lookback", type=int, default=60)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TradingMode],
        default=TradingMode.SPOT_LONG_ONLY.value,
    )
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--output", default="results/latest")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    use_demo = args.demo or not args.ticker
    backtest = BacktestConfig(
        trading_mode=TradingMode(args.mode),
        gross_leverage=args.leverage,
        transaction_cost_bps=args.cost_bps,
    )
    if args.strategy == "momentum":
        prices = (
            generate_regime_prices(args.demo_periods, args.seed)
            if use_demo
            else download_adjusted_close(args.ticker, args.start, args.end)
        )
        walk_forward = WalkForwardConfig(
            candidate_half_lives=args.half_lives,
            train_periods=args.train_periods,
            test_periods=args.test_periods,
            expanding=args.expanding,
        )
        result = run_research(
            prices,
            walk_forward,
            backtest,
            fixed_half_life=args.fixed_half_life,
            threshold=args.threshold,
        )
        save_research_artifacts(result, args.output)
        source_name = "synthetic momentum demo" if use_demo else args.ticker.upper()
    else:
        if not use_demo and not args.ticker_b:
            raise ValueError("--ticker-b is required for a real-data pairs study")
        prices = (
            generate_cointegrated_pair(args.demo_periods, args.seed)
            if use_demo
            else download_adjusted_pair(
                args.ticker, args.ticker_b, args.start, args.end
            )
        )
        pair_model = PairsConfig(
            lookback=args.pair_lookback,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
        )
        result = run_pairs_research(prices, pair_model, backtest)
        save_pairs_artifacts(result, args.output)
        source_name = (
            "synthetic pairs demo"
            if use_demo
            else f"{args.ticker.upper()}/{args.ticker_b.upper()}"
        )

    print(f"\nAdaptive Markets Lab | {source_name}")
    print(
        f"Evaluation: {result.equity_curves.index[0].date()} to "
        f"{result.equity_curves.index[-1].date()}"
    )
    print("\n" + result.comparison.round(4).to_string())
    print(f"\nSaved reproducible tables to {args.output}")
    if use_demo:
        print("Synthetic results are a pipeline demonstration, not market evidence.")


if __name__ == "__main__":
    main()
