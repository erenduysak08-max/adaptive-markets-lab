"""Streamlit interface for Adaptive Markets Lab."""

import inspect
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from adaptive_markets_lab import (
    BacktestConfig,
    MomentumConfig,
    PairsConfig,
    TradingMode,
    WalkForwardConfig,
)
from adaptive_markets_lab.data import (
    download_adjusted_close,
    download_adjusted_pair,
    generate_cointegrated_pair,
    generate_regime_prices,
)
from adaptive_markets_lab.momentum import momentum_score, momentum_signal
from adaptive_markets_lab.pairs import (
    pairs_target_state,
    rolling_ols_zscore,
    run_pairs_backtest,
)
from adaptive_markets_lab.research import (
    run_pairs_research,
    run_research,
)
from adaptive_markets_lab.sensitivity import (
    MOMENTUM_PARAMETER_VALUES,
    PAIRS_PARAMETER_VALUES,
    SUPPORTED_METRICS,
    momentum_parameter_surface,
    pairs_parameter_surface,
)
from adaptive_markets_lab.walk_forward import run_walk_forward

st.set_page_config(page_title="Adaptive Markets Lab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_momentum_prices(
    source: str,
    ticker: str,
    start: str,
    end: str,
    demo_periods: int,
    seed: int,
) -> pd.Series:
    if source == "Offline synthetic demo":
        return generate_regime_prices(demo_periods, seed)
    return download_adjusted_close(ticker, start, end)


@st.cache_data(show_spinner=False)
def load_pair_prices(
    source: str,
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
    demo_periods: int,
    seed: int,
) -> pd.DataFrame:
    if source == "Offline synthetic demo":
        return generate_cointegrated_pair(demo_periods, seed)
    return download_adjusted_pair(ticker_a, ticker_b, start, end)


def format_metrics(frame: pd.DataFrame):
    percentage_columns = [
        "total_return",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
    ]
    return frame.style.format(
        {
            **{column: "{:.2%}" for column in percentage_columns},
            "sharpe_ratio": "{:.2f}",
            "annual_turnover": "{:.2f}×",
        }
    )


def exposure_periods(labels: pd.Series) -> pd.DataFrame:
    """Summarise contiguous invested and out-of-market intervals."""
    if labels.empty:
        return pd.DataFrame(columns=["state", "start", "end", "observations"])
    groups = labels.ne(labels.shift()).cumsum()
    rows = []
    for _, values in labels.groupby(groups):
        rows.append(
            {
                "state": values.iloc[0],
                "start": values.index[0],
                "end": values.index[-1],
                "observations": len(values),
            }
        )
    return pd.DataFrame(rows)


def growth_chart(equity_curves: pd.DataFrame):
    equity = (
        equity_curves.rename_axis("date")
        .reset_index()
        .melt("date", var_name="model", value_name="growth of £1")
    )
    figure = px.line(
        equity,
        x="date",
        y="growth of £1",
        color="model",
        title="Growth of £1 on the identical evaluation window",
    )
    figure.update_layout(legend_title_text="", hovermode="x unified")
    return figure


def heatmap_table(surface: pd.DataFrame, metric_label: str):
    metric = SUPPORTED_METRICS[metric_label]
    number_format = "{:.2f}" if metric == "sharpe_ratio" else "{:.2%}"
    return surface.style.format(number_format, na_rep="—").background_gradient(
        cmap="RdYlGn", axis=None
    )


st.title("Adaptive Markets Lab")
st.caption(
    "Test adaptive momentum and rolling-beta pairs strategies without hiding "
    "timing, costs or negative results."
)

with st.sidebar:
    st.header("Experiment")
    strategy = st.selectbox(
        "Strategy",
        ["Adaptive momentum", "Pairs trading"],
        help="Choose the research model to configure and test.",
    )
    source = st.radio(
        "Data",
        ["Offline synthetic demo", "Yahoo Finance"],
        help=(
            "The offline data is reproducible. Yahoo Finance downloads real "
            "adjusted closes."
        ),
    )
    is_live = source == "Yahoo Finance"
    if strategy == "Adaptive momentum":
        ticker = st.text_input(
            "Ticker",
            "SPY",
            disabled=not is_live,
            help="Yahoo Finance symbol for the asset, such as SPY, AAPL or MSFT.",
        )
        ticker_a, ticker_b = "", ""
    else:
        ticker_a = st.text_input(
            "First ticker",
            "KO",
            disabled=not is_live,
            help="Asset A in the rolling regression and spread definition.",
        )
        ticker_b = st.text_input(
            "Second ticker",
            "PEP",
            disabled=not is_live,
            help="Asset B, used as the explanatory leg in the rolling regression.",
        )
        ticker = ""

    start = st.date_input(
        "Start",
        date(2014, 1, 1),
        disabled=not is_live,
        help="First date requested from Yahoo Finance.",
    )
    end = st.date_input(
        "End",
        date.today(),
        disabled=not is_live,
        help="Exclusive end date requested from Yahoo Finance.",
    )
    demo_periods = st.slider(
        "Synthetic observations",
        800,
        3_000,
        1_500,
        100,
        disabled=is_live,
        help=(
            "Number of business-day observations in the deterministic offline dataset."
        ),
    )
    seed = st.number_input(
        "Synthetic seed",
        0,
        10_000,
        7,
        disabled=is_live,
        help="Using the same seed recreates exactly the same synthetic prices.",
    )

    st.divider()
    if strategy == "Adaptive momentum":
        half_lives = st.multiselect(
            "Candidate half-lives (days)",
            [3, 5, 10, 20, 40, 60, 80, 120],
            default=[5, 10, 20, 40, 80],
            help="Memory lengths considered at each walk-forward selection date.",
        )
        fixed_half_life = st.slider(
            "Fixed benchmark half-life",
            1,
            252,
            20,
            help="Unchanging half-life used as the predeclared momentum benchmark.",
        )
        threshold = st.slider(
            "Signal threshold",
            0.0,
            0.01,
            0.0,
            0.00025,
            format="%.5f",
            help=(
                "Minimum EWMA return required to enter; larger values trade less often."
            ),
        )
        train_periods = st.slider(
            "Rolling training observations",
            126,
            1_260,
            504,
            21,
            help="Past observations used to choose the next test block's half-life.",
        )
        test_periods = st.slider(
            "Test block observations",
            21,
            252,
            63,
            21,
            help="Unseen observations for which the selected half-life remains frozen.",
        )
        expanding = st.checkbox(
            "Use expanding training history",
            value=False,
            help=(
                "When enabled, every selection uses all history rather than a "
                "rolling window."
            ),
        )
        pair_lookback, entry_z, exit_z = 60, 2.0, 0.5
    else:
        pair_lookback = st.slider(
            "Regression lookback",
            20,
            252,
            60,
            5,
            help=(
                "Trailing observations used to estimate beta, intercept and "
                "residual scale."
            ),
        )
        entry_z = st.slider(
            "Entry z-score",
            0.5,
            4.0,
            2.0,
            0.25,
            help=(
                "Absolute residual z-score required before opening a "
                "relative-value trade."
            ),
        )
        exit_z = st.slider(
            "Exit z-score",
            0.0,
            2.0,
            0.5,
            0.25,
            help="Close the trade after the residual converges through this level.",
        )
        half_lives, fixed_half_life = [20], 20
        threshold, train_periods, test_periods, expanding = 0.0, 504, 63, False

    cost_bps = st.slider(
        "Transaction cost (basis points)",
        0.0,
        50.0,
        5.0,
        0.5,
        help="Cost paid per unit of absolute position turnover; 5 bps equals 0.05%.",
    )
    mode_label = st.selectbox(
        "Trading constraint",
        ["Spot long-only", "Long-short"],
        help=(
            "Spot mode never shorts or leverages. Pair spot mode rotates into the "
            "relatively undervalued asset rather than creating a market-neutral spread."
        ),
    )
    mode = (
        TradingMode.SPOT_LONG_ONLY
        if mode_label == "Spot long-only"
        else TradingMode.LONG_SHORT
    )
    leverage = st.number_input(
        "Gross leverage",
        0.5,
        5.0,
        1.0,
        0.5,
        disabled=mode is TradingMode.SPOT_LONG_ONLY,
        help="Maximum sum of absolute position weights in long-short mode.",
    )
    run = st.button("Run research study", type="primary", width="stretch")

if strategy == "Adaptive momentum" and not half_lives:
    st.info("Choose at least one candidate half-life.")
    st.stop()
if strategy == "Pairs trading" and exit_z >= entry_z:
    st.info("The exit z-score must be smaller than the entry z-score.")
    st.stop()

existing = st.session_state.get("study_bundle")
needs_run = run or existing is None or existing["strategy"] != strategy
if needs_run:
    try:
        with st.spinner("Running the costed research engine…"):
            backtest = BacktestConfig(
                trading_mode=mode,
                gross_leverage=(
                    1.0 if mode is TradingMode.SPOT_LONG_ONLY else float(leverage)
                ),
                transaction_cost_bps=cost_bps,
            )
            if strategy == "Adaptive momentum":
                prices = load_momentum_prices(
                    source, ticker, str(start), str(end), int(demo_periods), int(seed)
                )
                walk_forward = WalkForwardConfig(
                    candidate_half_lives=tuple(float(value) for value in half_lives),
                    train_periods=train_periods,
                    test_periods=test_periods,
                    expanding=expanding,
                )
                base_model = MomentumConfig(
                    half_life=float(fixed_half_life), threshold=threshold
                )
                result = run_research(
                    prices,
                    walk_forward,
                    backtest,
                    fixed_half_life=fixed_half_life,
                    threshold=threshold,
                )
                bundle = {
                    "strategy": strategy,
                    "source": source,
                    "prices": prices,
                    "result": result,
                    "model": base_model,
                    "backtest": backtest,
                    "walk_forward": walk_forward,
                }
            else:
                prices = load_pair_prices(
                    source,
                    ticker_a,
                    ticker_b,
                    str(start),
                    str(end),
                    int(demo_periods),
                    int(seed),
                )
                base_model = PairsConfig(
                    lookback=pair_lookback, entry_z=entry_z, exit_z=exit_z
                )
                result = run_pairs_research(prices, base_model, backtest)
                bundle = {
                    "strategy": strategy,
                    "source": source,
                    "prices": prices,
                    "result": result,
                    "model": base_model,
                    "backtest": backtest,
                }
            st.session_state["study_bundle"] = bundle
            st.session_state.pop("heatmap_bundle", None)
    except Exception as error:
        st.error(f"Experiment could not run: {error}")
        st.stop()

bundle = st.session_state["study_bundle"]
result = bundle["result"]
if bundle["source"] == "Offline synthetic demo":
    st.warning(
        "Synthetic data demonstrates the pipeline. Its results are not evidence "
        "about real markets."
    )

tabs = st.tabs(
    [
        "Overview",
        "Performance",
        "Exposure & trades",
        "Parameter heatmap",
        "Diagnostics",
        "Strategy code",
        "Methodology",
    ]
)

with tabs[0]:
    if strategy == "Adaptive momentum":
        primary_name = "Adaptive momentum"
        st.markdown(
            "The walk-forward model selects its momentum memory using past data "
            "and freezes that choice for the next unseen block."
        )
    else:
        primary_name = result.strategy_name
        pair = result.pair
        st.markdown(
            f"The model estimates `{pair.asset_a} = intercept + beta × "
            f"{pair.asset_b} + residual` on a rolling window and trades residual "
            "convergence."
        )
        if bundle["backtest"].trading_mode is TradingMode.SPOT_LONG_ONLY:
            st.info(
                "Spot mode is a long-only relative-value rotation. Traditional "
                "market-neutral pairs trading is available in Long-short mode."
            )

    primary_metrics = result.comparison.loc[primary_name]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total return", f"{primary_metrics['total_return']:.1%}")
    metric_columns[1].metric("Annual return", f"{primary_metrics['annual_return']:.1%}")
    metric_columns[2].metric("Sharpe ratio", f"{primary_metrics['sharpe_ratio']:.2f}")
    metric_columns[3].metric(
        "Maximum drawdown", f"{primary_metrics['max_drawdown']:.1%}"
    )
    st.subheader("Comparable metrics")
    st.dataframe(format_metrics(result.comparison.copy()), width="stretch")

with tabs[1]:
    st.plotly_chart(growth_chart(result.equity_curves), width="stretch")
    st.caption(
        "Every line begins on the same evaluation date. Transaction costs are "
        "included before compounding."
    )

with tabs[2]:
    if strategy == "Adaptive momentum":
        frame = result.adaptive.frame
        position = frame["position"]
        labels = pd.Series("Out of market", index=frame.index)
        labels.loc[position > 0] = "Long / invested"
        labels.loc[position < 0] = "Short / invested"
        exposure = position.abs().rename("gross exposure").reset_index()
        exposure.columns = ["date", "gross exposure"]
        exposure_figure = px.area(
            exposure,
            x="date",
            y="gross exposure",
            title="When the strategy was invested or out of the market",
        )
        exposure_figure.update_traces(line_shape="hv")
        st.plotly_chart(exposure_figure, width="stretch")

        score = frame["momentum_score"].rename("EWMA momentum").reset_index()
        score.columns = ["date", "EWMA momentum"]
        score_figure = px.line(score, x="date", y="EWMA momentum")
        score_figure.add_hline(y=bundle["model"].threshold, line_dash="dash")
        if bundle["backtest"].trading_mode is TradingMode.LONG_SHORT:
            score_figure.add_hline(y=-bundle["model"].threshold, line_dash="dash")
        st.plotly_chart(score_figure, width="stretch")
    else:
        frame = result.pair.frame.iloc[bundle["model"].lookback :]
        state = frame["state"]
        labels = pd.Series("Out of market", index=frame.index)
        if bundle["backtest"].trading_mode is TradingMode.SPOT_LONG_ONLY:
            labels.loc[state > 0] = f"Long {result.pair.asset_a}"
            labels.loc[state < 0] = f"Long {result.pair.asset_b}"
        else:
            labels.loc[state > 0] = "Long spread"
            labels.loc[state < 0] = "Short spread"

        weights = frame[["weight_a", "weight_b"]].rename(
            columns={
                "weight_a": result.pair.asset_a,
                "weight_b": result.pair.asset_b,
            }
        )
        weight_data = (
            weights.rename_axis("date")
            .reset_index()
            .melt("date", var_name="asset", value_name="portfolio weight")
        )
        weight_figure = px.line(
            weight_data,
            x="date",
            y="portfolio weight",
            color="asset",
            title="Held pair weights after the one-day execution delay",
        )
        weight_figure.update_traces(line_shape="hv")
        st.plotly_chart(weight_figure, width="stretch")

        z_data = frame["z_score"].rename("residual z-score").reset_index()
        z_data.columns = ["date", "residual z-score"]
        z_figure = px.line(z_data, x="date", y="residual z-score")
        z_figure.add_hline(y=bundle["model"].entry_z, line_dash="dash")
        z_figure.add_hline(y=-bundle["model"].entry_z, line_dash="dash")
        z_figure.add_hline(y=bundle["model"].exit_z, line_dash="dot")
        z_figure.add_hline(y=-bundle["model"].exit_z, line_dash="dot")
        st.plotly_chart(z_figure, width="stretch")

    st.subheader("Invested and out-of-market intervals")
    st.dataframe(exposure_periods(labels), width="stretch", hide_index=True)

with tabs[3]:
    st.markdown(
        "Choose two different parameters. Each coloured cell reruns the selected "
        "strategy using that exact combination. Green is a higher value of the "
        "chosen metric; red is lower."
    )
    parameter_values = (
        MOMENTUM_PARAMETER_VALUES
        if strategy == "Adaptive momentum"
        else PAIRS_PARAMETER_VALUES
    )
    parameter_names = list(parameter_values)
    controls = st.columns(3)
    x_parameter = controls[0].selectbox(
        "Horizontal-axis variable",
        parameter_names,
        index=0,
        help="Parameter varied across the table's columns.",
    )
    y_parameter = controls[1].selectbox(
        "Vertical-axis variable",
        parameter_names,
        index=1,
        help="Parameter varied down the table's rows.",
    )
    metric_label = controls[2].selectbox(
        "Cell metric",
        list(SUPPORTED_METRICS),
        help="Statistic calculated independently for every parameter combination.",
    )
    x_values = st.multiselect(
        "Horizontal-axis values",
        parameter_values[x_parameter],
        default=parameter_values[x_parameter],
        help="Values to evaluate for the column parameter.",
    )
    y_values = st.multiselect(
        "Vertical-axis values",
        parameter_values[y_parameter],
        default=parameter_values[y_parameter],
        help="Values to evaluate for the row parameter.",
    )
    calculate = st.button("Calculate colour-scale table")
    signature = (
        strategy,
        x_parameter,
        tuple(x_values),
        y_parameter,
        tuple(y_values),
        metric_label,
    )
    if x_parameter == y_parameter:
        st.warning("Choose different variables for the horizontal and vertical axes.")
    elif not x_values or not y_values:
        st.warning("Choose at least one value for each axis.")
    elif calculate:
        with st.spinner("Evaluating every parameter combination…"):
            if strategy == "Adaptive momentum":
                surface = momentum_parameter_surface(
                    bundle["prices"],
                    result.adaptive.frame.index,
                    x_parameter,
                    tuple(float(value) for value in x_values),
                    y_parameter,
                    tuple(float(value) for value in y_values),
                    bundle["model"],
                    bundle["backtest"],
                    SUPPORTED_METRICS[metric_label],
                )
            else:
                surface = pairs_parameter_surface(
                    bundle["prices"],
                    x_parameter,
                    tuple(float(value) for value in x_values),
                    y_parameter,
                    tuple(float(value) for value in y_values),
                    bundle["model"],
                    bundle["backtest"],
                    SUPPORTED_METRICS[metric_label],
                )
            st.session_state["heatmap_bundle"] = {
                "signature": signature,
                "surface": surface,
            }
    stored_heatmap = st.session_state.get("heatmap_bundle")
    if stored_heatmap and stored_heatmap["signature"] == signature:
        st.dataframe(
            heatmap_table(stored_heatmap["surface"], metric_label),
            width="stretch",
        )
        if strategy == "Adaptive momentum":
            st.caption(
                "This is a fixed-momentum parameter surface over the same unseen "
                "dates as the adaptive comparison. It does not reselect candidates."
            )
        else:
            st.caption(
                "All cells use the same date range after the largest requested "
                "regression warm-up. Invalid entry/exit combinations are blank."
            )

with tabs[4]:
    if strategy == "Adaptive momentum":
        diagnostic_tabs = st.tabs(
            ["Walk-forward folds", "Candidate scores", "Regimes", "Uncertainty"]
        )
        with diagnostic_tabs[0]:
            st.dataframe(result.adaptive.selections, width="stretch", hide_index=True)
        with diagnostic_tabs[1]:
            st.dataframe(
                result.adaptive.candidate_scores, width="stretch", hide_index=True
            )
        with diagnostic_tabs[2]:
            st.dataframe(result.regime_metrics, width="stretch", hide_index=True)
        with diagnostic_tabs[3]:
            st.dataframe(result.uncertainty, width="stretch")
    else:
        diagnostic_tabs = st.tabs(
            ["Rolling regression", "Trade statistics", "Uncertainty"]
        )
        pair_frame = result.pair.frame.iloc[bundle["model"].lookback :]
        with diagnostic_tabs[0]:
            st.dataframe(
                pair_frame[["hedge_ratio", "intercept", "z_score"]], width="stretch"
            )
        with diagnostic_tabs[1]:
            trade_summary = pd.DataFrame(
                {
                    "value": {
                        "Invested observations": int(
                            (pair_frame["gross_exposure"] > 0).sum()
                        ),
                        "Out-of-market observations": int(
                            (pair_frame["gross_exposure"] == 0).sum()
                        ),
                        "Average gross exposure": pair_frame["gross_exposure"].mean(),
                        "Total turnover": pair_frame["turnover"].sum(),
                        "Total transaction costs": pair_frame["transaction_cost"].sum(),
                    }
                }
            )
            st.dataframe(trade_summary, width="stretch")
        with diagnostic_tabs[2]:
            st.dataframe(result.uncertainty, width="stretch")

with tabs[5]:
    st.markdown(
        "These are the actual core functions imported by the running dashboard, "
        "rather than a simplified pseudocode copy."
    )
    if strategy == "Adaptive momentum":
        code_tabs = st.tabs(["Momentum score", "Trading signal", "Walk-forward loop"])
        with code_tabs[0]:
            st.code(inspect.getsource(momentum_score), language="python")
        with code_tabs[1]:
            st.code(inspect.getsource(momentum_signal), language="python")
        with code_tabs[2]:
            st.code(inspect.getsource(run_walk_forward), language="python")
    else:
        code_tabs = st.tabs(["Rolling regression", "Entry and exit", "Backtest"])
        with code_tabs[0]:
            st.code(inspect.getsource(rolling_ols_zscore), language="python")
        with code_tabs[1]:
            st.code(inspect.getsource(pairs_target_state), language="python")
        with code_tabs[2]:
            st.code(inspect.getsource(run_pairs_backtest), language="python")

with tabs[6]:
    if strategy == "Adaptive momentum":
        st.markdown(
            """
### Adaptive momentum

Each fold scores candidate half-lives using observations ending before its test
block. The winner is frozen for that block. The exponentially weighted momentum
signal observed after close $t$ becomes the position for the next return, so it
cannot earn a return that has already occurred.

The heatmap is a diagnostic fixed-model surface. It is kept separate from the
walk-forward selection to avoid quietly redefining the strategy after seeing the
test results.
            """
        )
    else:
        st.markdown(
            """
### Rolling-beta pairs trading

The first log price is regressed on the second over a trailing window with an
intercept. The newest residual is divided by the regression residual standard
deviation. A large positive residual opens a short-spread trade; a large negative
residual opens a long-spread trade. The position closes after convergence through
the exit threshold.

Long-short weights are normalised to the chosen gross leverage. Spot mode holds
only the relatively undervalued leg and must not be described as market-neutral.
            """
        )
    st.markdown(
        """
### Shared limitations

This is a daily close-to-close educational model. It omits intraday execution,
bid-ask spread dynamics, market impact, financing, borrow availability and taxes.
Yahoo Finance is convenient rather than institutional point-in-time data. The
results and colour table are research diagnostics, not investment advice or a
claim of future profitability.
        """
    )
