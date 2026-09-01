# Methodology and assumptions

This note states both experiments precisely enough for a reader to reproduce
them, challenge them and identify what they do not establish.

## 1. Research questions

1. Does selecting the memory length of a momentum signal using only recent
   training data improve subsequent unseen performance relative to a fixed
   memory length?
2. For a selected pair, does a large rolling-regression residual subsequently
   converge enough to survive execution delay and two-leg transaction costs?

Neither experiment is designed to predict a particular price, claim market
efficiency is false or search until a profitable chart is found.

## 2. Adaptive momentum signal

Let the close-to-close simple return be

$$
r_t = \frac{P_t}{P_{t-1}} - 1.
$$

For half-life $h$, pandas' exponentially weighted mean uses

$$
\alpha = 1 - 2^{-1/h},
\qquad
m_t = (1-\alpha)m_{t-1} + \alpha r_t.
$$

With threshold $q \geq 0$, spot long-only exposure is

$$
s_t = \mathbf{1}(m_t > q).
$$

Long-short exposure is $+1$ above $q$, $-1$ below $-q$ and zero otherwise.
Gross leverage, when explicitly enabled, scales that signal.

## 3. Rolling-beta pairs signal

Let $y_t=\log(P^A_t)$ and $x_t=\log(P^B_t)$. Over the trailing $L$
observations, the engine estimates an ordinary least-squares regression with an
intercept:

$$
y_i = a_t + \beta_t x_i + e_i.
$$

The newest residual is standardised using the sample standard deviation of the
residuals from that same regression window:

$$
z_t=\frac{e_t}{\mathrm{sd}(e_{t-L+1},\ldots,e_t)}.
$$

When $z_t$ rises above the entry threshold, the strategy targets a short-spread
position. Below the negative threshold, it targets a long-spread position. The
position closes after convergence through the configured exit level.

In long-short mode, asset weights are scaled so
$|w^A_t|+|w^B_t|$ equals the requested gross leverage. Because both weights vary
with the rolling hedge ratio, hedge rebalancing also creates turnover. Spot mode
instead holds only the relatively undervalued leg. It is a long-only rotation,
not a market-neutral pairs trade. The public backtest and research APIs, CLI and
dashboard all default to spot long-only; long-short mode must be requested
explicitly. Even in long-short mode, the construction is not automatically
market-neutral.

## 4. Timing and costs

A score observed only after close $t$ cannot earn the return that has already
occurred at $t$. Every target is therefore shifted by one observation:

$$
x_t=s_{t-1}.
$$

For proportional cost $c$, single-asset net return is

$$
R_t=x_t r_t-c|x_t-x_{t-1}|.
$$

For the pair, gross return is $w^A_t r^A_t+w^B_t r^B_t$ and turnover is the sum
of absolute weight changes across both legs. This transparent model is not a
substitute for a spread, impact or order-book execution model.

## 5. Walk-forward momentum selection

Default configuration:

- Rolling training window: 504 observations
- Frozen test block: 63 observations
- Candidate half-lives: 5, 10, 20, 40 and 80
- Training objective: annualised Sharpe ratio with zero risk-free rate
- Deterministic tie-break: shorter half-life

At fold $k$, every candidate is evaluated only between `train_start` and
`train_end`. The winner is frozen from `test_start` through `test_end`.
`train_end < test_start` is an enforced and tested invariant. Reported adaptive
statistics concatenate test blocks only.

Here, "out-of-sample" means unseen by the walk-forward parameter-selection
algorithm. It does not mean a final researcher holdout that has remained unseen
throughout project development. Looking at these test blocks and then changing
candidate values, thresholds, assets or dates can produce researcher-level data
snooping. A stronger follow-up would pre-register the experiment and reserve a
final holdout until all choices are frozen.

The saved `selections.csv` and `candidate_scores.csv` provide a complete audit
trail. Expanding training is available but is not the default.

## 6. Baselines, diagnostics and parameter surfaces

The momentum comparison contains walk-forward adaptive momentum, a predeclared
fixed 20-day momentum model and initial-cost-adjusted buy-and-hold. All three use
the exact adaptive out-of-sample index.

The momentum regime label uses the sign of the previous day's trailing
126-observation asset return. Shifting the label prevents today's return from
classifying itself. It is descriptive and is not a trading input.

The pairs strategy is compared with an initial-cost-adjusted buy-and-hold
portfolio over the same post-warm-up dates. Half the initial capital is assigned
to each asset and no rebalancing follows, so the weights drift as the two sleeves
change value. If $G^A_t$ and $G^B_t$ are their cumulative gross growth factors,
the benchmark wealth before costs is

$$
V_t = 0.5G^A_t + 0.5G^B_t.
$$

The first return is charged one unit of turnover at the configured proportional
cost. This is distinct from averaging the two asset returns every day, which
would describe a daily-rebalanced 50/50 portfolio. Pair diagnostics expose the
rolling hedge ratio, intercept, z-score, weights, exposure and turnover.

The colour-scale table reruns the selected model at every requested combination
of two parameters. Momentum surfaces use the adaptive experiment's unseen index
but contain fixed models; they do not retroactively redefine the walk-forward
winner. Pairs surfaces begin after the largest requested regression lookback so
every cell uses the same dates. Invalid entry/exit combinations are blank.

## 7. Metrics and uncertainty

- Total compounded return
- Annualised geometric return
- Annualised sample volatility
- Annualised Sharpe ratio, assuming a zero risk-free rate
- Maximum peak-to-trough drawdown, including initial wealth as the first peak
- Mean daily turnover annualised by 252

The output also reports a 95% circular moving-block bootstrap interval for the
annualised arithmetic mean return difference between the selected strategy and
its baseline. Twenty-observation blocks preserve short-range dependence; a fixed
seed makes the calculation reproducible.

No metric or bootstrap interval alone establishes a positive expected return.
Multiple parameter comparisons, serial dependence and a small number of market
regimes all weaken naive significance claims.

## 8. Reproducibility and leakage checks

The momentum generator changes drift and volatility four times. It does not
directly create changing return autocorrelation or define a known optimal
momentum half-life, so it cannot validate the adaptive model's economic premise.
The pairs generator deliberately creates two correlated log-price paths with a
mean-reverting spread. Positive performance on that dataset verifies the
pipeline against designed-in behaviour and is not evidence of finding market
mean reversion. Both generators accept fixed seeds and exist solely to make the
complete pipeline testable without internet access.

Every CLI study writes `run_config.json` beside its result tables. The file
records the project version, data source, seed and sample length where relevant,
trading constraints, transaction costs and model settings. It contains no
timestamps, so repeated runs with identical inputs remain easy to diff.

The strongest leakage test changes every price after a future cutoff, reruns the
experiment and asserts that every earlier diagnostic, signal, position, cost and
return is identical. It is applied independently to momentum and pairs. Other
tests cover signal delay, entry/exit state, two-leg costs, leverage, spot-only
non-negativity, parameter surfaces, validation, metrics and common indices.

## 9. Known limitations and sensible extensions

The current scope is one asset or one pair. A stronger empirical study would
pre-register assets and dates, test pair formation separately from pair trading,
verify cointegration stability, use survivorship-bias-controlled data, include
spread, borrow and financing estimates, and test whether conclusions survive
across asset classes.

Portfolio construction and an order-book simulator remain separate research
questions and should not be added merely to increase the feature count.
