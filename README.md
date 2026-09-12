# Option Pricing Engine

A from-scratch option pricing engine in Python: analytical **Black-Scholes**, **binomial trees** (CRR), and **Monte Carlo** simulation with variance reduction for exotic options plus the **Greeks**, **implied volatility**, and a reconstructed **3D volatility surface** from real market data.

> Personal quantitative finance project. Focus: stochastic calculus applied to derivatives pricing, numerical methods, and clean, tested, object-oriented code.

## Overview

The engine values a contract by computing the same risk-neutral expectation three different ways, each suited to a class of product:

| Method                                 | Handles                                        | Used for                          |
| -------------------------------------- | ---------------------------------------------- | --------------------------------- |
| **Black–Scholes–Merton (closed form)** | European options + analytical Greeks           | Fast, exact reference             |
| **Binomial tree (CRR)**                | European and American options (early exercise) | Early-exercise premium            |
| **Monte Carlo (variance reduction)**   | Path-dependent / exotic payoffs                | Asian, barrier, digital, lookback |

## Highlights

* 3 independent pricing methods for European options (closed form, lattice, simulation) that agree to ~1e-2.
* American options via CRR trees, with the early-exercise premium recovered explicitly.
* Exotic options: arithmetic / geometric Asian, barriers (up/down, in/out), digital and floating-strike lookback, all by Monte Carlo.
* Payoff-driven engine: `mc_price(payoff, ...)` takes the payoff as an argument — adding a product needs no change to the engine core.
* Variance reduction: antithetic variates + control variate. On the European call, combining both cuts empirical variance by ~98 %, divides the standard error by more than 5, and improves efficiency by ~×33 (Table 10 of the report). On the arithmetic Asian, the geometric-Asian control variate reaches ρ ≈ 0.9996 and cuts the simulation budget by a factor > 1300.
* Greeks analytically (Black–Scholes) and by finite differences on the tree / Monte Carlo, applicable to any payoff including exotics.
* Implied volatility inversion (Newton–Raphson, with Brent fallback when vega is small) and a 3D volatility surface with skew, term structure, a leave-one-out stability test and static no-arbitrage diagnostics (calendar + butterfly).
* Validated against QuantLib and covered by a 44-test pytest suite.

## Project Structure

```text
option-pricing-engine/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── src/                              # reusable, tested pricing modules (no side effects)
│   ├── simulation.py                 # GBM path & terminal simulation
│   ├── black_scholes.py              # closed form + Greeks (dividend q / FX)
│   ├── binomial.py                   # CRR tree, European & American
│   ├── monte_carlo.py                # MC engine, exotics, variance reduction
│   ├── implied_vol.py                # implied vol inversion (Newton / Brent)
│   └── forward_surface.py            # implied vol surface in forward space
├── notebooks/                        # one pedagogical notebook per phase
│   ├── 00_fondations.ipynb           # GBM, Itô, risk-neutral measure
│   ├── 01_black_scholes.ipynb        # BS + Greeks + MC validation
│   ├── 02_binomial.ipynb             # trees, convergence, American options
│   ├── 03_monte_carlo.ipynb          # exotics + variance reduction
│   ├── 04_volatility_surface.ipynb  # implied vol + 3D surface (AAPL)
│   └── 05_validation.ipynb           # QuantLib cross-validation
├── tests/                            # pytest unit & property tests
├── tools/                            # figure & benchmark scripts feeding the report
├── data/                             # AAPL option chain (kept for reproducibility)
└── images/                           # exported figures
```

## Installation

```bash
# From the project root:
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Requires Python 3.11+ (developed on 3.13). QuantLib and yfinance are only needed for the validation notebook and the live surface; everything else runs without them.

## Usage

Run the notebooks in order (recommended):

```bash
jupyter notebook
```

Then open `notebooks/00_fondations.ipynb` and run all cells.

Or use the engine directly:

```python
from src.black_scholes import BlackScholes
from src.binomial import BinomialTree
from src.monte_carlo import mc_asian

# European option + Greeks
opt = BlackScholes(S=100, K=100, r=0.05, sigma=0.20, T=1.0)
print(opt.price("call"), opt.delta("call"), opt.vega())

# American put (early exercise via CRR tree)
print(
    BinomialTree(
        100, 100, 0.05, 0.20, 1.0, n_steps=1000
    ).price("put", "american")
)

# Arithmetic Asian option (Monte Carlo)
print(
    mc_asian(
        100, 100, 0.05, 0.20, 1.0,
        average="arithmetic"
    )
)
```

## Run the Tests

```bash
pytest -v
```

44 tests; QuantLib tests are skipped if QuantLib isn't installed.

## Run the Benchmark

```bash
python tools/benchmark.py
```

## Volatility Surface

The volatility-surface notebook (04) fetches a live chain via `yfinance`. Offline, it falls back to the bundled AAPL chain in `data/`, so it always runs.

## References

* J. Hull — *Options, Futures and Other Derivatives* (11th ed.)
* P. Glasserman — *Monte Carlo Methods in Financial Engineering* (Springer, 2004)
* A. B. Owen — *Monte Carlo Theory, Methods and Examples* (2013)
* J. Gatheral & A. Jacquier — *Arbitrage-free SVI volatility surfaces* (2012)
* M. Broadie, P. Glasserman & S. Kou — *A Continuity Correction for Discrete Barrier Options* (1997)
