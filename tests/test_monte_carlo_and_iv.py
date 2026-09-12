"""Tests unitaires des modules Monte Carlo et volatilité implicite."""
import numpy as np
import pytest

from src.black_scholes import BlackScholes
from src.monte_carlo import (mc_european, mc_asian, mc_barrier,
                             geometric_asian_closed_form)
from src.implied_vol import implied_vol


# --- Monte Carlo ---------------------------------------------------------
def test_mc_european_within_ci_of_black_scholes():
    bs = BlackScholes(100, 100, 0.05, 0.20, 1.0).price("call")
    rng = np.random.default_rng(0)
    res = mc_european(100, 100, 0.05, 0.20, 1.0, n_paths=200_000, rng=rng)
    # Le prix BS doit tomber dans l'intervalle de confiance à 95 %
    assert abs(res.price - bs) < res.ci95 * 1.5


def test_variance_reduction_reduces_error():
    rng = np.random.default_rng(1)
    base = mc_european(100, 100, 0.05, 0.20, 1.0, 100_000, rng=rng)
    rng = np.random.default_rng(1)
    cv = mc_european(100, 100, 0.05, 0.20, 1.0, 100_000, rng=rng,
                     control_variate=True)
    # La variable de controle doit reduire SE
    assert cv.std_error < base.std_error


def test_asian_geometric_matches_closed_form():
    rng = np.random.default_rng(2)
    mc = mc_asian(100, 100, 0.05, 0.20, 1.0, n_steps=50, n_paths=200_000,
                  average="geometric", rng=rng, antithetic=True)
    cf = geometric_asian_closed_form(100, 100, 0.05, 0.20, 1.0, 50)
    assert abs(mc.price - cf) < mc.ci95 * 2


def test_asian_cheaper_than_european():
    bs = BlackScholes(100, 100, 0.05, 0.20, 1.0).price("call")
    rng = np.random.default_rng(3)
    asian = mc_asian(100, 100, 0.05, 0.20, 1.0, n_steps=50, n_paths=100_000,
                     average="arithmetic", rng=rng).price
    assert asian < bs


def test_barrier_in_out_parity():
    bs = BlackScholes(100, 100, 0.05, 0.20, 1.0).price("call")
    rng = np.random.default_rng(4)
    out = mc_barrier(100, 100, 120, 0.05, 0.20, 1.0, n_steps=100,
                     n_paths=200_000, barrier_type="up-and-out", rng=rng).price
    rng = np.random.default_rng(4)
    inn = mc_barrier(100, 100, 120, 0.05, 0.20, 1.0, n_steps=100,
                     n_paths=200_000, barrier_type="up-and-in", rng=rng).price
    # in + out = vanille (à l'erreur de discrétisation près)
    assert (out + inn) == pytest.approx(bs, abs=0.1)


# --- Volatilité implicite ------------------------------------------------
@pytest.mark.parametrize("true_sigma", [0.10, 0.20, 0.35, 0.50])
def test_implied_vol_roundtrip(true_sigma):
    # sigma -> prix -> on doit retrouver sigma
    price = BlackScholes(100, 100, 0.05, true_sigma, 1.0).price("call")
    iv = implied_vol(price, 100, 100, 0.05, 1.0, "call")
    assert iv == pytest.approx(true_sigma, abs=1e-6)


def test_implied_vol_out_of_bounds_returns_nan():
    assert np.isnan(implied_vol(0.0, 100, 100, 0.05, 1.0, "call"))
    assert np.isnan(implied_vol(200, 100, 100, 0.05, 1.0, "call"))
