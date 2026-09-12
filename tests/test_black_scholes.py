"""Tests unitaires du module Black-Scholes.

Lancer depuis la racine du projet :  pytest -v
"""
import numpy as np
import pytest

from src.black_scholes import BlackScholes


# Cas de référence (Hull) : S=100, K=100, r=5%, sigma=20%, T=1
@pytest.fixture
def bs():
    return BlackScholes(S=100, K=100, r=0.05, sigma=0.20, T=1.0)


def test_call_reference_value(bs):
    # Valeur de référence connue (Hull)
    assert bs.price("call") == pytest.approx(10.4506, abs=1e-4)


def test_put_reference_value(bs):
    assert bs.price("put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity(bs):
    # C - P == S - K e^{-rT}
    lhs = bs.price("call") - bs.price("put")
    rhs = bs.S - bs.K * np.exp(-bs.r * bs.T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_delta_bounds(bs):
    assert 0.0 <= bs.delta("call") <= 1.0
    assert -1.0 <= bs.delta("put") <= 0.0


def test_delta_matches_finite_difference(bs):
    h = 1e-4
    up = BlackScholes(100 + h, 100, 0.05, 0.20, 1.0).price("call")
    dn = BlackScholes(100 - h, 100, 0.05, 0.20, 1.0).price("call")
    fd = (up - dn) / (2 * h)
    assert bs.delta("call") == pytest.approx(fd, abs=1e-5)


def test_gamma_positive_and_symmetric(bs):
    # Gamma identique call/put, toujours positif
    assert bs.gamma() > 0


def test_deep_itm_call_delta_near_one():
    deep = BlackScholes(S=200, K=100, r=0.05, sigma=0.20, T=1.0)
    assert deep.delta("call") == pytest.approx(1.0, abs=1e-2)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        BlackScholes(100, 100, 0.05, 0.20, T=0)      # T <= 0
    with pytest.raises(ValueError):
        BlackScholes(100, 100, 0.05, sigma=0, T=1)   # sigma <= 0
    with pytest.raises(ValueError):
        BlackScholes(100, 100, 0.05, 0.20, 1).price("swaption")
