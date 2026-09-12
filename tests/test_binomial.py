"""Tests unitaires du module arbre binomial (CRR)."""
import pytest

from src.binomial import BinomialTree
from src.black_scholes import BlackScholes


def test_european_converges_to_black_scholes():
    # Avec beaucoup de pas, l'arbre européen doit égaler Black-Scholes
    bs = BlackScholes(100, 100, 0.05, 0.20, 1.0).price("call")
    tree = BinomialTree(100, 100, 0.05, 0.20, 1.0, n_steps=2000).price("call", "european")
    assert tree == pytest.approx(bs, abs=1e-2)


def test_european_put_converges():
    bs = BlackScholes(100, 100, 0.05, 0.20, 1.0).price("put")
    tree = BinomialTree(100, 100, 0.05, 0.20, 1.0, n_steps=2000).price("put", "european")
    assert tree == pytest.approx(bs, abs=1e-2)


def test_american_put_premium_positive():
    # Le put américain vaut plus que l'européen (prime d'exercice anticipé)
    t = BinomialTree(100, 100, 0.05, 0.20, 1.0, n_steps=500)
    assert t.price("put", "american") > t.price("put", "european")


def test_american_call_equals_european_without_dividend():
    # Sans dividende : call américain = call européen
    t = BinomialTree(100, 100, 0.05, 0.20, 1.0, n_steps=500)
    assert t.price("call", "american") == pytest.approx(
        t.price("call", "european"), abs=1e-6)


def test_american_call_premium_with_dividend():
    # Avec dividende, le call américain reprend de la valeur
    t = BinomialTree(100, 100, 0.05, 0.20, 1.0, n_steps=500, q=0.06)
    assert t.price("call", "american") > t.price("call", "european")


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        BinomialTree(100, 100, 0.05, 0.20, 1.0, n_steps=0)
    with pytest.raises(ValueError):
        BinomialTree(100, 100, 0.05, 0.20, 1.0, 100).price("call", "bermudan")
