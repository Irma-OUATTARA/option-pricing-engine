"""Validation du moteur contre QuantLib, la référence du secteur.

Ces tests se sautent automatiquement si QuantLib n'est pas installé
(pip install QuantLib).
"""
import pytest

ql = pytest.importorskip("QuantLib")

from src.black_scholes import BlackScholes
from src.binomial import BinomialTree


def _ql_process(S, r, sigma, q, today):
    day = ql.Actual365Fixed()
    cal = ql.NullCalendar()
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day))
    qTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day))
    volTS = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, sigma, day))
    return ql.BlackScholesMertonProcess(spot, qTS, rTS, volTS)


@pytest.fixture
def setup():
    today = ql.Date(15, 6, 2024)
    ql.Settings.instance().evaluationDate = today
    S, K, r, sigma, q, T = 100.0, 100.0, 0.05, 0.20, 0.0, 1.0
    process = _ql_process(S, r, sigma, q, today)
    maturity = today + int(T * 365)
    return dict(S=S, K=K, r=r, sigma=sigma, q=q, T=T,
                process=process, maturity=maturity)


def test_european_call_matches_quantlib(setup):
    s = setup
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, s["K"])
    opt = ql.VanillaOption(payoff, ql.EuropeanExercise(s["maturity"]))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(s["process"]))
    ours = BlackScholes(s["S"], s["K"], s["r"], s["sigma"], s["T"]).price("call")
    assert ours == pytest.approx(opt.NPV(), abs=1e-3)


def test_european_greeks_match_quantlib(setup):
    s = setup
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, s["K"])
    opt = ql.VanillaOption(payoff, ql.EuropeanExercise(s["maturity"]))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(s["process"]))
    ours = BlackScholes(s["S"], s["K"], s["r"], s["sigma"], s["T"])
    assert ours.delta("call") == pytest.approx(opt.delta(), abs=1e-3)
    assert ours.gamma() == pytest.approx(opt.gamma(), abs=1e-3)


def test_american_put_matches_quantlib(setup):
    s = setup
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, s["K"])
    opt = ql.VanillaOption(payoff, ql.AmericanExercise(ql.Date(15, 6, 2024), s["maturity"]))
    opt.setPricingEngine(ql.BinomialVanillaEngine(s["process"], "crr", 1000))
    ours = BinomialTree(s["S"], s["K"], s["r"], s["sigma"], s["T"], 1000).price("put", "american")
    assert ours == pytest.approx(opt.NPV(), abs=1e-2)
