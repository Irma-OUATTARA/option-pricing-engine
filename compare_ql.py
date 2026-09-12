"""
Comparaison complete du moteur avec QuantLib.

Deux blocs :
  A. PRODUITS A REFERENCE DETERMINISTE  -> ecart direct nous / QuantLib
     (Black-Scholes + Greeks, arbre CRR europeen et americain + Greeks arbre,
      asiatique geometrique par formule fermee).
  B. PRODUITS SANS FORMULE FERMEE       -> ecart absolu nous / QuantLib
     Notre prix vient de notre Monte-Carlo (barrieres a 252 fixings avec
     correction Broadie-Glasserman-Kou) ; la reference QuantLib est analytique
     (barriere continue) ou Monte-Carlo (asiatique). L'ecart absolu est affiche
     et commente : il combine le bruit Monte-Carlo residuel et, pour les
     barrieres, la difference fixings discrets / barriere continue.

Usage : python compare_ql.py
Necessite : pip install QuantLib
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import QuantLib as ql

from src.black_scholes import BlackScholes
from src.binomial import BinomialTree
from src.monte_carlo import (mc_european, mc_asian, mc_barrier,
                             geometric_asian_closed_form)

S, K, r, sigma, q, T = 100.0, 100.0, 0.05, 0.20, 0.0, 1.0
B, NFIX = 120.0, 252
GRAINE = 12345
M = 200_000

today = ql.Date(15, 6, 2024)
ql.Settings.instance().evaluationDate = today
day = ql.Actual365Fixed()
cal = ql.NullCalendar()
spot = ql.QuoteHandle(ql.SimpleQuote(S))
rTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day))
qTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day))
volTS = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, sigma, day))
proc = ql.BlackScholesMertonProcess(spot, qTS, rTS, volTS)
mat = today + int(round(T * 365))


def L(nom, ours, theirs):
    print(f"{nom:<34} nous={ours:>13.8f}  QuantLib={theirs:>13.8f}  "
          f"ecart={abs(ours - theirs):.2e}")


def Lb(nom, prix, ic, theirs):
    print(f"{nom:<26} nous={prix:>10.5f} (IC +/-{ic:.5f})  "
          f"QuantLib={theirs:>10.5f}  ecart abs={abs(prix - theirs):.5f}")


print("=" * 92)
print("A. PRODUITS A REFERENCE DETERMINISTE  (ecart direct)")
print("=" * 92)

# --- Black-Scholes : prix + 3 Greeks ---
c = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K),
                     ql.EuropeanExercise(mat))
c.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
p = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K),
                     ql.EuropeanExercise(mat))
p.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
bs = BlackScholes(S, K, r, sigma, T)
L("BS call (prix)",  bs.price("call"), c.NPV())
L("BS put  (prix)",  bs.price("put"),  p.NPV())
L("BS call delta",   bs.delta("call"), c.delta())
L("BS call gamma",   bs.gamma(),       c.gamma())
L("BS call vega (par pt de vol)", bs.vega() / 100.0, c.vega() / 100.0)

# --- Arbre CRR : europeen (call/put) et americain (put) ---
ce = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K),
                      ql.EuropeanExercise(mat))
ce.setPricingEngine(ql.BinomialVanillaEngine(proc, "crr", 2000))
pe = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K),
                      ql.EuropeanExercise(mat))
pe.setPricingEngine(ql.BinomialVanillaEngine(proc, "crr", 2000))
pa = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K),
                      ql.AmericanExercise(today, mat))
pa.setPricingEngine(ql.BinomialVanillaEngine(proc, "crr", 2000))
L("Arbre CRR 2000 call europ.", BinomialTree(S, K, r, sigma, T, 2000).price("call", "european"), ce.NPV())
L("Arbre CRR 2000 put europ.",  BinomialTree(S, K, r, sigma, T, 2000).price("put", "european"),  pe.NPV())
L("Arbre CRR 2000 put americ.", BinomialTree(S, K, r, sigma, T, 2000).price("put", "american"),  pa.NPV())

# --- Greeks de l'arbre (americain : QuantLib les fournit aussi) ---
ga = BinomialTree(S, K, r, sigma, T, 2000).greeks("put", "american")
L("Arbre put amer. delta", ga["delta"], pa.delta())
L("Arbre put amer. gamma", ga["gamma"], pa.gamma())

# --- Asiatique geometrique : formule fermee des deux cotes ---
dates = [today + int(round(i * T * 365 / NFIX)) for i in range(1, NFIX + 1)]
geo_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
geo = ql.DiscreteAveragingAsianOption(
    ql.Average().Geometric, 0.0, 0, dates, geo_payoff,
    ql.EuropeanExercise(mat))
geo.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(proc))
L("Asiat. geometrique (formule)",
  geometric_asian_closed_form(S, K, r, sigma, T, NFIX), geo.NPV())

print()
print("=" * 92)
print("B. PRODUITS SANS FORMULE FERMEE  (ecart absolu nous / QuantLib)")
print("=" * 92)

# Asiatique arithmetique : notre MC (avec controle) vs MC QuantLib
asi = ql.DiscreteAveragingAsianOption(
    ql.Average().Arithmetic, 0.0, 0, dates, geo_payoff,
    ql.EuropeanExercise(mat))
asi.setPricingEngine(
    ql.MCDiscreteArithmeticAPEngine(proc, "pseudorandom",
                                    requiredSamples=500_000, seed=42))
res = mc_asian(S, K, r, sigma, T, NFIX, M,
               rng=np.random.default_rng(GRAINE), control_variate=True)
Lb("Asiat. arithmetique", res.price, res.ci95, asi.NPV())

# Barrieres : notre MC a 252 fixings + correction BGK vs QuantLib analytique
for bt, ql_type, lib in [("up-and-out", ql.Barrier.UpOut, "up-and-out"),
                         ("up-and-in", ql.Barrier.UpIn, "up-and-in")]:
    bar = ql.BarrierOption(ql_type, B, 0.0,
                           ql.PlainVanillaPayoff(ql.Option.Call, K),
                           ql.EuropeanExercise(mat))
    bar.setPricingEngine(ql.AnalyticBarrierEngine(proc))
    ctrl = (bt == "up-and-in")
    res = mc_barrier(S, K, B, r, sigma, T, NFIX, M,
                     rng=np.random.default_rng(GRAINE), barrier_type=bt,
                     control_variate=ctrl, continuity_correction=True)
    Lb(f"Barriere {lib}", res.price, res.ci95, bar.NPV())

# MC europeen vs BS exact (controle croise)
res = mc_european(S, K, r, sigma, T, M, rng=np.random.default_rng(GRAINE))
Lb("MC europeen (vs BS exact)", res.price, res.ci95, bs.price("call"))

print()
print("Note : les barrieres QuantLib sont a barriere CONTINUE ; notre MC est a")
print("252 fixings avec correction Broadie-Glasserman-Kou. Un ecart residuel")
print("est donc attendu et ne traduit pas une erreur, mais une convention.")
