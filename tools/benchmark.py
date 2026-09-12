"""Mesure des temps d'exécution du moteur. Usage : python tools/benchmark.py

Médiane sur n répétitions après préchauffage. Écrit aussi
tools/benchmark_table.tex, directement insérable dans le rapport.
"""
import sys, os, time, platform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.black_scholes import BlackScholes
from src.binomial import BinomialTree
from src.monte_carlo import (mc_european, mc_asian, mc_barrier,
                             mc_price, payoff_lookback_floating)
from src.implied_vol import implied_vol

S, K, r, sig, T = 100.0, 100.0, 0.05, 0.20, 1.0


def chrono(fn, n=10, w=2):
    for _ in range(w):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def fmt(t):
    if t < 1e-3:
        return f"{t*1e6:.1f} us"
    if t < 1.0:
        return f"{t*1e3:.1f} ms"
    return f"{t:.2f} s"


rng = lambda s=0: np.random.default_rng(s)
chain = [(kk, tt) for kk in np.linspace(70, 130, 20)
                  for tt in np.linspace(0.1, 2.0, 20)]
prices = [BlackScholes(S, kk, r, 0.20 + 0.15 * (100 - kk) / 100, tt).price()
          for kk, tt in chain]

CASES = [
    ("Call europeen",        "Black-Scholes (prix seul)", "--",
     lambda: BlackScholes(S, K, r, sig, T).price(), 500, 50),
    ("Call europeen",        "Black-Scholes (prix + 5 Greeks)", "--",
     lambda: BlackScholes(S, K, r, sig, T).greeks(), 500, 50),
    ("Call europeen",        "Arbre CRR", "N = 1 000",
     lambda: BinomialTree(S, K, r, sig, T, 1000).price(), 5, 1),
    ("Call europeen",        "Arbre CRR", "N = 10 000",
     lambda: BinomialTree(S, K, r, sig, T, 10000).price(), 3, 1),
    ("Put americain",        "Arbre CRR", "N = 1 000",
     lambda: BinomialTree(S, K, r, sig, T, 1000).price("put", "american"), 5, 1),
    ("Call europeen",        "Monte-Carlo", "M = 1e5",
     lambda: mc_european(S, K, r, sig, T, 100_000, rng=rng()), 10, 2),
    ("Call europeen",        "Monte-Carlo + controle", "M = 1e5",
     lambda: mc_european(S, K, r, sig, T, 100_000, rng=rng(),
                         control_variate=True), 10, 2),
    ("Asiatique arithm.",    "Monte-Carlo", "M = 1e5, n = 252",
     lambda: mc_asian(S, K, r, sig, T, 252, 100_000, rng=rng()), 3, 1),
    ("Asiatique arithm.",    "Monte-Carlo + controle geo.", "M = 1e5, n = 252",
     lambda: mc_asian(S, K, r, sig, T, 252, 100_000, rng=rng(),
                      control_variate=True), 3, 1),
    ("Barriere up-and-out",  "Monte-Carlo", "M = 1e5, n = 252",
     lambda: mc_barrier(S, K, 130.0, r, sig, T, 252, 100_000, rng=rng()), 3, 1),
    ("Lookback flottante",   "Monte-Carlo", "M = 1e5, n = 252",
     lambda: mc_price(payoff_lookback_floating("call"), S, r, sig, T,
                      n_steps=252, n_paths=100_000, rng=rng()), 3, 1),
    ("Volatilite implicite", "Newton + repli Brent", "1 inversion",
     lambda: implied_vol(prices[0], S, chain[0][0], r, chain[0][1]), 500, 50),
    ("Surface complete",     "inversion en lot", f"{len(chain)} options",
     lambda: [implied_vol(p, S, kk, r, tt)
              for (kk, tt), p in zip(chain, prices)], 5, 1),
]

print(f"Machine : {platform.machine()} / {platform.system()} / "
      f"Python {platform.python_version()} / NumPy {np.__version__}\n")
rows = []
for prod, meth, par, fn, n, w in CASES:
    t = chrono(fn, n, w)
    print(f"{prod:<22}{meth:<32}{par:<20}{fmt(t):>10}")
    rows.append(f"{prod} & {meth} & {par} & {fmt(t).replace('us','\\\\textmu s')} \\\\\\\\")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_table.tex"), "w") as fh:
    fh.write("\n".join(rows) + "\n")
