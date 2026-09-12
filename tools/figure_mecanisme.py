"""Figure : pourquoi n depend du produit (1 tirage contre 252)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.simulation import simulate_gbm_paths, simulate_gbm_terminal

S0, r, sig, T = 100.0, 0.05, 0.20, 1.0
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
rng = np.random.default_rng(4)
K = 100.0

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

# ---- gauche : europeenne, n = 1 ----
ST = simulate_gbm_terminal(S0, r, sig, T, 6, rng=np.random.default_rng(4))
for s in ST:
    a1.plot([0, T], [S0, s], "--", color="#8fa8c8", lw=1.1)
    a1.plot(T, s, "o", color="#1f4e9c", ms=6)
a1.plot(0, S0, "o", color="black", ms=6)
a1.axhline(K, color="#c0392b", lw=1, ls=":", label="strike $K$")
a1.set_title("Call européen : $n = 1$", fontsize=12)
a1.set_xlabel("temps (années)")
a1.set_ylabel("prix du sous-jacent")
a1.text(0.5, 0.04, "1 tirage gaussien par trajectoire\nseul $S_T$ compte",
        transform=a1.transAxes, ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#eef3fa", ec="#1f4e9c", lw=.8))
a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=.25, ls=":")

# ---- droite : asiatique, n = 252 ----
n = 252
paths = simulate_gbm_paths(S0, r, sig, T, n, 6, rng=np.random.default_rng(4),
                           include_spot=True)
t = np.linspace(0, T, n + 1)
for p in paths:
    a2.plot(t, p, "-", color="#8fa8c8", lw=.9)
    a2.plot(t[-1], p[-1], "o", color="#1f4e9c", ms=5)
    a2.axhline(p[1:].mean(), color="#2e8b57", lw=.8, alpha=.45)
a2.plot(0, S0, "o", color="black", ms=6)
a2.axhline(K, color="#c0392b", lw=1, ls=":", label="strike $K$")
a2.plot([], [], color="#2e8b57", lw=1, label=r"moyenne $\bar S$ de chaque trajectoire")
a2.set_title("Asiatique : $n = 252$", fontsize=12)
a2.set_xlabel("temps (années)")
a2.text(0.5, 0.04, "252 tirages gaussiens par trajectoire\ntoute la trajectoire compte",
        transform=a2.transAxes, ha="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#eef7f0", ec="#2e8b57", lw=.8))
a2.legend(fontsize=8, loc="upper left"); a2.grid(alpha=.25, ls=":")

fig.suptitle("Le coût d'une trajectoire dépend de ce que le payoff exige",
             fontsize=13)
fig.tight_layout()
fig.savefig(f"{IMG}/07_mecanisme_n.png", dpi=150)
print("figure ecrite : images/07_mecanisme_n.png")
