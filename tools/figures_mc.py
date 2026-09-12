"""Figures de la section Monte-Carlo, avec la notation du rapport :
   M = nombre de trajectoires, n = nombre de pas de temps par trajectoire."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.black_scholes import BlackScholes
from src.monte_carlo import mc_european

S, K, r, sig, T = 100., 100., 0.05, 0.20, 1.0
BS = BlackScholes(S, K, r, sig, T).price()
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")

Ms = np.unique(np.logspace(2, 6, 26).astype(int))
Ms = Ms[Ms % 2 == 0]

# NOTE : la figure de convergence du prix vers Black-Scholes n'est plus
# produite ici. Elle faisait doublon avec 01_mc_convergence.png, generee par
# notebooks/01_black_scholes.ipynb, qui est la version utilisee par le rapport.

# ---- Erreur type des quatre methodes ----
cfg = [("Monte-Carlo seul", dict(), "#1f4e9c", "o"),
       ("+ antithetique", dict(antithetic=True), "#e67e22", "s"),
       ("+ variable de controle", dict(control_variate=True), "#2e8b57", "^"),
       ("+ les deux", dict(antithetic=True, control_variate=True), "#8e44ad", "d")]
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for lib, kw, c, mk in cfg:
    ses = [mc_european(S, K, r, sig, T, int(M), rng=np.random.default_rng(7), **kw).std_error
           for M in Ms]
    # la pente est MESUREE par regression, jamais imposee : elle est
    # affichee telle quelle dans la legende
    pente = -np.polyfit(np.log(Ms), np.log(ses), 1)[0]
    ax.loglog(Ms, ses, mk + "-", ms=3.5, lw=1.2, color=c,
              label=f"{lib}  (pente mesuree {pente:.3f})")
ax.set_xlabel("nombre de trajectoires  $M$")
ax.set_ylabel("SE de l'estimateur")
ax.set_title("Erreur type selon $M$")
ax.legend(fontsize=8); ax.grid(alpha=.3, ls=":", which="both")
fig.tight_layout(); fig.savefig(f"{IMG}/07_mc_reduction_variance.png", dpi=150)
print("figure ecrite : 07_mc_reduction_variance.png")
