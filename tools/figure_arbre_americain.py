"""Schema pedagogique d'un arbre binomial CRR a 4 pas, put americain.

Montre a chaque noeud : le prix du sous-jacent, la valeur de l'option, et
surtout la DECISION prise (continuer / exercer). Les noeuds ou l'exercice
anticipe est optimal sont mis en evidence : c'est ce que Black-Scholes ne sait
pas faire et que l'arbre capture naturellement.

Sortie : images/06_arbre_americain.png (et .svg)
Produit avec les parametres du rapport, mais a N=4 pour la lisibilite.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.binomial import BinomialTree

S, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.20, 1.0, 4
t = BinomialTree(S, K, r, sigma, T, N)
dt, u, d, p = t.params
disc = np.exp(-r * dt)

# --- reconstruire l'arbre en gardant la trace des decisions ---
Snode, V, exercised = {}, {}, {}
for i in range(N + 1):
    for j in range(i + 1):
        Snode[(i, j)] = S * u**j * d**(i - j)
for j in range(N + 1):
    V[(N, j)] = max(K - Snode[(N, j)], 0.0)
    exercised[(N, j)] = V[(N, j)] > 0
for i in range(N - 1, -1, -1):
    for j in range(i + 1):
        cont = disc * (p * V[(i + 1, j + 1)] + (1 - p) * V[(i + 1, j)])
        intr = max(K - Snode[(i, j)], 0.0)
        V[(i, j)] = max(cont, intr)
        exercised[(i, j)] = intr > cont and intr > 0

fig, ax = plt.subplots(figsize=(11, 7.2))
BLEU = "#1f3a68"
ROUGE = "#b23b3b"
VERT = "#2e7d52"
GRIS = "#8a8a8a"

# position : x = date i, y = niveau j centre
def pos(i, j):
    return i * 2.6, (j - i / 2.0) * 1.6

# --- branches ---
for i in range(N):
    for j in range(i + 1):
        x0, y0 = pos(i, j)
        for (ni, nj) in [(i + 1, j + 1), (i + 1, j)]:
            x1, y1 = pos(ni, nj)
            ax.plot([x0, x1], [y0, y1], color=GRIS, lw=0.8, zorder=1)

# --- noeuds ---
for i in range(N + 1):
    for j in range(i + 1):
        x, y = pos(i, j)
        ex = exercised[(i, j)]
        term = (i == N)
        if ex and not term:
            fc, ec, tc = "#f6e0e0", ROUGE, ROUGE
        elif term:
            fc, ec, tc = "#eef1f6", BLEU, BLEU
        else:
            fc, ec, tc = "#ffffff", BLEU, BLEU
        box = FancyBboxPatch((x - 0.62, y - 0.42), 1.24, 0.84,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             fc=fc, ec=ec, lw=1.3, zorder=2)
        ax.add_patch(box)
        ax.text(x, y + 0.16, f"S={Snode[(i,j)]:.1f}", ha="center",
                va="center", fontsize=8, color=tc, zorder=3)
        ax.text(x, y - 0.17, f"V={V[(i,j)]:.2f}", ha="center",
                va="center", fontsize=8.5, weight="bold", color=tc, zorder=3)

# --- dates ---
for i in range(N + 1):
    x, _ = pos(i, 0)
    ax.text(x, (-N / 2.0 - 0.9) * 1.6, f"$t_{i}$\n{i*dt:.2f} an",
            ha="center", va="top", fontsize=9, color="#333")

# --- legende ---
handles = [
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#ffffff",
               markeredgecolor=BLEU, markersize=12, label="continuer (garder l'option)"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#f6e0e0",
               markeredgecolor=ROUGE, markersize=12, label="exercer maintenant (optimal)"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#eef1f6",
               markeredgecolor=BLEU, markersize=12, label="echeance : payoff final"),
]
ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.95)

ax.text(0.5 * 2.6 * 0 + 5.2, (N / 2.0 + 0.75) * 1.6,
        f"Put americain, K={K:.0f}  —  prix = {V[(0,0)]:.3f}  (N=4 pas)",
        ha="center", fontsize=11, weight="bold", color=BLEU)

ax.set_xlim(-1.2, N * 2.6 + 1.2)
ax.set_ylim((-N / 2.0 - 1.5) * 1.6, (N / 2.0 + 1.2) * 1.6)
ax.axis("off")
plt.tight_layout()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "images")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "06_arbre_americain.png"), dpi=150,
            bbox_inches="tight")
fig.savefig(os.path.join(out, "06_arbre_americain.svg"), bbox_inches="tight")
print("figure ecrite :", V[(0, 0)])
