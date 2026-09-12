"""Figures de diagnostic du Projet 1.

Produit deux figures :
  1) convergence_methodes.png  -- erreur type (SE) vs nombre de trajectoires M,
     en log-log, pour les 4 variantes de reduction de variance, sur DEUX
     produits cote a cote : call europeen et asiatique arithmetique. C'est la
     version "deux panneaux" (celle de ton image 1). Le rapport, lui, utilise
     la version un seul panneau (07_mc_reduction_variance.png).
  2) smiles_bruts_aapl.png     -- les smiles bruts de la chaine AAPL (aucune
     interpolation), plus l'illustration de l'illusion de platitude par l'axe Z.
     C'est ton image 2 ; elle sert au diagnostic "la surface est-elle plate ?".

Usage : python tools/figures_diagnostic.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from src.monte_carlo import mc_european, mc_asian

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(RACINE, "images")
CSV = os.path.join(RACINE, "data", "chaine_AAPL_2026-07-27.csv")

VARIANTES = [
    ("Monte-Carlo seul", "#1f77b4", "o", dict(antithetic=False, control_variate=False)),
    ("+ antithetique",   "#ff7f0e", "s", dict(antithetic=True,  control_variate=False)),
    ("+ variable de controle", "#2ca02c", "^", dict(antithetic=False, control_variate=True)),
    ("+ les deux",       "#d62728", "D", dict(antithetic=True,  control_variate=True)),
]


# ===========================================================================
# FIGURE 1 : convergence des methodes, deux produits cote a cote
# ===========================================================================
def figure_convergence():
    S, K, r, sig, T = 100.0, 100.0, 0.05, 0.20, 1.0
    Ms       = np.array([10_000, 100_000, 1_000_000, 10_000_000])
    Ms_asian = np.array([10_000, 100_000, 1_000_000])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # -- panneau gauche : call europeen --
    ax = axes[0]
    for lib, col, mk, kw in VARIANTES:
        ses = [mc_european(S, K, r, sig, T, int(M), rng=np.random.default_rng(7),
                           **kw).std_error for M in Ms]
        ax.loglog(Ms, ses, mk + "-", color=col, ms=6, label=lib)
    ref = ses[0] * np.sqrt(Ms[0]) / np.sqrt(Ms)      # pente -1/2 calee sur le dernier
    ax.loglog(Ms, ref, "k:", lw=1.2)
    ax.set_title("Call europeen")
    ax.set_xlabel("nombre de trajectoires $M$"); ax.set_ylabel("erreur type")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    # -- panneau droit : asiatique arithmetique --
    ax = axes[1]
    for lib, col, mk, kw in VARIANTES:
        ses = [mc_asian(S, K, r, sig, T, n_steps=252, n_paths=int(M),
                        rng=np.random.default_rng(7), **kw).std_error for M in Ms_asian]
        ax.loglog(Ms_asian, ses, mk + "-", color=col, ms=6, label=lib)
    ref = ses[0] * np.sqrt(Ms_asian[0]) / np.sqrt(Ms_asian)
    ax.loglog(Ms_asian, ref, "k:", lw=1.2)
    ax.set_title("Asiatique arithmetique (n=252)")
    ax.set_xlabel("nombre de trajectoires $M$"); ax.set_ylabel("erreur type")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    out = os.path.join(IMG, "05_convergence_methodes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("ecrit :", os.path.relpath(out, RACINE))


# ===========================================================================
# FIGURE 2 : smiles bruts + illusion de platitude (axe Z)
# ===========================================================================
def figure_smiles_bruts():
    df = pd.read_csv(CSV)
    spot = df["spot"].iloc[0]
    mats = sorted(df["T"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    NAVY, RED = "#143C78", "#AA2828"

    # -- gauche : smiles bruts par maturite --
    ax = axes[0]
    for T, c in zip(mats, cm.viridis(np.linspace(0, 0.9, len(mats)))):
        g = df[df["T"] == T].sort_values("K")
        ax.plot(g["K"] / spot, g["iv"] * 100, "-o", ms=3, lw=1.2, color=c,
                label=f"{round(T*365)} j")
    ax.axvline(1.0, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("moneyness  K / S"); ax.set_ylabel("vol implicite (%)")
    ax.set_title("Smiles bruts AAPL (aucune interpolation)", fontsize=10)
    ax.legend(fontsize=6, ncol=2, title="maturite"); ax.grid(alpha=0.25)

    # -- droite : le meme smile sur un axe 0-100 % --
    ax = axes[1]
    g = df[df["T"] == mats[3]].sort_values("K")
    ax.plot(g["K"] / spot, g["iv"] * 100, "-o", ms=3, color=NAVY)
    ax.set_ylim(0, 100)
    ax.set_xlabel("moneyness  K / S"); ax.set_ylabel("vol implicite (%)")
    ax.set_title("Le meme smile sur un axe 0-100 %\n(l'illusion de platitude)",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.text(0.5, 0.5, "ecrase", transform=ax.transAxes, color=RED,
            fontsize=9, ha="center")

    fig.tight_layout()
    out = os.path.join(IMG, "smiles_bruts_aapl.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("ecrit :", os.path.relpath(out, RACINE))


# ===========================================================================
# FIGURE 3 : validation leave-one-out (predite vs observee + erreur par maturite)
# ===========================================================================
def figure_leave_one_out():
    df = pd.read_csv(CSV)
    obs, pred, Tj = [], [], []
    for T, g in df.groupby("T"):
        g = g.sort_values("K").reset_index(drop=True)
        if len(g) < 5:
            continue
        for i in range(1, len(g) - 1):
            reste = g.drop(index=i)
            p = np.interp(g.loc[i, "K"], reste["K"], reste["iv"])
            obs.append(g.loc[i, "iv"] * 100); pred.append(p * 100)
            Tj.append(round(T * 365))
    obs, pred, Tj = np.array(obs), np.array(pred), np.array(Tj)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    NAVY = "#143C78"
    umat = sorted(set(Tj)); cmap = cm.viridis(np.linspace(0, 0.9, len(umat)))
    # gauche : predite vs observee
    ax = axes[0]
    for T, c in zip(umat, cmap):
        m = Tj == T
        ax.scatter(obs[m], pred[m], s=14, color=c, label=f"{T} j", alpha=0.8)
    ax.plot([25, 55], [25, 55], "k--", lw=0.8)
    ax.set_xlim(25, 55); ax.set_ylim(25, 55)
    ax.set_xlabel("vol observee (%)"); ax.set_ylabel("vol predite hors-echantillon (%)")
    ax.set_title("Reconstruction leave-one-out : predite vs observee", fontsize=10)
    ax.legend(fontsize=6, ncol=2, title="maturite"); ax.grid(alpha=0.25)
    # droite : erreur par maturite
    ax = axes[1]
    err = np.abs(pred - obs)
    data = [err[Tj == T] for T in umat]
    bp = ax.boxplot(data, tick_labels=[str(T) for T in umat], patch_artist=True)
    for patch, c in zip(bp["boxes"], cmap):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xlabel("maturite (jours)"); ax.set_ylabel("erreur absolue (pts de vol)")
    ax.set_title("Erreur de reconstruction par maturite", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = os.path.join(IMG, "leave_one_out.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("ecrit :", os.path.relpath(out, RACINE))


if __name__ == "__main__":
    figure_convergence()
    figure_smiles_bruts()
    figure_leave_one_out()
