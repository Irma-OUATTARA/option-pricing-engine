"""Régénère les trois figures du chapitre surface À PARTIR DU CSV FIGÉ,
sans aucun appel réseau (pas de yfinance). Objectif : rendre le rapport
reproductible sans jamais relancer le carnet 04.

  - images/04_smile.png          : smile d'une maturité MOYENNE (le bord court
                                   est bruité) en ln(K/F) ;
  - images/04_term_structure.png : vol ATM en fonction de la maturité ;
  - images/04_vol_surface.png    : surface 3D en ln(K/F), axe Z borné [15,55].

Usage : python tools/figures_surface_main.py
Le carnet 04 ne sert plus qu'à créer un NOUVEAU snapshot (data/chaine_AAPL_2026-07-27.csv).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(RACINE, "images")
CSV = os.path.join(RACINE, "data", "chaine_AAPL_2026-07-27.csv")
R, Q = 0.04, 0.004
NAVY = "#143C78"


def charger():
    df = pd.read_csv(CSV)
    S = df["spot"].iloc[0]
    df["F"] = S * np.exp((R - Q) * df["T"])
    df["k"] = np.log(df["K"] / df["F"])
    return df, S


def smile(df):
    # multi-courbes : un smile par maturite, en ln(K/F) (comme dans le rapport)
    from matplotlib import cm
    mats = np.sort(df["T"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for T, c in zip(mats, cm.viridis(np.linspace(0, 0.9, len(mats)))):
        g = df[df["T"] == T].sort_values("k")
        ax.plot(g["k"], g["iv"] * 100, "-o", ms=3, lw=1.1, color=c,
                label=f"{round(T*365)} j")
    ax.axvline(0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("log-moneyness-forward  k = ln(K/F)")
    ax.set_ylabel("volatilite implicite (%)")
    #ax.set_title("Smiles AAPL en ln(K/F)  (une courbe par maturite)", fontsize=10)
    ax.legend(fontsize=6, ncol=2, title="maturite"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "04_smile.png"), dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print("ecrit : images/04_smile.png  (multi-maturites)")


def term_structure(df, S):
    pts = []
    for T, g in df.groupby("T"):
        i = (g["K"] - S).abs().idxmin()
        pts.append((T, g.loc[i, "iv"] * 100))
    pts.sort()
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, y, "o-", color="orange", ms=6)
    ax.set_xlabel("maturite (annees)"); ax.set_ylabel("volatilite implicite ATM (%)")
    ax.set_title("Structure par terme de la volatilite (a la monnaie)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "04_term_structure.png"),
                                    dpi=150, bbox_inches="tight"); plt.close(fig)
    print("ecrit : images/04_term_structure.png")


def surface(df):
    mats = np.sort(df["T"].unique())
    K_LO, K_HI = -0.13, 0.13
    kg = np.linspace(K_LO, K_HI, 60)
    W = np.zeros((len(mats), len(kg)))
    for i, T in enumerate(mats):
        g = df[df["T"] == T]
        coef = np.polyfit(g["k"].values, g["iv"].values, 2)
        W[i] = (np.polyval(coef, kg) ** 2) * T
    Tg = np.linspace(mats.min(), mats.max(), 60)
    Wg = np.zeros((len(Tg), len(kg)))
    for j in range(len(kg)):
        Wg[:, j] = interp1d(mats, W[:, j], kind="linear")(Tg)
    Xi, Yi = np.meshgrid(kg, Tg); Zi = np.sqrt(Wg / Yi) * 100
    pts = df[(df["k"] >= K_LO) & (df["k"] <= K_HI)]
    fig = plt.figure(figsize=(9, 6.4)); ax = fig.add_subplot(111, projection="3d")
    s = ax.plot_surface(Xi, Yi, Zi, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.scatter(pts["k"], pts["T"], pts["iv"] * 100, color="crimson", s=10, alpha=0.45)
    ax.set_xlabel("k = ln(K/F)"); ax.set_ylabel("maturite (annees)")
    ax.set_zlabel("vol implicite (%)"); ax.set_zlim(15, 55)
    #ax.set_title("Surface de volatilite implicite AAPL (ln(K/F))", fontsize=12)
    ax.view_init(elev=22, azim=-125)
    fig.colorbar(s, shrink=0.5, aspect=12, label="vol implicite (%)")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "04_vol_surface.png"),
                                    dpi=150, bbox_inches="tight"); plt.close(fig)
    print("ecrit : images/04_vol_surface.png")


if __name__ == "__main__":
    df, S = charger()
    smile(df); term_structure(df, S); surface(df)
    print("Les trois figures du chapitre surface sont regenerees depuis le CSV fige.")
