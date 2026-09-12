"""Génère les trois éléments ajoutés au chapitre surface :
  - images/04_pipeline.png  : schéma conceptuel marché -> surface (sans données) ;
  - images/04_filtrage.png  : barres avant/après filtrage ;
  - le TABLEAU des caractéristiques du smile (imprimé), à recopier dans le rapport
    (tab:smilecar) : vol ATM, pente du skew, nb de points, par maturité.

Le nombre d'options brutes (avant filtrage) n'est PAS dans le CSV exporté : il
vit dans le dict `stats` du carnet 04. Passe-le en argument si tu veux la valeur
exacte du jour ; sinon on utilise 608 (chiffre documenté au 27/07).

Usage : python tools/figures_surface_extra.py [n_brut]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(RACINE, "images")
CSV = os.path.join(RACINE, "data", "chaine_AAPL_2026-07-27.csv")
NAVY = "#143C78"
R, Q = 0.04, 0.004


def pipeline():
    etapes = ["Chaine d options AAPL\n(yfinance)", "Prix bid / ask",
              "Filtrage des cotations", "Prix mid representatif",
              "Forward lu (parite call-put)", "Inversion Black-Scholes (Brent)",
              "Volatilites implicites", "Smile par maturite",
              "Interpolation (variance totale)", "Surface de volatilite"]
    fig, ax = plt.subplots(figsize=(4.6, 8)); ax.axis("off")
    n = len(etapes); h = 1.0 / n
    for i, txt in enumerate(etapes):
        y = 1 - (i + 0.5) * h
        ax.add_patch(FancyBboxPatch((0.12, y - h*0.32), 0.76, h*0.62,
                     boxstyle="round,pad=0.01", fc="#eef2f8", ec=NAVY, lw=1.3,
                     transform=ax.transAxes))
        ax.text(0.5, y, txt, ha="center", va="center", fontsize=8.5,
                transform=ax.transAxes)
        if i < n - 1:
            ax.annotate("", xy=(0.5, y - h*0.34), xytext=(0.5, y - h*0.66),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.4))
    fig.savefig(os.path.join(IMG, "04_pipeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("ecrit : images/04_pipeline.png")


def filtrage(n_brut):
    n_final = len(pd.read_csv(CSV))
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.barh(["Apres filtrage", "Chaine brute"], [n_final, n_brut],
            color=[NAVY, "#c8d4e6"])
    for i, v in enumerate([n_final, n_brut]):
        ax.text(v + 8, i, str(v), va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("nombre d options"); ax.set_xlim(0, n_brut * 1.12)
    ax.set_title("Ampleur du nettoyage de la chaine AAPL", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "04_filtrage.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"ecrit : images/04_filtrage.png  (brut={n_brut}, final={n_final})")


def tableau_smile():
    df = pd.read_csv(CSV); S = df["spot"].iloc[0]
    df["F"] = S * np.exp((R - Q) * df["T"]); df["k"] = np.log(df["K"] / df["F"])
    print("\nTABLEAU DU SMILE (tab:smilecar) -- a recopier dans le rapport :")
    print(f"  {'maturite':>9}{'vol ATM':>9}{'pente skew':>12}{'nb pts':>8}")
    for T, g in df.groupby("T"):
        coef = np.polyfit(g["k"].values, g["iv"].values, 2)
        print(f"  {round(T*365):>6} j{np.polyval(coef,0)*100:>8.1f}%"
              f"{coef[1]*100:>+12.1f}{len(g):>8}")


if __name__ == "__main__":
    n_brut = int(sys.argv[1]) if len(sys.argv) > 1 else 608
    pipeline()
    filtrage(n_brut)
    tableau_smile()
