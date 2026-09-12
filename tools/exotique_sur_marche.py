"""AAPL -> volatilite implicite -> cotation d'une exotique.

C'est le seul endroit du projet ou une donnee de marche entre dans le pricer.
Ailleurs, sigma est choisi ; ici il est LU dans les prix cotes.

Chaine de traitement :
  1. charger la chaine d'options nettoyee produite par le carnet 04
  2. construire la structure par terme de la volatilite a la monnaie
  3. interpoler sigma a la maturite de l'exotique (en VARIANCE TOTALE)
  4. coter une asiatique et une barriere sur AAPL avec ce sigma

Usage :
    python tools/exotique_sur_marche.py [chemin_csv]
Le CSV attendu est celui exporte par le carnet 04 : colonnes type, K, T, mid, iv.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from src.black_scholes import BlackScholes
from src.monte_carlo import mc_asian, mc_barrier

CSV_DEFAUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "chaine_AAPL_2026-07-27.csv")


# ---------------------------------------------------------------------------
def charger_chaine(chemin):
    if os.path.exists(chemin):
        df = pd.read_csv(chemin)
        meta = f"donnees reelles : {os.path.basename(chemin)}"
        return df, meta, True
    print("!" * 74)
    print("! AUCUN FICHIER DE CHAINE TROUVE :", chemin)
    print("! Les chiffres ci-dessous sont produits a partir d'une chaine")
    print("! SYNTHETIQUE. Ils ne valent RIEN pour le rapport.")
    print("! Lancer le carnet 04 avec reseau, exporter la chaine, relancer.")
    print("!" * 74)
    return _chaine_factice(), "CHAINE SYNTHETIQUE - NE PAS UTILISER", False


def _chaine_factice(S0=230.0, r=0.045, q=0.004):
    rows = []
    for T in (0.05, 0.12, 0.25, 0.50, 0.75, 1.00):
        atm = 0.26 + 0.02 * np.sqrt(T)
        for K in np.arange(0.80 * S0, 1.201 * S0, 5.0):
            m = np.log(K / S0)
            iv = max(atm - 0.22 * m + 0.85 * m**2, 0.08)
            typ = "call" if K >= S0 else "put"
            rows.append(dict(type=typ, K=K, T=T,
                             mid=BlackScholes(S0, K, r, iv, T, q).price(typ),
                             iv=iv, spot=S0))
    return pd.DataFrame(rows)


def structure_atm(chain, spot):
    """Volatilite a la monnaie par maturite (strike le plus proche du spot)."""
    pts = []
    for T, g in chain.groupby("T"):
        i = (g["K"] - spot).abs().idxmin()
        pts.append((float(T), float(g.loc[i, "iv"]), float(g.loc[i, "K"])))
    return sorted(pts)


def vol_interpolee(pts, T_cible):
    """Interpolation en VARIANCE TOTALE sigma^2 * T, pas en sigma.

    C'est la seule interpolation qui ne cree pas d'arbitrage calendaire : la
    variance totale doit etre croissante en T. Interpoler sigma directement
    peut produire une variance decroissante entre deux maturites, donc une
    variance forward negative - un modele qui n'existe pas.
    """
    Ts = np.array([p[0] for p in pts])
    var_tot = np.array([p[1] ** 2 * p[0] for p in pts])
    v = np.interp(T_cible, Ts, var_tot)
    return float(np.sqrt(v / T_cible))


def vol_au_strike(chain, T_cible, K_cible, spot):
    """Vol implicite lue sur le smile, au strike demande, a la maturite cible."""
    Ts = np.array(sorted(chain["T"].unique()))
    T_proche = Ts[np.argmin(np.abs(Ts - T_cible))]
    g = chain[chain["T"] == T_proche].sort_values("K")
    return float(np.interp(K_cible, g["K"], g["iv"])), float(T_proche)


# ---------------------------------------------------------------------------
def main(chemin):
    chain, meta, reel = charger_chaine(chemin)
    spot = float(chain["spot"].iloc[0]) if "spot" in chain else \
        float(chain.loc[(chain["K"] - chain["K"].median()).abs().idxmin(), "K"])
    r, q = 0.045, 0.004
    T_exo = 0.25                      # exotique 3 mois

    print(f"\nSource   : {meta}")
    print(f"Spot     : {spot:.2f}   |   {len(chain)} options, "
          f"{chain['T'].nunique()} maturites")

    pts = structure_atm(chain, spot)
    print("\nStructure par terme a la monnaie")
    print(f"  {'maturite':>10}{'strike ATM':>12}{'vol impl.':>12}{'var. totale':>13}")
    for T, iv, Kk in pts:
        print(f"  {T*365:>8.0f} j{Kk:>12.1f}{iv:>11.2%}{iv**2*T:>13.5f}")

    sigma = vol_interpolee(pts, T_exo)
    print(f"\n=> vol interpolee a {T_exo*365:.0f} jours : {sigma:.2%}")
    print("   (interpolation en variance totale, sans arbitrage calendaire)")

    K = round(spot / 5) * 5.0
    B = round(spot * 1.15 / 5) * 5.0
    g = lambda s=1: np.random.default_rng(s)

    print("\n" + "=" * 74)
    print(f"COTATION D'EXOTIQUES SUR {'AAPL' if reel else '[SYNTHETIQUE]'}"
          f"   K={K:.0f}, T={T_exo*365:.0f}j, sigma={sigma:.2%}")
    print("=" * 74)

    van = BlackScholes(spot, K, r, sigma, T_exo, q).price()
    print(f"{'Call vanille (reference)':<38}{van:>10.4f}")

    asi = mc_asian(spot, K, r, sigma, T_exo, 63, 200_000, q=q,
                   rng=g(), control_variate=True)
    print(f"{'Asiatique arithmetique (63 fixings)':<38}{asi.price:>10.4f}"
          f"   +/- {asi.ci95:.4f}")
    print(f"{'  decote vs vanille':<38}{asi.price/van-1:>9.1%}")

    bar = mc_barrier(spot, K, B, r, sigma, T_exo, 63, 200_000, q=q, rng=g(2))
    print(f"{f'Barriere up-and-out B={B:.0f}':<38}{bar.price:>10.4f}"
          f"   +/- {bar.ci95:.4f}")
    print(f"{'  decote vs vanille':<38}{bar.price/van-1:>9.1%}")

    # --- pourquoi la surface compte : quelle vol prendre pour la barriere ? --
    iv_K, T_lu = vol_au_strike(chain, T_exo, K, spot)
    iv_B, _ = vol_au_strike(chain, T_exo, B, spot)
    print("\n" + "-" * 74)
    print("Sensibilite au POINT de la surface retenu (barriere up-and-out)")
    print("-" * 74)
    print(f"{'vol utilisee':<30}{'valeur':>10}{'prix':>12}{'ecart':>10}")
    ref = None
    for lib, s in [("ATM interpolee", sigma),
                   (f"lue au strike K={K:.0f}", iv_K),
                   (f"lue a la barriere B={B:.0f}", iv_B)]:
        px = mc_barrier(spot, K, B, r, s, T_exo, 63, 200_000, q=q, rng=g(2)).price
        if ref is None:
            ref = px
        print(f"{lib:<30}{s:>10.2%}{px:>12.4f}{px/ref-1:>9.1%}")
    print("\nUn seul chiffre de volatilite ne suffit pas a coter une barriere :")
    print("le produit depend de toute la surface, pas d'un point.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAUT)
