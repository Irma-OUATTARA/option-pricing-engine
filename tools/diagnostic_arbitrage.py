"""Diagnostic d'absence d'arbitrage de la surface AAPL.

Deux tests, sur la SEULE fenetre liquide k in [-0.13, 0.13] (la ou la parabole
est fiable ; hors de cette fenetre on ne revendique rien, cf. la formule des
moments de Roger Lee).

  1) ARBITRAGE CALENDAIRE. La variance totale w(k,T) = sigma(k,T)^2 * T doit
     croitre avec la maturite, a chaque moneyness k fixe (Gatheral-Jacquier,
     Lemme 2.1 : d_T w >= 0 est necessaire et suffisant). On verifie la
     monotonie de chaque colonne de la surface.

  2) ARBITRAGE PAPILLON. La densite implicite de chaque tranche doit rester
     positive. Gatheral la condense dans une fonction g(k) : la tranche est sans
     arbitrage papillon ssi g(k) >= 0 pour tout k. On calcule g sur la fenetre.

Usage : python tools/diagnostic_arbitrage.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RACINE, "data", "chaine_AAPL_2026-07-27.csv")
K_LO, K_HI = -0.13, 0.13          # fenetre liquide en k = ln(K/F)
R, Q = 0.04, 0.004


def g_gatheral(k, w, wp, wpp):
    """Fonction g de Gatheral (arXiv 1204.0646, eq. 2.1).
    Densite positive <=> g(k) >= 0."""
    return (1 - k * wp / (2 * w))**2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2


def main():
    df = pd.read_csv(CSV)
    S = df["spot"].iloc[0]
    df["F"] = S * np.exp((R - Q) * df["T"])
    df["k"] = np.log(df["K"] / df["F"])
    mats = np.sort(df["T"].unique())

    kg = np.linspace(K_LO, K_HI, 120)
    dk = kg[1] - kg[0]
    W = np.zeros((len(mats), len(kg)))       # variance totale par (maturite, k)

    print("=" * 68)
    print("DIAGNOSTIC D'ARBITRAGE  (fenetre liquide k in [%.2f, %.2f])"
          % (K_LO, K_HI))
    print("=" * 68)

    # ---- test papillon, tranche par tranche ----
    print("\n[1] ARBITRAGE PAPILLON  (densite positive <=> g(k) >= 0)")
    print(f"    {'maturite':>9}{'min g(k)':>12}{'verdict':>16}")
    res_bf = []
    for i, T in enumerate(mats):
        g = df[df["T"] == T]
        coef = np.polyfit(g["k"].values, g["iv"].values, 2)   # parabole en ln(K/F)
        iv = np.polyval(coef, kg)
        w = iv**2 * T                                         # variance totale
        W[i] = w
        wp = np.gradient(w, dk)
        wpp = np.gradient(wp, dk)
        gk = g_gatheral(kg, w, wp, wpp)
        mn = float(gk.min())
        ok = mn >= 0
        res_bf.append(ok)
        print(f"    {round(T*365):>6} j{mn:>12.4f}{'  OK' if ok else '  VIOLATION':>16}")

    # ---- test calendaire, colonne par colonne ----
    print("\n[2] ARBITRAGE CALENDAIRE  (variance totale croissante en T)")
    diffs = np.diff(W, axis=0)                # W[i+1]-W[i] par colonne
    viol = (diffs < -1e-12)
    n_viol = int(viol.sum())
    print(f"    colonnes de moneyness testees : {W.shape[1]}")
    print(f"    paires de maturites testees   : {W.shape[0]-1}")
    print(f"    violations (w decroissant)    : {n_viol}")
    if n_viol == 0:
        wmin, wmax = W[0].min(), W[-1].max()
        print(f"    variance totale croissante partout : OK "
              f"(de {wmin:.4f} a {wmax:.4f})")

    # ---- synthese ----
    print("\n" + "=" * 68)
    bf_ok = all(res_bf)
    cal_ok = (n_viol == 0)
    print("SYNTHESE  papillon : %s   |   calendaire : %s"
          % ("OK" if bf_ok else "VIOLATION", "OK" if cal_ok else "VIOLATION"))
    if bf_ok and cal_ok:
        print("La surface est sans arbitrage statique sur la fenetre liquide.")
    print("=" * 68)
    return {"papillon_ok": bf_ok, "calendaire_ok": cal_ok,
            "n_viol_calendaire": n_viol}


if __name__ == "__main__":
    main()
