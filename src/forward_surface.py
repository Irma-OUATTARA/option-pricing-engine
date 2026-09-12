"""
forward_surface.py
==================
Extraction du forward de marche et inversion de la volatilite implicite en
espace-forward, pour la reconstruction de la surface de volatilite.

POURQUOI CE MODULE.
  Inverser Black-Scholes avec un taux SUPPOSE (r=5 %) revient a fabriquer
  soi-meme le forward F = S e^{(r-q)T}. Si r est faux, F est faux, et comme
  Black-Scholes ne "voit" le marche que par le forward, la volatilite
  implicite est fausse : l'aile put et l'aile call ne se raccordent plus a la
  monnaie (une "marche"). La solution : ne pas supposer r, mais LIRE le forward
  dans les prix, puis inverser avec ce forward-la.

UTILISATION (carnet 04, en remplacement de compute_iv) :
    from forward_surface import reconstruire_iv
    chain = reconstruire_iv(chain, spot, r_disc=0.04)   # ajoute les colonnes iv, F

Reference : Hull, ch. 5 (forward) et ch. 11 (parite call-put).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar


# ==========================================================================
# 1. Black-Scholes en espace FORWARD : le forward F et l'actualisation DF
#    sont deux entrees separees.
#       C = DF * [ F N(d1) - K N(d2) ]
#       d1 = [ ln(F/K) + s^2 T / 2 ] / (s sqrt(T)),   d2 = d1 - s sqrt(T)
# ==========================================================================
def bs_forward(F, K, T, sigma, DF, typ):
    if sigma <= 0 or T <= 0:
        intrinseque = max(F - K, 0.0) if typ == "call" else max(K - F, 0.0)
        return DF * intrinseque
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if typ == "call":
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def iv_forward(price, F, K, T, DF, typ):
    """Volatilite implicite en espace-forward. NaN si le prix viole les
    bornes de non-arbitrage (aucune volatilite ne peut le reproduire)."""
    borne = DF * (max(F - K, 0.0) if typ == "call" else max(K - F, 0.0))
    if not np.isfinite(price) or price <= borne + 1e-8 or price >= DF * F:
        return np.nan
    try:
        return brentq(lambda s: bs_forward(F, K, T, s, DF, typ) - price,
                      1e-4, 5.0, maxiter=200)
    except Exception:
        return np.nan


# ==========================================================================
# 2. Deux facons de LIRE le forward dans les prix.
# --------------------------------------------------------------------------
# 2a. Par PARITE call-put (la methode de reference) :  F = K + (C - P) / DF.
#     Necessite des strikes ou call ET put sont cotes (chaine brute).
def _forward_parite(strikes, calls, puts, S, DF, n_near=5):
    strikes = np.asarray(strikes, float)
    c = np.asarray(calls, float); p = np.asarray(puts, float)
    ok = np.isfinite(c) & np.isfinite(p) & (c > 0) & (p > 0)
    if ok.sum() < 2:
        return None
    idx = np.where(ok)[0]
    idx = idx[np.argsort(np.abs(strikes[idx] - S))][:n_near]  # proches monnaie
    return float(np.median(strikes[idx] + (c[idx] - p[idx]) / DF))


# 2b. Par LISSAGE du smile (quand on n'a que des options hors de la monnaie,
#     donc pas de strike commun call/put). On cherche le forward F qui rend le
#     smile le plus lisse : la "marche" put/call est justement ce qui gonfle
#     le residu d'un ajustement parabolique. C'est l'equivalent pratique de la
#     parite quand la parite n'est pas calculable.
def _forward_lissage(K, mid, typ, S, DF, T, r_disc):
    # HEURISTIQUE DE REPLI, pas la definition du forward.
    # Utilisee UNIQUEMENT quand la parite call-put est impossible (aucun strike
    # portant a la fois un call et un put bien cotes). On cherche alors le forward
    # F qui rend le smile le plus lisse (residu d'une parabole minimal), car un
    # forward mal place cree une "marche" put/call qui gonfle ce residu.
    # En salle de marche, on prefere la parite, la courbe des taux, les dividendes
    # ou les futures ; ce lissage n'est qu'un secours.
    K = np.asarray(K, float); mid = np.asarray(mid, float)
    typ = np.asarray(typ)
    def residu(F):
        iv = np.array([iv_forward(mid[i], F, K[i], T, DF, typ[i])
                       for i in range(len(K))])
        m = np.isfinite(iv)
        if m.sum() < 5:
            return 1e9
        x = np.log(K[m] / F)
        A = np.vstack([np.ones_like(x), x, x * x]).T
        coef, *_ = np.linalg.lstsq(A, iv[m], rcond=None)
        return float(np.sum((A @ coef - iv[m]) ** 2))
    lo, hi = S * np.exp((r_disc - 0.15) * T), S * np.exp((r_disc + 0.15) * T)
    return float(minimize_scalar(residu, bounds=(lo, hi), method="bounded").x)


# ==========================================================================
# 3. FONCTION PRINCIPALE : reconstruit la volatilite implicite propre.
#    Pour chaque maturite : lit le forward, puis inverse en espace-forward.
#    Repli sur le forward theorique F = S e^{(r-q)T} si le forward extrait
#    est absurde (bord court bruite).
# ==========================================================================
def reconstruire_iv(chain, spot, r_disc=0.04, q=0.004, r_band=(-0.01, 0.10)):
    """
    chain : DataFrame avec au moins les colonnes type, K, T, mid.
            (chaine hors-de-la-monnaie OU chaine brute : les deux marchent.)
    Retour : le meme DataFrame avec les colonnes ajoutees
             iv (volatilite implicite propre), F (forward), source_F.
    """
    out = chain.copy()
    out["iv"] = np.nan; out["F"] = np.nan; out["source_F"] = ""
    for T, gT in out.groupby("T"):
        DF = np.exp(-r_disc * T)
        F_theo = spot * np.exp((r_disc - q) * T)

        # essai 1 : parite (si des strikes portent call ET put)
        piv = gT.pivot_table(index="K", columns="type", values="mid",
                             aggfunc="first")
        F, src = None, ""
        if {"call", "put"}.issubset(piv.columns):
            F = _forward_parite(piv.index.values, piv["call"].values,
                                piv["put"].values, spot, DF)
            if F is not None:
                src = "parite"
        # essai 2 : lissage (chaine hors-de-la-monnaie seule)
        if F is None:
            F = _forward_lissage(gT["K"].values, gT["mid"].values,
                                 gT["type"].values, spot, DF, T, r_disc)
            src = "lissage"
        # garde-fou : forward absurde -> repli theorique
        r_impl = np.log(F / spot) / T + q
        if not (r_band[0] <= r_impl <= r_band[1]):
            F, src = F_theo, "theorique"

        # inversion en espace-forward pour toute la maturite
        for i, row in gT.iterrows():
            out.at[i, "iv"] = iv_forward(row["mid"], F, row["K"], T, DF,
                                         row["type"])
            out.at[i, "F"] = F
            out.at[i, "source_F"] = src
    return out


# ==========================================================================
# TEST sur la chaine reelle exportee.
# ==========================================================================
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("chaine_options.csv")[["type", "K", "T", "mid", "spot"]]
    S = float(df["spot"].iloc[0])
    res = reconstruire_iv(df, S, r_disc=0.04)
    print(f"spot = {S:.2f}\n")
    print(f"{'T(j)':>6} {'F':>8} {'source':>10} {'saut ATM':>10}")
    for T, g in res.groupby("T"):
        g = g.sort_values("K")
        p = g[g["type"] == "put"].iloc[-1]; c = g[g["type"] == "call"].iloc[0]
        print(f"{T*365:6.0f} {g['F'].iloc[0]:8.1f} {g['source_F'].iloc[0]:>10} "
              f"{c['iv']-p['iv']:+9.2%}")