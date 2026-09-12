"""
Volatilité implicite : inversion de la formule de Black-Scholes.

RÔLE DANS LE PROJET
-------------------
La volatilité est le SEUL paramètre de Black-Scholes qu'on ne peut pas observer
directement. La "volatilité implicite" est celle qui, injectée dans Black-Scholes,
redonne EXACTEMENT le prix observé sur le marché. C'est la vision du marché sur la
volatilité future.

On INVERSE donc la fonction prix -> on cherche le sigma tel que BS(sigma) = prix_marché.
Deux méthodes :
  - Newton-Raphson : rapide (utilise la dérivée = vega), converge en ~3-5 itérations.
  - Brent : robuste (encadrement), sert de filet de sécurité si Newton diverge.

C'est la brique qui permet, en agrégeant sur tous les strikes et maturités, de
reconstruire le SMILE puis la SURFACE de volatilité (le livrable signature).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

from src.black_scholes import BlackScholes

# Domaine de recherche, PARTAGÉ par Newton et par Brent.
#
# Il y avait ici une incohérence : Newton bornait son amorce à 5.0 mais
# laissait ses itérés monter jusqu'à 10, tandis que Brent ne balayait que
# [1e-6, 5]. Une volatilité implicite entre 5 et 10 était donc trouvable par
# Newton et introuvable par le filet censé le rattraper - le comportement du
# moteur dépendait de quelle méthode avait convergé. Un seul domaine désormais.
SIGMA_MIN = 1e-6
SIGMA_MAX = 5.0


def _bs_price(sigma, S, K, r, T, option_type, q):
    return BlackScholes(S, K, r, sigma, T, q).price(option_type)


def implied_vol_newton(price, S, K, r, T, option_type="call", q=0.0,
                       tol=1e-8, max_iter=100):
    """Volatilité implicite par Newton-Raphson.

    Itère : sigma <- sigma - (BS(sigma) - prix) / vega(sigma)
    Renvoie np.nan si l'itération échoue (vega trop faible, divergence).
    """
    # Estimation initiale : approximation de Brenner-Subrahmanyam
    # (bonne pour les options proches de la monnaie)
    sigma = np.sqrt(2 * np.pi / T) * price / S
    sigma = min(max(sigma, 1e-3), SIGMA_MAX)

    for _ in range(max_iter):
        bs = BlackScholes(S, K, r, sigma, T, q)
        diff = bs.price(option_type) - price
        if abs(diff) < tol:
            return sigma
        v = bs.vega()                    # dérivée du prix par rapport à sigma
        if v < 1e-8:                     # vega trop faible : Newton instable
            return np.nan
        sigma -= diff / v
        if sigma < SIGMA_MIN or sigma > SIGMA_MAX:   # sortie de domaine
            return np.nan
    return np.nan


def implied_vol_brent(price, S, K, r, T, option_type="call", q=0.0,
                      lo=SIGMA_MIN, hi=SIGMA_MAX):
    """Volatilité implicite par la méthode de Brent (encadrement robuste)."""
    f = lambda sig: _bs_price(sig, S, K, r, T, option_type, q) - price
    try:
        # Il faut un changement de signe sur [lo, hi]
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol(price, S, K, r, T, option_type="call", q=0.0):
    """Volatilité implicite : Newton d'abord, Brent en secours.

    Renvoie np.nan si le prix est hors des bornes d'arbitrage (aucune vol
    implicite ne peut le reproduire).
    """
    if price is None or np.isnan(price) or price <= 0:
        return np.nan

    # Bornes de non-arbitrage : le prix doit être entre la valeur intrinsèque
    # actualisée et le spot (call) / strike actualisé (put).
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        intrinsic = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        intrinsic = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    if price < intrinsic - 1e-8 or price > upper + 1e-8:
        return np.nan

    iv = implied_vol_newton(price, S, K, r, T, option_type, q)
    if np.isnan(iv):
        iv = implied_vol_brent(price, S, K, r, T, option_type, q)
    return iv
