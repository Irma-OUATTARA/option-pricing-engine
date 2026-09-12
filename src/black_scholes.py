"""
Modèle de Black-Scholes-Merton : prix analytique et Greeks.

RÔLE DANS LE PROJET
-------------------
Coeur analytique du moteur. Sert de RÉFÉRENCE EXACTE pour valider toutes les
autres méthodes :
  - l'arbre binomial (Phase 2) doit converger vers ce prix ;
  - le Monte Carlo (Phase 3) doit y converger (à l'erreur statistique près) ;
  - c'est aussi la fonction qu'on INVERSE en Phase 4 pour extraire la
    volatilité implicite (on cherche le sigma tel que prix_BS = prix_marché).

On inclut le rendement de dividende continu `q`. Poser q = taux étranger donne
directement le modèle de Garman-Kohlhagen utilisé pour les options de change (FX).
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr

# Fonction de répartition et densité de la loi N(0,1).
#
# On appelle scipy.special.ndtr et NON scipy.stats.norm.cdf. Mesuré sur ce
# projet : 0,4 us contre 45,5 us par appel scalaire, soit un facteur 114. La
# différence n'est pas dans le calcul mais dans la couche de dispatch de
# scipy.stats (validation d'arguments, gel de distribution). Sur un prix, qui
# fait deux appels, c'est 90 us de surcoût pur ; sur l'inversion d'une surface
# de 400 options a ~5 iterations de Newton, c'est plusieurs secondes.
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _cdf(x):
    return ndtr(x)


def _pdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / _SQRT_2PI


class BlackScholes:
    """Option européenne valorisée par la formule de Black-Scholes-Merton.

    Paramètres
    ----------
    S : prix spot du sous-jacent
    K : strike (prix d'exercice)
    r : taux sans risque (continu, annualisé)
    sigma : volatilité (annualisée)
    T : maturité en années
    q : rendement de dividende continu (0 par défaut ; = taux étranger pour le FX)
    """

    def __init__(self, S, K, r, sigma, T, q=0.0):
        if T <= 0:
            raise ValueError("La maturité T doit être strictement positive.")
        if sigma <= 0:
            raise ValueError("La volatilité sigma doit être strictement positive.")
        self.S = float(S)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.q = float(q)

        # -- Termes intermédiaires, calculés UNE FOIS --------------------------
        # d1 et d2 sont partagés par le prix et par les cinq Greeks. Les
        # exposer en @property les faisait recalculer à chaque accès : un
        # appel à greeks() déclenchait une dizaine de log/sqrt identiques,
        # soit 495 us contre 104 us pour price() seul. On les fige ici.
        #
        # CONTRAT : l'objet est IMMUABLE. Modifier self.sigma après
        # construction laisserait d1/d2 périmés. Pour changer un paramètre,
        # construire un nouvel objet (c'est ce que fait implied_vol à chaque
        # itération de Newton).
        self._sqrt_T = np.sqrt(self.T)
        self._vol_T = self.sigma * self._sqrt_T
        self.d1 = (np.log(self.S / self.K)
                   + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / self._vol_T
        self.d2 = self.d1 - self._vol_T

    # -- Prix ------------------------------------------------------------------
    def price(self, option_type="call"):
        """Prix de l'option. option_type : 'call' ou 'put'."""
        d1, d2 = self.d1, self.d2
        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)
        if option_type == "call":
            return self.S * disc_q * _cdf(d1) - self.K * disc_r * _cdf(d2)
        elif option_type == "put":
            return self.K * disc_r * _cdf(-d2) - self.S * disc_q * _cdf(-d1)
        raise ValueError("option_type doit être 'call' ou 'put'.")

    # -- Greeks ----------------------------------------------------------------
    def delta(self, option_type="call"):
        """dPrix/dS : sensibilité au prix du sous-jacent."""
        disc_q = np.exp(-self.q * self.T)
        if option_type == "call":
            return disc_q * _cdf(self.d1)
        elif option_type == "put":
            return -disc_q * _cdf(-self.d1)
        raise ValueError("option_type doit être 'call' ou 'put'.")

    def gamma(self):
        """d2Prix/dS2 : vitesse de variation du delta (identique call/put)."""
        disc_q = np.exp(-self.q * self.T)
        return disc_q * _pdf(self.d1) / (self.S * self._vol_T)

    def vega(self):
        """dPrix/dsigma : sensibilité à la volatilité (identique call/put).

        Valeur brute (variation de vol de 1.00). Diviser par 100 pour l'effet
        d'une hausse de vol de 1 point (convention de marché).
        """
        disc_q = np.exp(-self.q * self.T)
        return self.S * disc_q * _pdf(self.d1) * self._sqrt_T

    def theta(self, option_type="call"):
        """dPrix/dt : érosion temporelle (par an ; diviser par 365 pour /jour)."""
        d1, d2 = self.d1, self.d2
        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)
        common = -(self.S * disc_q * _pdf(d1) * self.sigma) / (2 * self._sqrt_T)
        if option_type == "call":
            return (common
                    - self.r * self.K * disc_r * _cdf(d2)
                    + self.q * self.S * disc_q * _cdf(d1))
        elif option_type == "put":
            return (common
                    + self.r * self.K * disc_r * _cdf(-d2)
                    - self.q * self.S * disc_q * _cdf(-d1))
        raise ValueError("option_type doit être 'call' ou 'put'.")

    def rho(self, option_type="call"):
        """dPrix/dr : sensibilité au taux (brute ; diviser par 100 pour +1%)."""
        d2 = self.d2
        disc_r = np.exp(-self.r * self.T)
        if option_type == "call":
            return self.K * self.T * disc_r * _cdf(d2)
        elif option_type == "put":
            return -self.K * self.T * disc_r * _cdf(-d2)
        raise ValueError("option_type doit être 'call' ou 'put'.")

    def greeks(self, option_type="call"):
        """Renvoie tous les Greeks dans un dictionnaire."""
        return {
            "price": self.price(option_type),
            "delta": self.delta(option_type),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(option_type),
            "rho": self.rho(option_type),
        }
