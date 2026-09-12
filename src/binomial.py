"""
Arbre binomial de Cox-Ross-Rubinstein (CRR) : pricing par récursion arrière.

RÔLE DANS LE PROJET
-------------------
L'arbre est une version DISCRÈTE du mouvement brownien géométrique. Il remplit
deux fonctions que Black-Scholes ne couvre pas seul :
  1. VALIDATION : pour une option européenne, le prix de l'arbre doit CONVERGER
     vers le prix Black-Scholes quand le nombre de pas augmente -> 2e preuve
     indépendante que notre moteur est correct.
  2. OPTIONS AMÉRICAINES : à chaque noeud on peut comparer "garder l'option" vs
     "l'exercer maintenant". Black-Scholes n'a pas de formule pour ça ; l'arbre,
     si, car il gère naturellement la décision d'exercice anticipé.

IDÉE CLÉ (valorisation risque-neutre, rendue concrète)
------------------------------------------------------
À chaque pas, le prix monte (x u) ou descend (x d). La probabilité "risque-neutre"
p est choisie pour que l'actif rapporte en moyenne le taux sans risque r. Le prix
d'une option en un noeud = espérance actualisée de sa valeur au pas suivant :
      V = e^{-r dt} * [ p * V_haut + (1 - p) * V_bas ]
"""
from __future__ import annotations
import numpy as np


class BinomialTree:
    """Option valorisée par un arbre binomial CRR.

    Paramètres
    ----------
    S, K, r, sigma, T : comme Black-Scholes (spot, strike, taux, vol, maturité)
    n_steps : nombre de pas de l'arbre (plus il est grand, plus on est proche
              du modèle continu -> proche de Black-Scholes)
    q : rendement de dividende continu (0 par défaut)
    """

    def __init__(self, S, K, r, sigma, T, n_steps=500, q=0.0):
        if T <= 0 or sigma <= 0 or n_steps < 1:
            raise ValueError("T, sigma > 0 et n_steps >= 1 requis.")
        self.S = float(S)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.q = float(q)

    @property
    def params(self):
        """Paramètres CRR : (dt, u, d, p).

        u = e^{sigma sqrt(dt)}  (facteur de hausse)
        d = 1/u                 (facteur de baisse, symétrique)
        p = (e^{(r-q) dt} - d)/(u - d)   (probabilité risque-neutre)

        u et d sont choisis pour reproduire la volatilité sigma du GBM ;
        p pour que l'actif dérive au taux (r - q) : c'est la mesure risque-neutre.
        """
        dt = self.T / self.n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        return dt, u, d, p

    def price(self, option_type="call", exercise="european"):
        """Prix de l'option.

        option_type : 'call' ou 'put'
        exercise    : 'european' (exercice à l'échéance seulement)
                      ou 'american' (exercice possible à tout moment)
        """
        return self._price_and_nodes(option_type, exercise)[0]

    def _price_and_nodes(self, option_type="call", exercise="european"):
        """Coeur de la récursion. Renvoie le prix ET les valeurs des premiers
        noeuds, nécessaires au calcul des Greeks.

        Renvoie (prix, infos) où `infos` est un dict contenant, si n >= 2, les
        valeurs de l'option et du sous-jacent aux dates 0, 1 et 2 de l'arbre.
        Ces noeuds sont déjà calculés par la récursion : les récupérer ne coûte
        rien, et ils suffisent à lire delta, gamma et theta sans reconstruire
        l'arbre (cf. méthode `greeks`).
        """
        if option_type not in ("call", "put"):
            raise ValueError("option_type doit être 'call' ou 'put'.")
        if exercise not in ("european", "american"):
            raise ValueError("exercise doit être 'european' ou 'american'.")

        dt, u, d, p = self.params
        n = self.n_steps
        disc = np.exp(-self.r * dt)

        # --- 1. Prix du sous-jacent aux feuilles (échéance) ---
        j = np.arange(n + 1)                 # nb de hausses : 0..n
        ST = self.S * u**j * d**(n - j)

        # --- 2. Payoff aux feuilles ---
        if option_type == "call":
            V = np.maximum(ST - self.K, 0.0)
        else:
            V = np.maximum(self.K - ST, 0.0)

        # On mémorise les valeurs de l'option aux dates 2, 1 et 0 en remontant.
        nodes = {}

        # --- 3. Récursion arrière : on remonte l'arbre ---
        for i in range(n, 0, -1):
            V = disc * (p * V[1:i + 1] + (1.0 - p) * V[0:i])
            if exercise == "american":
                # Prix du sous-jacent au niveau i-1
                jj = np.arange(i)
                S_nodes = self.S * u**jj * d**(i - 1 - jj)
                if option_type == "call":
                    intrinsic = np.maximum(S_nodes - self.K, 0.0)
                else:
                    intrinsic = np.maximum(self.K - S_nodes, 0.0)
                # À chaque noeud : garder (V) ou exercer (intrinsic) -> le mieux
                V = np.maximum(V, intrinsic)
            # i-1 est la date qu'on vient d'atteindre : on capture 0, 1, 2.
            if (i - 1) in (0, 1, 2):
                nodes[i - 1] = V.copy()

        infos = {}
        if n >= 2:
            infos["V0"] = float(nodes[0][0])
            infos["V1"] = nodes[1]          # 2 valeurs : bas, haut
            infos["V2"] = nodes[2]          # 3 valeurs
            infos["S1"] = np.array([self.S * d, self.S * u])
            infos["S2"] = np.array([self.S * d * d, self.S, self.S * u * u])
            infos["dt"] = dt
        return float(V[0]), infos

    def greeks(self, option_type="call", exercise="european"):
        """Delta, gamma et theta lus directement sur le treillis.

        Aucune reconstruction d'arbre, aucun bump : les trois sensibilités se
        lisent sur des noeuds déjà calculés par la récursion de prix. C'est
        l'avantage propre de l'arbre — là où Monte-Carlo doit re-simuler pour
        chaque Greek (différences finies), l'arbre les donne « gratuitement ».

        DELTA (pente au premier pas). Entre les deux noeuds de la date 1 :
            delta = (V_haut - V_bas) / (S*u - S*d)

        GAMMA (courbure, à la date 2). La recombinaison (u*d = 1) fait que le
        noeud central de la date 2 revient au spot initial. On dispose donc de
        trois prix S2 = (S d^2, S, S u^2) et de leurs valeurs. On estime les
        deux pentes latérales, puis leur variation :
            delta_haut = (V2[2] - V2[1]) / (S2[2] - S2[1])
            delta_bas  = (V2[1] - V2[0]) / (S2[1] - S2[0])
            gamma = (delta_haut - delta_bas) / (0.5*(S2[2] - S2[0]))

        THETA (érosion dans le temps). Le noeud central de la date 2 porte le
        MÊME sous-jacent que la racine (=S), à 2*dt plus tard. La différence de
        valeur, ramenée au temps, donne theta directement :
            theta = (V2[1] - V0) / (2*dt)

        Renvoie un dict {'delta','gamma','theta'}. Exige n_steps >= 2.
        """
        if self.n_steps < 2:
            raise ValueError("greeks() exige n_steps >= 2.")
        _, info = self._price_and_nodes(option_type, exercise)
        V0, V1, V2 = info["V0"], info["V1"], info["V2"]
        S1, S2, dt = info["S1"], info["S2"], info["dt"]

        delta = (V1[1] - V1[0]) / (S1[1] - S1[0])

        delta_haut = (V2[2] - V2[1]) / (S2[2] - S2[1])
        delta_bas = (V2[1] - V2[0]) / (S2[1] - S2[0])
        gamma = (delta_haut - delta_bas) / (0.5 * (S2[2] - S2[0]))

        theta = (V2[1] - V0) / (2.0 * dt)

        return {"delta": float(delta), "gamma": float(gamma),
                "theta": float(theta)}


def convergence_data(S, K, r, sigma, T, steps_list, option_type="call", q=0.0):
    """Renvoie les prix de l'arbre européen pour une liste de n_steps.

    Utilisé pour tracer la convergence vers Black-Scholes.
    """
    prices = []
    for n in steps_list:
        tree = BinomialTree(S, K, r, sigma, T, n_steps=n, q=q)
        prices.append(tree.price(option_type, "european"))
    return np.array(prices)
