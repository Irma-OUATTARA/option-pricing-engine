"""
Simulation de trajectoires de prix (mouvement brownien géométrique).

RÔLE DANS LE PROJET
-------------------
Ce module est la SOURCE UNIQUE d'aléa du moteur. Toute trajectoire simulée,
où qu'elle soit consommée (pricer Monte Carlo, carnets, calcul de Greeks),
passe par ici :
  - `simulate_gbm_terminal` : ne renvoie que S_T -> options européennes.
  - `simulate_gbm_paths`    : renvoie les trajectoires complètes -> options
                              exotiques path-dependent (asiatiques, barrière...).

Pour PRICER, on appelle ces fonctions avec `mu = r - q` : c'est la mesure
risque-neutre. Le module ne connaît volontairement ni `r` ni `q` séparément,
seulement une dérive générique `mu`. C'est ce qui permet de l'utiliser aussi
sous la mesure historique (mu = rendement réel estimé) pour du backtest ou de
la VaR, sans une ligne de changement.

CONVENTION DE FORME DES TRAJECTOIRES
------------------------------------
`simulate_gbm_paths` expose `include_spot` :
  - True  -> forme (n_paths, n_steps + 1), colonne 0 = S0 (instant t=0)
  - False -> forme (n_paths, n_steps),     première colonne = t_1 = dt

Ce n'est pas un détail cosmétique. Une asiatique dont les fixings sont
t_1..t_n ne doit PAS inclure S0 dans la moyenne : l'inclure revient à ajouter
un point non aléatoire, ce qui abaisse artificiellement la variance de la
moyenne et donc le prix. La convention de marché est d'exclure t=0.
"""
from __future__ import annotations
import numpy as np


def draw_normals(shape, rng=None, antithetic=False):
    """Tire des N(0,1), avec appariement antithétique optionnel.

    Point unique de tirage du moteur : centraliser ici garantit que la
    convention d'APPARIEMENT est la même partout. C'est indispensable, car
    l'estimateur de SE antithetique (cf. monte_carlo._summarize)
    suppose que la paire du tirage i est le tirage i + m.

    Avec `antithetic=True`, on tire m = n/2 aléas et on renvoie la
    concaténation [Z, -Z] : la ligne i et la ligne i+m sont antithétiques.
    """
    if rng is None:
        rng = np.random.default_rng()

    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)

    if not antithetic:
        return rng.standard_normal(size=shape)

    n = shape[0]
    if n % 2 != 0:
        raise ValueError(
            "n_paths doit etre pair en mode antithetique "
            f"(recu {n}) : les tirages vont par paires."
        )
    half = (n // 2,) + shape[1:]
    Z = rng.standard_normal(size=half)
    return np.concatenate([Z, -Z], axis=0)


def simulate_gbm_terminal(S0, mu, sigma, T, n_paths, rng=None, antithetic=False):
    """Simule uniquement le prix terminal S_T par la solution exacte du GBM.

    Rapide (pas de boucle temporelle) : suffisant pour les options
    européennes, dont le payoff ne dépend que de S_T.

    Renvoie un tableau 1D de taille (n_paths,).
    """
    Z = draw_normals(n_paths, rng, antithetic)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, rng=None,
                       antithetic=False, include_spot=True):
    """Simule des trajectoires complètes du GBM par la solution exacte.

    Nécessaire pour les options path-dependent, dont le payoff dépend de tout
    le chemin (moyenne, maximum, franchissement de barrière...).

    Le schéma est EXACT, pas un schéma d'Euler : on simule directement
    ln S_t = ln S0 + (mu - sigma^2/2) t + sigma W_t, dont la loi est connue.
    Il n'y a donc AUCUN biais de discrétisation sur la loi marginale de S_t,
    quel que soit n_steps. Le biais qui subsiste est d'une autre nature : il
    porte sur les fonctionnelles du CHEMIN (maximum, minimum), que l'on
    n'observe qu'aux n_steps dates de la grille. C'est ce biais-là, et lui
    seul, que traite la correction de continuite des options a barriere.

    Renvoie (n_paths, n_steps + 1) si include_spot, sinon (n_paths, n_steps).
    """
    dt = T / n_steps
    Z = draw_normals((n_paths, n_steps), rng, antithetic)

    # W_t par accroissements cumulés, puis solution exacte
    W = np.cumsum(np.sqrt(dt) * Z, axis=1)
    t = np.arange(1, n_steps + 1) * dt
    paths = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    if include_spot:
        paths = np.hstack([np.full((n_paths, 1), float(S0)), paths])
    return paths
