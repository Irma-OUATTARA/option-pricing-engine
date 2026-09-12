"""
Pricing par simulation de Monte Carlo : moteur générique + produits.

RÔLE DANS LE PROJET
-------------------
Le Monte Carlo est la seule méthode capable de pricer les options
path-dependent (payoff dépendant de TOUTE la trajectoire) : asiatiques,
barrières, lookbacks. Ni Black-Scholes ni l'arbre binomial ne le font.

ARCHITECTURE : LE PAYOFF EST UN ARGUMENT
----------------------------------------
Le moteur `mc_price` ne connaît aucun produit. Il reçoit un objet `Payoff`
- une fonction de la matrice des trajectoires - et ne fait que quatre choses :
simuler, appliquer le payoff, actualiser, résumer.

Conséquence directe : ajouter un produit ne demande PAS de modifier le
moteur. Une asiatique, une digitale ou une lookback tiennent chacune en trois
lignes (cf. section "Catalogue de payoffs"). C'est la traduction en code d'une
observation mathématique : dans l'algorithme de simulation, la seule étape qui
change d'un produit à l'autre est le calcul du payoff.

Les fonctions `mc_european`, `mc_asian`, `mc_barrier` sont conservées : ce sont
désormais de simples enveloppes autour de `mc_price`, gardées pour la
compatibilité des carnets et des tests.

RÉDUCTION DE VARIANCE
---------------------
L'erreur décroît en 1/sqrt(N) : diviser l'erreur par 2 coûte 4x plus de
simulations. Deux techniques sont implémentées, antithétique et variable de
contrôle, la seconde avec coefficient beta optimal estimé sur l'échantillon.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

from src.black_scholes import BlackScholes
from src.simulation import simulate_gbm_terminal, simulate_gbm_paths

# Constante de Broadie-Glasserman-Kou : beta = -zeta(1/2)/sqrt(2 pi)
BGK_BETA = 0.5826


# ===========================================================================
#  Résultat d'un pricing MC : prix + incertitude
# ===========================================================================
class MCResult:
    """Prix Monte Carlo avec son intervalle de confiance."""

    def __init__(self, price, std_error, n_paths, sample=None):
        self.price = float(price)
        self.std_error = float(std_error)      # SE (erreur type)
        self.n_paths = int(n_paths)
        # Echantillon i.i.d. effectivement utilise pour calculer std_error :
        # les payoffs actualises, ou les moyennes de paires en mode
        # antithetique. Il permet d'empiler plusieurs executions et de
        # recalculer l'incertitude sur le cumul, sans aucune extrapolation.
        self.sample = sample

    @property
    def ci95(self):
        """Demi-largeur de l'intervalle de confiance à 95 %."""
        return 1.96 * self.std_error

    def __repr__(self):
        return (f"prix = {self.price:.4f} +/- {self.ci95:.4f} (IC95, "
                f"N={self.n_paths:,})")


def _summarize(discounted_payoffs, antithetic=False, n_paths=None):
    """Moyenne + SE (erreur type) d'un echantillon de payoffs actualises.

    ATTENTION (subtilité importante) : avec des variables antithétiques, les
    tirages ne sont PAS independants (Z et -Z vont par paires). SE
    doit se calculer sur la MOYENNE DE CHAQUE PAIRE, qui elle est i.i.d.
    Sinon on sous-estime/masque complètement le gain de la technique.

    L'appariement (ligne i <-> ligne i+m) est garanti par
    src.simulation.draw_normals, qui est le point unique de tirage du moteur.

    `n_paths` permet de rapporter le nombre de TRAJECTOIRES simulees meme
    lorsque l'echantillon transmis est deja constitue de moyennes de paires
    (cas antithetique + variable de controle, cf. mc_price).
    """
    n = len(discounted_payoffs)
    price = discounted_payoffs.mean()

    if antithetic:
        m = n // 2
        pair_means = 0.5 * (discounted_payoffs[:m] + discounted_payoffs[m:2 * m])
        se = pair_means.std(ddof=1) / np.sqrt(m)
        iid = pair_means
    else:
        se = discounted_payoffs.std(ddof=1) / np.sqrt(n)
        iid = discounted_payoffs

    return MCResult(price, se, n_paths if n_paths is not None else n,
                    sample=iid)


# ===========================================================================
#  Payoff : une fonction des trajectoires, plus un drapeau
# ===========================================================================
class Payoff:
    """Enveloppe d'une fonction payoff.

    `fn` reçoit une matrice (n_paths, n_dates) SANS la colonne t=0 et renvoie
    un vecteur (n_paths,) de flux à l'échéance, NON actualisés.

    `needs_path` dit au moteur s'il doit construire la trajectoire complète
    (coûteux) ou s'il peut sauter directement à S_T par un tirage unique. C'est
    l'optimisation la plus rentable du moteur : sur un call européen, elle
    évite n_steps colonnes inutiles.
    """

    def __init__(self, fn, needs_path=True, label=""):
        self.fn = fn
        self.needs_path = bool(needs_path)
        self.label = label or getattr(fn, "__name__", "payoff")

    def __call__(self, paths):
        return self.fn(paths)

    def __repr__(self):
        return f"Payoff({self.label}, needs_path={self.needs_path})"


# --- Catalogue de payoffs --------------------------------------------------
def payoff_european(K, option_type="call"):
    """max(S_T - K, 0) ou max(K - S_T, 0)."""
    s = 1.0 if option_type == "call" else -1.0
    _check_type(option_type)
    return Payoff(lambda p: np.maximum(s * (p[:, -1] - K), 0.0),
                  needs_path=False, label=f"european_{option_type}")


def payoff_asian(K, option_type="call", average="arithmetic"):
    """Payoff sur la MOYENNE des fixings (hors t=0)."""
    _check_type(option_type)
    if average not in ("arithmetic", "geometric"):
        raise ValueError("average doit être 'arithmetic' ou 'geometric'.")
    s = 1.0 if option_type == "call" else -1.0

    def fn(p):
        avg = p.mean(axis=1) if average == "arithmetic" \
            else np.exp(np.log(p).mean(axis=1))
        return np.maximum(s * (avg - K), 0.0)

    return Payoff(fn, needs_path=True, label=f"asian_{average}_{option_type}")


def payoff_barrier(K, B, option_type="call", barrier_type="up-and-out"):
    """Vanille conditionnée au franchissement (ou non) de la barrière B."""
    _check_type(option_type)
    if not (barrier_type.startswith(("up", "down"))
            and barrier_type.endswith(("in", "out"))):
        raise ValueError("barrier_type invalide.")
    s = 1.0 if option_type == "call" else -1.0
    up = barrier_type.startswith("up")
    knock_out = barrier_type.endswith("out")

    def fn(p):
        vanilla = np.maximum(s * (p[:, -1] - K), 0.0)
        touched = (p.max(axis=1) >= B) if up else (p.min(axis=1) <= B)
        return np.where(touched ^ knock_out, vanilla, 0.0)

    return Payoff(fn, needs_path=True, label=f"{barrier_type}_{option_type}")


def payoff_digital(K, option_type="call", cash=1.0):
    """Cash-or-nothing : verse `cash` si l'option finit dans la monnaie."""
    _check_type(option_type)
    if option_type == "call":
        fn = lambda p: np.where(p[:, -1] > K, cash, 0.0)
    else:
        fn = lambda p: np.where(p[:, -1] < K, cash, 0.0)
    return Payoff(fn, needs_path=False, label=f"digital_{option_type}")


def payoff_lookback_floating(option_type="call"):
    """Lookback à strike flottant : S_T - min(S) (call), max(S) - S_T (put)."""
    _check_type(option_type)
    if option_type == "call":
        fn = lambda p: p[:, -1] - p.min(axis=1)
    else:
        fn = lambda p: p.max(axis=1) - p[:, -1]
    return Payoff(fn, needs_path=True, label=f"lookback_float_{option_type}")


def _check_type(option_type):
    if option_type not in ("call", "put"):
        raise ValueError("option_type doit être 'call' ou 'put'.")


# ===========================================================================
#  Variables de contrôle
# ===========================================================================
class ControlVariate:
    """Quantité corrélée au payoff dont on connaît l'espérance exacte.

    `values(paths, ctx)` -> réalisations simulées du contrôle (actualisées)
    `mean(ctx)`          -> son espérance exacte sous Q

    Le coefficient beta n'est pas fixé à 1 mais estimé par régression sur
    l'échantillon : beta = Cov(payoff, X) / Var(X). C'est le beta qui minimise
    la variance résiduelle, et la réduction obtenue vaut 1 - rho^2.
    """

    def __init__(self, values, mean, label=""):
        self.values = values
        self.mean = mean
        self.label = label


def control_terminal_spot():
    """Contrôle = S_T actualisé. E[e^{-rT} S_T] = S0 e^{-qT}, exacte.

    Corrélation typique avec un call ATM : ~0.9. Contrôle par défaut des
    européennes.
    """
    return ControlVariate(
        values=lambda p, c: np.exp(-c["r"] * c["T"]) * p[:, -1],
        mean=lambda c: c["S0"] * np.exp(-c["q"] * c["T"]),
        label="terminal_spot",
    )


def control_geometric_asian(K, option_type="call"):
    """Contrôle = asiatique GÉOMÉTRIQUE, dont la formule fermée existe.

    C'est le contrôle de référence pour l'asiatique arithmétique : la
    corrélation entre les deux dépasse 0.99, donc la réduction de variance
    théorique 1 - rho^2 dépasse un facteur 50 en variance.
    """
    geo = payoff_asian(K, option_type, "geometric")
    return ControlVariate(
        values=lambda p, c: np.exp(-c["r"] * c["T"]) * geo(p),
        mean=lambda c: geometric_asian_closed_form(
            c["S0"], K, c["r"], c["sigma"], c["T"], c["n_steps"],
            option_type, c["q"]),
        label="geometric_asian",
    )


def control_vanilla(K, option_type="call"):
    """Contrôle = la VANILLE de mêmes K et T, dont le prix exact est connu.

    Contrôle naturel des options à barrière : la barrière EST la vanille,
    amputée des trajectoires qui franchissent le seuil. Les deux payoffs
    coïncident donc exactement sur toutes les trajectoires survivantes, d'où
    une corrélation élevée.

    C'est le seul contrôle disponible pour une barrière : l'antithétique y est
    quasi inopérant, le payoff n'étant pas monotone en Z (un choc très positif
    désactive l'option, un choc très négatif la laisse hors de la monnaie ;
    les deux extrêmes donnent zéro).
    """
    van = payoff_european(K, option_type)

    def values(p, c):
        # le payoff vanille ne depend que de S_T, present en derniere colonne
        return np.exp(-c["r"] * c["T"]) * van(p)

    def mean(c):
        return BlackScholes(c["S0"], K, c["r"], c["sigma"], c["T"],
                            c["q"]).price(option_type)

    return ControlVariate(values=values, mean=mean, label="vanilla")


# ===========================================================================
#  LE MOTEUR : il ne connaît aucun produit
# ===========================================================================
def mc_price(payoff, S0, r, sigma, T, n_steps=1, n_paths=100_000, q=0.0,
             rng=None, antithetic=False, control=None):
    """Price n'importe quel payoff par simulation sous la mesure risque-neutre.

    payoff  : objet Payoff (cf. catalogue) ou simple callable sur les chemins
    control : ControlVariate optionnelle, ou True pour le contrôle par défaut
              (S_T actualisé)

    Renvoie un MCResult (prix + SE + IC95).
    """
    if not isinstance(payoff, Payoff):
        payoff = Payoff(payoff, needs_path=True, label="custom")

    mu = r - q                      # dérive sous Q
    disc = np.exp(-r * T)

    # --- 1. Simulation : chemin complet seulement si le payoff l'exige ---
    if payoff.needs_path:
        paths = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths,
                                   rng=rng, antithetic=antithetic,
                                   include_spot=False)
    else:
        ST = simulate_gbm_terminal(S0, mu, sigma, T, n_paths,
                                   rng=rng, antithetic=antithetic)
        paths = ST[:, None]         # (n_paths, 1) : vue, coût nul

    # --- 2. Payoff actualisé ---
    disc_payoff = disc * np.asarray(payoff(paths), dtype=float)

    # --- 3. Variable de controle ---
    #
    # ORDRE DES OPERATIONS : APPARIER D'ABORD, REGRESSER ENSUITE.
    #
    # En mode antithetique les lignes i et i+m ne sont pas independantes ; les
    # unites i.i.d. sont les MOYENNES DE PAIRES. La regression qui estime beta
    # doit donc porter sur ces moyennes, exactement comme SE.
    #
    # Estimer beta sur l'echantillon APLATI est un bug silencieux : le prix
    # reste sans biais, car E[X] - EX = 0 quelle que soit la valeur de beta,
    # mais beta minimise la variance du mauvais echantillon. Le controle S_T
    # aplati n'exploite que la composante IMPAIRE de l'alea -- precisement
    # celle que l'antithetique vient d'eliminer -- d'ou un gain marginal
    # quasi nul et la conclusion erronee que les deux techniques ne se
    # composent pas. Sur la moyenne de paire, le controle ne retient au
    # contraire que la composante PAIRE : les deux deviennent complementaires
    # et leurs gains se multiplient.
    if control is not None and control is not False:
        if control is True:
            control = control_terminal_spot()
        ctx = dict(S0=S0, r=r, sigma=sigma, T=T, q=q, n_steps=n_steps)
        X = np.asarray(control.values(paths, ctx), dtype=float)
        EX = float(control.mean(ctx))

        if antithetic:
            m = len(disc_payoff) // 2
            Y = 0.5 * (disc_payoff[:m] + disc_payoff[m:2 * m])
            Xc = 0.5 * (X[:m] + X[m:2 * m])
        else:
            Y, Xc = disc_payoff, X

        cov = np.cov(Y, Xc, ddof=1)
        if cov[1, 1] > 0:
            Y = Y - (cov[0, 1] / cov[1, 1]) * (Xc - EX)

        # Y est deja constitue d'unites i.i.d. : ne pas re-apparier.
        return _summarize(Y, antithetic=False, n_paths=n_paths)

    return _summarize(disc_payoff, antithetic=antithetic)


def bgk_effective_barrier(B, sigma, T, n_steps, up=True):
    """Correction de continuité de Broadie-Glasserman-Kou.

    Le problème : on simule des fixings DISCRETS, donc on manque les
    franchissements survenus entre deux dates. Une option `out` survit dans la
    simulation alors qu'elle aurait dû mourir -> son prix est SURESTIMÉ.

    La correction déplace la barrière de exp(±beta sigma sqrt(dt)),
    beta = 0.5826. Deux usages, souvent confondus :

      - simulation discrète -> prix CONTINU  : rapprocher la barrière du spot
        (up : B e^{-beta...}),  c'est ce que fait cette fonction ;
      - formule continue -> prix d'un contrat à fixings DISCRETS : éloigner la
        barrière du spot (up : B e^{+beta...}).

    Ordre de grandeur : sigma=20%, T=1, 252 fixings -> décalage de 0,73 % du
    niveau de barrière. Sur une up-and-out proche du spot, cela déplace le
    prix de plusieurs pour cent : ce n'est pas un raffinement cosmétique.
    """
    dt = T / n_steps
    shift = BGK_BETA * sigma * np.sqrt(dt)
    return B * np.exp(-shift) if up else B * np.exp(shift)


# ===========================================================================
#  Enveloppes par produit (compatibilité : signatures inchangées)
# ===========================================================================
def mc_european(S0, K, r, sigma, T, n_paths=100_000, option_type="call",
                q=0.0, rng=None, antithetic=False, control_variate=False):
    """Option européenne. Enveloppe de mc_price + payoff_european."""
    return mc_price(payoff_european(K, option_type),
                    S0, r, sigma, T, n_steps=1, n_paths=n_paths, q=q, rng=rng,
                    antithetic=antithetic,
                    control=control_terminal_spot() if control_variate else None)


def mc_asian(S0, K, r, sigma, T, n_steps=252, n_paths=100_000,
             option_type="call", average="arithmetic", q=0.0, rng=None,
             antithetic=False, control_variate=False):
    """Option asiatique. `control_variate=True` active le contrôle géométrique.

    Usage réel : très courant sur les matières premières et le change, car la
    moyenne rend l'option moins manipulable et moins volatile.
    """
    control = None
    if control_variate:
        if average != "arithmetic":
            raise ValueError(
                "Le contrôle géométrique n'a de sens que pour l'asiatique "
                "arithmétique (pour la géométrique, la formule fermée suffit)."
            )
        control = control_geometric_asian(K, option_type)
    return mc_price(payoff_asian(K, option_type, average),
                    S0, r, sigma, T, n_steps=n_steps, n_paths=n_paths, q=q,
                    rng=rng, antithetic=antithetic, control=control)


def mc_barrier(S0, K, B, r, sigma, T, n_steps=252, n_paths=100_000,
               option_type="call", barrier_type="up-and-out", q=0.0, rng=None,
               antithetic=False, continuity_correction=False,
               control_variate=False):
    """Option à barrière.

    `continuity_correction=True` applique Broadie-Glasserman-Kou pour
    approcher le prix à monitoring CONTINU depuis une simulation à fixings
    discrets. Laissé à False par défaut : le contrat de marché est en général
    lui-même à fixings discrets, auquel cas la simulation brute est la bonne
    réponse. À activer pour comparer aux formules fermées de Rubinstein-Reiner.

    RELATION DE PARITÉ (garde-fou) : in + out = vanille, quelle que soit la
    barrière effective utilisée, tant qu'elle est la même des deux côtés.
    """
    B_eff = B
    if continuity_correction:
        B_eff = bgk_effective_barrier(B, sigma, T, n_steps,
                                      up=barrier_type.startswith("up"))
    return mc_price(payoff_barrier(K, B_eff, option_type, barrier_type),
                    S0, r, sigma, T, n_steps=n_steps, n_paths=n_paths, q=q,
                    rng=rng, antithetic=antithetic,
                    control=control_vanilla(K, option_type) if control_variate
                            else None)


def geometric_asian_closed_form(S0, K, r, sigma, T, n_steps, option_type="call", q=0.0):
    """Formule FERMÉE de l'asiatique GÉOMÉTRIQUE (elle existe, contrairement à
    l'arithmétique).

    RÔLE : sert de RÉFÉRENCE EXACTE pour valider le pricer Monte Carlo asiatique,
    et de variable de contrôle pour la version arithmétique.
    """
    # La moyenne géométrique d'un GBM est elle-même log-normale : on connaît
    # exactement la moyenne et la variance de son logarithme.
    # Moyenne discrète sur les instants t_i = i*dt, i = 1..n.
    n = int(n_steps)
    dt = T / n

    # m = E[ln G] et v = Var[ln G]
    m = np.log(S0) + (r - q - 0.5 * sigma**2) * dt * (n + 1) / 2.0
    v = sigma**2 * dt * (n + 1) * (2 * n + 1) / (6.0 * n)
    sg = np.sqrt(v)

    d1 = (m - np.log(K) + v) / sg
    d2 = d1 - sg
    disc = np.exp(-r * T)

    if option_type == "call":
        return disc * (np.exp(m + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return disc * (K * norm.cdf(-d2) - np.exp(m + 0.5 * v) * norm.cdf(-d1))


# ===========================================================================
#  GREEKS par différences finies : générique, donc applicable aux exotiques
# ===========================================================================
def mc_greek_fd(payoff, S0, r, sigma, T, greek="delta", h=None, n_steps=1,
                n_paths=200_000, q=0.0, seed=0, antithetic=False, control=None):
    """Sensibilité par différences finies centrées sur un pricer Monte Carlo.

    TECHNIQUE CLÉ : la MÊME graine pour les deux évaluations (common random
    numbers). Les deux simulations voient alors les mêmes chocs ; le bruit
    commun s'élimine dans la soustraction et seul subsiste l'effet du bump.
    Sans cela, le bruit MC (~0.05) est du même ordre que le signal recherché,
    et l'estimateur est inutilisable.

    `greek` : 'delta' (bump de S0) ou 'vega' (bump de sigma).
    Fonctionne pour N'IMPORTE quel payoff, y compris exotique : c'est la seule
    voie générale, aucune dérivée analytique n'existant pour ces produits.
    """
    if greek not in ("delta", "vega"):
        raise ValueError("greek doit être 'delta' ou 'vega'.")
    if h is None:
        h = 0.5 if greek == "delta" else 0.01

    def price_at(s0, sig):
        return mc_price(payoff, s0, r, sig, T, n_steps=n_steps,
                        n_paths=n_paths, q=q,
                        rng=np.random.default_rng(seed),   # <-- même graine
                        antithetic=antithetic, control=control).price

    if greek == "delta":
        up, dn = price_at(S0 + h, sigma), price_at(S0 - h, sigma)
    else:
        up, dn = price_at(S0, sigma + h), price_at(S0, sigma - h)
    return (up - dn) / (2 * h)


def mc_delta_fd(S0, K, r, sigma, T, n_paths=200_000, h=0.5,
                option_type="call", q=0.0, seed=0):
    """Delta d'une européenne (enveloppe de mc_greek_fd)."""
    return mc_greek_fd(payoff_european(K, option_type), S0, r, sigma, T,
                       greek="delta", h=h, n_paths=n_paths, q=q, seed=seed)


def mc_vega_fd(S0, K, r, sigma, T, n_paths=200_000, h=0.01,
               option_type="call", q=0.0, seed=0):
    """Vega d'une européenne (enveloppe de mc_greek_fd)."""
    return mc_greek_fd(payoff_european(K, option_type), S0, r, sigma, T,
                       greek="vega", h=h, n_paths=n_paths, q=q, seed=seed)
