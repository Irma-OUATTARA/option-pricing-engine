"""Tests des corrections apportées au moteur (refonte payoff, BGK, contrôles).

Chaque test ici correspond à un écart identifié entre le rapport et le code,
ou à une correction apportée. Ils servent de garde-fou : si quelqu'un défait
l'une de ces corrections, un test tombe.
"""
import numpy as np
import pytest
from scipy.stats import norm

from src.black_scholes import BlackScholes
from src.simulation import simulate_gbm_paths, draw_normals
from src.monte_carlo import (mc_price, mc_european, mc_asian, mc_barrier,
                             mc_greek_fd,
                             payoff_european, payoff_digital,
                             payoff_lookback_floating,
                             bgk_effective_barrier, BGK_BETA)


# --- 1. Le payoff est bien un argument du moteur -------------------------
def test_moteur_accepte_un_payoff_arbitraire():
    """Un lambda quelconque doit pouvoir être pricé sans toucher au moteur.

    On price ici un 'power call' max(S_T^1 - K, 0) écrit à la volée, qui
    n'existe nulle part dans le catalogue : c'est tout l'intérêt de la
    refonte. On vérifie qu'il retombe sur Black-Scholes.
    """
    bs = BlackScholes(100, 100, 0.05, 0.20, 1.0).price("call")
    res = mc_price(lambda p: np.maximum(p[:, -1] - 100, 0.0),
                   100, 0.05, 0.20, 1.0, n_steps=1, n_paths=200_000,
                   rng=np.random.default_rng(0))
    assert abs(res.price - bs) < res.ci95 * 2


def test_payoff_europeen_saute_la_construction_du_chemin():
    """needs_path=False sur une européenne : on ne simule que S_T."""
    assert payoff_european(100, "call").needs_path is False
    assert payoff_digital(100, "call").needs_path is False
    assert payoff_lookback_floating("call").needs_path is True


# --- 2. Produits ajoutés grâce à la refonte ------------------------------
def test_digitale_contre_formule_fermee():
    """Cash-or-nothing call = e^{-rT} N(d2), référence analytique exacte."""
    S, K, r, sig, T = 100.0, 100.0, 0.05, 0.20, 1.0
    exact = np.exp(-r * T) * norm.cdf(BlackScholes(S, K, r, sig, T).d2)
    res = mc_price(payoff_digital(K, "call"), S, r, sig, T,
                   n_paths=400_000, rng=np.random.default_rng(7))
    assert abs(res.price - exact) < res.ci95 * 2


def test_lookback_plus_chere_que_la_vanille():
    """Acheter au plus bas vaut forcément plus qu'acheter au strike ATM."""
    vanille = BlackScholes(100, 100, 0.05, 0.20, 1.0).price("call")
    look = mc_price(payoff_lookback_floating("call"), 100, 0.05, 0.20, 1.0,
                    n_steps=252, n_paths=50_000,
                    rng=np.random.default_rng(8)).price
    assert look > vanille


# --- 3. Correction de continuité de Broadie-Glasserman-Kou ---------------
def test_bgk_deplace_la_barriere_dans_le_bon_sens():
    """Barrière haute -> on la RAPPROCHE du spot (simulation -> continu)."""
    B = 120.0
    up = bgk_effective_barrier(B, 0.20, 1.0, 252, up=True)
    down = bgk_effective_barrier(80.0, 0.20, 1.0, 252, up=False)
    assert up < B                    # abaissée
    assert down > 80.0               # relevée
    shift = BGK_BETA * 0.20 * np.sqrt(1.0 / 252)
    assert up == pytest.approx(B * np.exp(-shift))


def test_bgk_abaisse_le_prix_dune_up_and_out():
    """Barrière plus proche -> plus de désactivations -> option moins chère.

    C'est la correction du biais : en monitoring discret on manquait des
    franchissements, donc on surestimait l'option 'out'.
    """
    kw = dict(n_steps=50, n_paths=200_000, barrier_type="up-and-out")
    brut = mc_barrier(100, 100, 115, 0.05, 0.20, 1.0,
                      rng=np.random.default_rng(11), **kw).price
    corr = mc_barrier(100, 100, 115, 0.05, 0.20, 1.0,
                      rng=np.random.default_rng(11),
                      continuity_correction=True, **kw).price
    assert corr < brut


def test_parite_in_out_tient_avec_correction():
    """in + out = vanille reste vrai tant que la barrière effective est la
    même des deux côtés."""
    bs = BlackScholes(100, 100, 0.05, 0.20, 1.0).price("call")
    kw = dict(n_steps=100, n_paths=200_000, continuity_correction=True)
    out = mc_barrier(100, 100, 120, 0.05, 0.20, 1.0,
                     barrier_type="up-and-out",
                     rng=np.random.default_rng(12), **kw).price
    inn = mc_barrier(100, 100, 120, 0.05, 0.20, 1.0,
                     barrier_type="up-and-in",
                     rng=np.random.default_rng(12), **kw).price
    assert (out + inn) == pytest.approx(bs, abs=0.1)


# --- 4. Variable de contrôle géométrique sur l'asiatique -----------------
def test_controle_geometrique_reduit_lerreur_asiatique():
    """La corrélation géométrique/arithmétique dépasse 0.99 : le gain doit
    être franc, pas marginal."""
    kw = dict(n_steps=50, n_paths=100_000, average="arithmetic")
    base = mc_asian(100, 100, 0.05, 0.20, 1.0,
                    rng=np.random.default_rng(13), **kw)
    ctrl = mc_asian(100, 100, 0.05, 0.20, 1.0,
                    rng=np.random.default_rng(13),
                    control_variate=True, **kw)
    assert ctrl.std_error < base.std_error / 5
    assert abs(ctrl.price - base.price) < base.ci95 * 2


def test_controle_geometrique_refuse_sur_asiatique_geometrique():
    with pytest.raises(ValueError):
        mc_asian(100, 100, 0.05, 0.20, 1.0, average="geometric",
                 control_variate=True)


# --- 5. Greeks génériques, applicables aux exotiques ---------------------
def test_delta_dune_asiatique_est_calculable_et_borne():
    """Avant la refonte, mc_delta_fd ne savait traiter que l'européenne."""
    from src.monte_carlo import payoff_asian
    delta = mc_greek_fd(payoff_asian(100, "call"), 100, 0.05, 0.20, 1.0,
                        greek="delta", n_steps=50, n_paths=100_000, seed=3)
    assert 0.0 < delta < 1.0


def test_vega_dune_asiatique_est_positif():
    from src.monte_carlo import payoff_asian
    vega = mc_greek_fd(payoff_asian(100, "call"), 100, 0.05, 0.20, 1.0,
                       greek="vega", n_steps=50, n_paths=100_000, seed=4)
    assert vega > 0.0


# --- 6. Conventions de simulation ----------------------------------------
def test_include_spot_change_bien_la_forme():
    kw = dict(S0=100, mu=0.05, sigma=0.20, T=1.0, n_steps=10, n_paths=8)
    assert simulate_gbm_paths(**kw, include_spot=True).shape == (8, 11)
    assert simulate_gbm_paths(**kw, include_spot=False).shape == (8, 10)


def test_colonne_zero_vaut_le_spot():
    p = simulate_gbm_paths(100, 0.05, 0.20, 1.0, 10, 8,
                           rng=np.random.default_rng(0), include_spot=True)
    assert np.allclose(p[:, 0], 100.0)


def test_antithetique_apparie_i_et_i_plus_m():
    """L'estimateur de SE suppose cet appariement exact."""
    Z = draw_normals((6, 3), rng=np.random.default_rng(0), antithetic=True)
    assert np.allclose(Z[:3], -Z[3:])


def test_antithetique_refuse_un_nombre_impair():
    with pytest.raises(ValueError):
        draw_normals(101, rng=np.random.default_rng(0), antithetic=True)


# --- 7. Le remplacement de scipy.stats.norm par ndtr est exact -----------
def test_ndtr_donne_les_memes_valeurs_que_scipy_stats():
    """Optimisation x114 : elle ne doit rien changer aux valeurs."""
    from src.black_scholes import _cdf, _pdf
    x = np.linspace(-6, 6, 501)
    assert np.allclose(_cdf(x), norm.cdf(x), atol=1e-15)
    assert np.allclose(_pdf(x), norm.pdf(x), atol=1e-15)


# ===========================================================================
#  Non-regression : les deux techniques de reduction de variance se composent
# ===========================================================================
def test_antithetique_et_controle_se_composent():
    """Combiner antithetique et variable de controle doit faire MIEUX que
    chacune des deux seule.

    C'est le garde-fou du bug corrige en CORRECTIONS.md §9 : beta etait estime
    sur l'echantillon aplati au lieu des moyennes de paires, ce qui faisait
    chuter le gain combine SOUS celui du controle seul. Aucun test existant ne
    pouvait le voir, parce que le prix restait juste (l'estimateur a variable de
    controle est sans biais quel que soit beta) : seule la PRECISION etait
    degradee.
    """
    kw = dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, n_paths=200_000)

    def se(**extra):
        return mc_european(kw["S0"], kw["K"], kw["r"], kw["sigma"], kw["T"],
                           kw["n_paths"], rng=np.random.default_rng(12345),
                           **extra).std_error

    se_seul = se()
    se_anti = se(antithetic=True)
    se_ctrl = se(control_variate=True)
    se_deux = se(antithetic=True, control_variate=True)

    # chaque technique apporte quelque chose
    assert se_anti < se_seul
    assert se_ctrl < se_seul
    # et surtout : la combinaison bat STRICTEMENT chacune des deux
    assert se_deux < se_ctrl, (
        f"combine ({se_deux:.5f}) devrait battre le controle seul "
        f"({se_ctrl:.5f}) : beta est-il estime sur les moyennes de paires ?"
    )
    assert se_deux < se_anti


def test_n_paths_rapporte_est_bien_le_nombre_de_trajectoires():
    """En mode antithetique + controle, l'echantillon i.i.d. transmis a
    _summarize est constitue de M/2 moyennes de paires. MCResult.n_paths doit
    neanmoins rapporter M, sans quoi tout comptage de trajectoires (cf.
    tools/chiffres_rapport.py) se decale d'un facteur 2.
    """
    for extra in ({}, {"antithetic": True}, {"control_variate": True},
                  {"antithetic": True, "control_variate": True}):
        res = mc_european(100.0, 100.0, 0.05, 0.20, 1.0, 50_000,
                          rng=np.random.default_rng(7), **extra)
        assert res.n_paths == 50_000, f"n_paths faux pour {extra}"


# ===========================================================================
#  Greeks de l'arbre binomial : lus sur le treillis, valides contre BS
# ===========================================================================
def test_greeks_arbre_concordent_avec_black_scholes():
    """Sur une europeenne, delta/gamma/theta lus sur le treillis doivent
    converger vers les formules fermees de Black-Scholes."""
    from src.binomial import BinomialTree
    bs = BlackScholes(100.0, 100.0, 0.05, 0.20, 1.0)
    g = BinomialTree(100.0, 100.0, 0.05, 0.20, 1.0, n_steps=2000).greeks("call", "european")
    assert g["delta"] == pytest.approx(bs.delta("call"), abs=1e-3)
    assert g["gamma"] == pytest.approx(bs.gamma(), abs=1e-3)
    assert g["theta"] == pytest.approx(bs.theta("call"), abs=1e-2)


def test_greeks_arbre_exige_deux_pas():
    from src.binomial import BinomialTree
    with pytest.raises(ValueError):
        BinomialTree(100.0, 100.0, 0.05, 0.20, 1.0, n_steps=1).greeks()
