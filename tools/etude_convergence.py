"""Etude de convergence Monte-Carlo : combien de trajectoires pour coter au cent ?

DEUX CHOSES SONT MESUREES ICI, ET AUCUNE N'EST POSTULEE
-------------------------------------------------------

1. LE NOMBRE DE TRAJECTOIRES NECESSAIRE (colonne "M mesure").
   On ajoute des trajectoires par paquets et, apres chaque paquet, on recalcule
   l'intervalle de confiance sur TOUT ce qui a ete tire depuis le debut. On
   s'arrete au premier paquet ou cet intervalle passe sous le centime. Le M
   rapporte est donc le nombre de trajectoires reellement simulees.

   (Une version anterieure extrapolait M a partir de l'erreur mesuree sur un
   seul echantillon, via la loi en 1/sqrt(M). C'etait correct mais indirect :
   on annoncait un M qu'on n'avait jamais simule. La mesure directe coute
   quelques secondes de plus et ne repose sur aucune hypothese.)

2. LA PENTE DE CONVERGENCE (colonne "pente").
   On mesure SE a quatre tailles d'echantillon et on regresse
   log(SE) sur log(M). La pente obtenue est ce qu'elle est : elle
   n'est ni imposee, ni utilisee dans un calcul ulterieur. Elle sert
   uniquement de diagnostic. La theorie predit 0,50 pour tout estimateur
   Monte-Carlo correctement construit ; le script affiche l'ecart a 0,50 pour
   que le lecteur juge lui-meme de l'accord.

La question posee est celle d'un desk : "combien de temps pour coter au cent ?"
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.black_scholes import BlackScholes
from src.monte_carlo import mc_european, mc_asian, mc_barrier

TOL = 0.01          # demi-largeur d'IC95 visee, en unites de prix (1 cent)

GRAINE_ESTIMATION = 12345   # graine des mesures
GRAINE_REFERENCE = 999      # graine des references : DIFFERENTE, volontairement

Z95 = 1.96          # quantile normal a 95 % : IC95 = Z95 x SE


# ===========================================================================
#  Valeur de reference
# ===========================================================================
def reference_par_lots(pricer, n_lots=10, M_lot=100_000,
                       graine=GRAINE_REFERENCE):
    """Valeur de reference haute precision, par moyenne de lots independants.

    Trois raisons de proceder ainsi plutot que par un unique gros appel :

    1. INDEPENDANCE. La graine differe de celle des mesures. Sinon la reference
       partage ses trajectoires avec les estimations qu'elle est censee valider
       (numpy remplit ligne par ligne : les M premieres lignes d'un tirage de
       2M sont exactement le tirage de taille M), et le controle "la reference
       tombe dans l'IC" devient partiellement auto-realisateur.

    2. MEMOIRE. Un appel a M = 10^6 et n = 252 demanderait 2 Go d'un seul
       tenant. Dix lots de 10^5 n'en demandent jamais plus de 200 Mo.

    3. INCERTITUDE MESUREE. Une reference est elle-meme une estimation. Son
       incertitude est ici mesuree par la dispersion des lots (batch means).

    Renvoie (moyenne, demi-largeur IC95 de la reference).
    """
    prix = np.array([pricer(M_lot, np.random.default_rng(graine + k))
                     for k in range(n_lots)])
    return prix.mean(), Z95 * prix.std(ddof=1) / np.sqrt(n_lots)


# ===========================================================================
#  Mesure directe du nombre de trajectoires necessaire
# ===========================================================================
def M_pour_atteindre(pricer, tol=TOL, M_max=20_000_000,
                     graine=GRAINE_ESTIMATION):
    """Simule jusqu'a ce que l'IC95 cumule passe sous `tol`. Aucune extrapolation.

    Comment ca marche, en trois phrases :
      - on tire un paquet de trajectoires ;
      - on met a jour trois compteurs (nombre de tirages, somme, somme des
        carres) qui suffisent a recalculer moyenne et ecart-type de TOUT ce
        qui a ete tire depuis le debut ;
      - des que 1,96 x ecart-type / sqrt(nombre de tirages) descend sous la
        tolerance, on s'arrete et on rapporte le nombre de trajectoires
        simulees.

    Les compteurs evitent de garder les trajectoires en memoire : seuls trois
    nombres passent d'un paquet a l'autre.

    Taille des paquets : petite au debut (500 trajectoires : resolution fine quand la reponse est
    petite), puis proportionnelle a ce qui est deja tire (pas de temps perdu en
    appels quand la reponse se compte en millions). C'est un compromis
    vitesse/resolution ; il ne change pas le resultat, seulement la finesse
    avec laquelle on le localise.

    Renvoie (M_total, prix, ci95, temps).
    """
    n = 0            # nombre d'unites i.i.d. cumulees
    somme = 0.0
    somme_carres = 0.0
    M_total = 0      # nombre de trajectoires cumulees
    k = 0
    t0 = time.perf_counter()

    while M_total < M_max:
        # taille du paquet, arrondie a un nombre pair : le mode antithetique
        # exige des trajectoires appariees
        M_lot = int(max(500, min(200_000, M_total // 8)))
        M_lot -= M_lot % 2
        res = pricer(M_lot, rng=np.random.default_rng(graine + k))
        x = res.sample
        n += x.size
        somme += float(x.sum())
        somme_carres += float(np.square(x).sum())
        M_total += res.n_paths
        k += 1

        if n > 1:
            var = (somme_carres - somme * somme / n) / (n - 1)
            ci = Z95 * np.sqrt(max(var, 0.0) / n)
            if ci <= tol:
                return M_total, somme / n, ci, time.perf_counter() - t0

    var = (somme_carres - somme * somme / n) / (n - 1)
    return M_total, somme / n, Z95 * np.sqrt(var / n), time.perf_counter() - t0


# ===========================================================================
#  Pente de convergence (diagnostic, jamais reinjectee dans un calcul)
# ===========================================================================
def pente_mesuree(pricer, tailles, graine=GRAINE_ESTIMATION):
    """Regresse log(SE) sur log(M) et renvoie (pente, liste des SE).

    La pente n'est PAS contrainte : on l'affiche telle quelle, avec son ecart a
    la valeur theorique 0,50, pour que le lecteur juge de l'accord.
    """
    ses = [pricer(M, rng=np.random.default_rng(graine)).std_error
           for M in tailles]
    pente = -np.polyfit(np.log(tailles), np.log(ses), 1)[0]
    return pente, ses


# ===========================================================================
#  Etude d'un produit
# ===========================================================================
def etude(nom, pricer, tailles, methodes, ref=None, ref_label="", ref_ci=None):
    print("=" * 100)
    print(f"{nom}   |   cible : IC95 = +/- {TOL:.2f}")
    if ref is not None:
        marge = "" if ref_ci is None else f" +/- {ref_ci:.4f}"
        print(f"reference : {ref:.4f}{marge}  ({ref_label})")
        if ref_ci is not None and ref_ci > TOL:
            print(f"  NOTE : la reference est moins precise que la cible "
                  f"({ref_ci:.4f} > {TOL:.2f}).")
    print("=" * 100)
    print(f"{'methode':<24}{'pente':>7}{'ecart a 0,50':>14}"
          f"{'M mesure':>13}{'prix':>10}{'IC95':>9}{'temps':>9}{'gain':>8}")

    base_M = None
    for lib, kw in methodes:
        p = (lambda M, rng, kw=kw: pricer(M, rng=rng, **kw))
        pente, _ = pente_mesuree(p, tailles)
        M_mes, prix, ci, t = M_pour_atteindre(p)
        if base_M is None:
            base_M = M_mes
        print(f"{lib:<24}{pente:>7.3f}{pente-0.50:>+14.3f}"
              f"{M_mes:>13,}{prix:>10.4f}{ci:>9.5f}{t:>8.1f}s"
              f"{base_M/M_mes:>7.1f}x")
    print()


# ===========================================================================
#  Programme
# ===========================================================================
if __name__ == "__main__":
    S, K, r, sig, T = 100.0, 100.0, 0.05, 0.20, 1.0
    B = 120.0
    N_FIX = 252

    M4 = [("Monte-Carlo seul",       dict()),
          ("+ antithetique",         dict(antithetic=True)),
          ("+ variable de controle", dict(control_variate=True)),
          ("+ les deux",             dict(antithetic=True, control_variate=True))]

    etude("CALL EUROPEEN  (n=1)",
          lambda M, rng, **kw: mc_european(S, K, r, sig, T, M, rng=rng, **kw),
          [50_000, 100_000, 200_000, 400_000], M4,
          ref=BlackScholes(S, K, r, sig, T).price(),
          ref_label="Black-Scholes exact, aucune incertitude")

    ref_a, ci_a = reference_par_lots(
        lambda M, rng: mc_asian(S, K, r, sig, T, N_FIX, M, rng=rng,
                                control_variate=True).price)
    etude(f"ASIATIQUE ARITHMETIQUE  (n={N_FIX})",
          lambda M, rng, **kw: mc_asian(S, K, r, sig, T, N_FIX, M, rng=rng, **kw),
          [25_000, 50_000, 100_000, 200_000], M4,
          ref=ref_a, ref_ci=ci_a,
          ref_label="MC + controle geometrique, 10 lots de 100 000")

    # Barriere out : aucun controle efficace n'existe (rapport, section 8.2.3).
    ref_o, ci_o = reference_par_lots(
        lambda M, rng: mc_barrier(S, K, B, r, sig, T, N_FIX, M, rng=rng).price)
    etude(f"BARRIERE up-and-out B={B:.0f}  (n={N_FIX})",
          lambda M, rng, **kw: mc_barrier(S, K, B, r, sig, T, N_FIX, M,
                                          rng=rng, **kw),
          [25_000, 50_000, 100_000, 200_000], M4[:3],
          ref=ref_o, ref_ci=ci_o, ref_label="MC seul, 10 lots de 100 000")

    ref_i, ci_i = reference_par_lots(
        lambda M, rng: mc_barrier(S, K, B, r, sig, T, N_FIX, M, rng=rng,
                                  barrier_type="up-and-in",
                                  control_variate=True).price)
    etude(f"BARRIERE up-and-in B={B:.0f}  (n={N_FIX})",
          lambda M, rng, **kw: mc_barrier(S, K, B, r, sig, T, N_FIX, M, rng=rng,
                                          barrier_type="up-and-in", **kw),
          [25_000, 50_000, 100_000, 200_000], [M4[0], M4[2]],
          ref=ref_i, ref_ci=ci_i,
          ref_label="MC + controle vanille, 10 lots de 100 000")

    van = BlackScholes(S, K, r, sig, T).price()
    print(f"parite in/out : {ref_o:.4f} + {ref_i:.4f} = {ref_o+ref_i:.4f}"
          f"   vanille = {van:.4f}   ecart = {ref_o+ref_i-van:+.4f}"
          f"   (marge cumulee +/- {ci_o+ci_i:.4f})")
    print("NOTE : cet ecart est quasi nul par construction. Le controle vanille "
          "sur la 'in' revient")
    print("       algebriquement a 'vanille moins out'. C'est un controle "
          "d'implementation, pas une")
    print("       verification independante (rapport, section 8.2.3).")
